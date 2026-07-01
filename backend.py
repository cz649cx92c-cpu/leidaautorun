#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import core_backend as core  # type: ignore # noqa: E402
except ImportError:
    import autorun.backend as core  # type: ignore # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent
VENDOR_ROOT = PROJECT_ROOT / "vendor"
SHADOW_ODIN_ROOT = VENDOR_ROOT / "odin_ros_driver"
SHADOW_ODIN_CONFIG = SHADOW_ODIN_ROOT / "config" / "control_command.yaml"
SHADOW_ODIN_LAUNCH = SHADOW_ODIN_ROOT / "launch_ROS2" / "odin1_ros2.launch.py"
SHADOW_ODIN_PACKAGE_ROS2 = SHADOW_ODIN_ROOT / "package_ros2.xml"
SHADOW_ODIN_PACKAGE = SHADOW_ODIN_ROOT / "package.xml"
MISSIONS_DIR = PROJECT_ROOT / "missions"
LOG_DIR = PROJECT_ROOT / "logs"
ROS_LOG_DIR = LOG_DIR / "ros"
RUNTIME_DIR = PROJECT_ROOT / "runtime"
TEMP_CONFIG_DIR = RUNTIME_DIR / "configs"
SHADOW_COLCON_WS = RUNTIME_DIR / "shadow_ros2_ws"
SHADOW_COLCON_SRC = SHADOW_COLCON_WS / "src"
SHADOW_COLCON_INSTALL = SHADOW_COLCON_WS / "install" / "setup.bash"
SHADOW_HOST_SDK_BIN = SHADOW_COLCON_WS / "install" / "lib" / "odin_ros_driver" / "host_sdk_sample"
SHADOW_HOST_SDK_BUILD_BIN = SHADOW_COLCON_WS / "build" / "odin_ros_driver" / "host_sdk_sample"
LOCAL_ROW_RUNNER = PROJECT_ROOT / "plant_lidar_centerline_ros_follower.py"
LOCAL_ROW_CONFIG = PROJECT_ROOT / "gui_settings.json"
LIDAR_DRIVER_ROOT = Path("/home/orangepi/ugv")
LIDAR_DRIVER_BIN = LIDAR_DRIVER_ROOT / "install" / "lidar_pkg" / "lib" / "lidar_pkg" / "lidar_node"
LIDAR_DRIVER_PARAMS = LIDAR_DRIVER_ROOT / "install" / "lidar_pkg" / "share" / "lidar_pkg" / "config" / "lidar_params.yaml"

for path in (MISSIONS_DIR, LOG_DIR, ROS_LOG_DIR, RUNTIME_DIR, TEMP_CONFIG_DIR, VENDOR_ROOT, SHADOW_COLCON_SRC):
    path.mkdir(parents=True, exist_ok=True)

core.PROJECT_ROOT = PROJECT_ROOT
core.MISSIONS_DIR = MISSIONS_DIR
core.LOG_DIR = LOG_DIR


class Ros2NodeThread:
    _init_lock = threading.Lock()
    _refcount = 0

    def __init__(self, node_name: str) -> None:
        with self._init_lock:
            if not rclpy.ok():
                rclpy.init(args=None)
            self.__class__._refcount += 1
        self.node = Node(node_name)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, name=f"{node_name}-spin", daemon=True)
        self.thread.start()

    def close(self) -> None:
        try:
            self.executor.shutdown()
        except Exception:
            pass
        try:
            self.executor.remove_node(self.node)
        except Exception:
            pass
        try:
            self.node.destroy_node()
        except Exception:
            pass
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        with self._init_lock:
            self.__class__._refcount = max(0, self.__class__._refcount - 1)
            if self.__class__._refcount == 0 and rclpy.ok():
                rclpy.shutdown()


def install_signal_handlers() -> None:
    def _handler(signum, frame):
        del signum, frame
        core.STOP_REQUESTED = True
        core.log("Stop requested. Finishing the current stage safely...")

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def _replace_yaml_scalar(text: str, key: str, value: str) -> str:
    pattern = rf"(^\s*{re.escape(key)}:\s*).*$"
    repl = rf"\g<1>{value}"
    new_text, count = re.subn(pattern, repl, text, flags=re.MULTILINE)
    if count == 0:
        raise RuntimeError(f"Key not found in Odin config: {key}")
    return new_text


def sync_shadow_package() -> None:
    if SHADOW_ODIN_PACKAGE_ROS2.exists():
        shutil.copy2(SHADOW_ODIN_PACKAGE_ROS2, SHADOW_ODIN_PACKAGE)
    _ensure_shadow_workspace_built()
    core.log(f"Shadow odin_ros_driver prepared: {SHADOW_ODIN_ROOT}")


def _shadow_package_mtime() -> float:
    latest = 0.0
    for path in SHADOW_ODIN_ROOT.rglob("*"):
        if path.is_file():
            try:
                latest = max(latest, path.stat().st_mtime)
            except OSError:
                continue
    return latest


def _valid_shadow_binary(path: Path) -> bool:
    try:
        return path.exists() and path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _repair_shadow_host_sdk_install() -> bool:
    if not _valid_shadow_binary(SHADOW_HOST_SDK_BUILD_BIN):
        return False
    SHADOW_HOST_SDK_BIN.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SHADOW_HOST_SDK_BUILD_BIN, SHADOW_HOST_SDK_BIN)
    try:
        SHADOW_HOST_SDK_BIN.chmod(0o755)
    except OSError:
        pass
    return _valid_shadow_binary(SHADOW_HOST_SDK_BIN)


def _reset_shadow_workspace() -> None:
    for path in (
        SHADOW_COLCON_WS / "build",
        SHADOW_COLCON_WS / "install",
        SHADOW_COLCON_WS / "log",
    ):
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)


def _ensure_shadow_workspace_built() -> None:
    package_link = SHADOW_COLCON_SRC / "odin_ros_driver"
    if package_link.is_symlink():
        if package_link.resolve() != SHADOW_ODIN_ROOT.resolve():
            package_link.unlink()
    elif package_link.exists():
        if package_link.is_dir():
            shutil.rmtree(package_link)
        else:
            package_link.unlink()
    if not package_link.exists():
        package_link.symlink_to(SHADOW_ODIN_ROOT, target_is_directory=True)

    source_mtime = _shadow_package_mtime()
    binary_mtime = SHADOW_HOST_SDK_BIN.stat().st_mtime if _valid_shadow_binary(SHADOW_HOST_SDK_BIN) else 0.0
    if _valid_shadow_binary(SHADOW_HOST_SDK_BIN) and binary_mtime >= source_mtime:
        return

    if _repair_shadow_host_sdk_install():
        repaired_mtime = SHADOW_HOST_SDK_BIN.stat().st_mtime
        if repaired_mtime >= source_mtime:
            core.log(f"Repaired empty shadow host_sdk_sample install from build artifact: {SHADOW_HOST_SDK_BIN}")
            return

    build_log = LOG_DIR / f"shadow_build_{time.strftime('%Y%m%d_%H%M%S')}.log"
    core.log("Building patched shadow odin_ros_driver workspace for relocalization startup...")
    cmd = [
        "/bin/bash",
        "-lc",
        (
            f"source /opt/ros/humble/setup.bash && "
            f"cd {SHADOW_COLCON_WS} && "
            "colcon build --packages-select odin_ros_driver --merge-install"
        ),
    ]
    result = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    build_log.write_text(result.stdout or "", encoding="utf-8")
    if result.returncode != 0:
        output_text = result.stdout or ""
        if "CMakeCache.txt directory" in output_text and "is different than the directory" in output_text:
            core.log("Detected shadow workspace CMake cache mismatch. Resetting shadow build workspace and retrying once...")
            _reset_shadow_workspace()
            retry_result = subprocess.run(
                cmd,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            retry_output = retry_result.stdout or ""
            build_log.write_text(
                (output_text + "\n\n[retry after shadow workspace reset]\n" + retry_output).strip() + "\n",
                encoding="utf-8",
            )
            if retry_result.returncode == 0:
                if not _valid_shadow_binary(SHADOW_HOST_SDK_BIN):
                    if not _repair_shadow_host_sdk_install():
                        raise RuntimeError(
                            "Shadow odin_ros_driver rebuild succeeded after reset, but host_sdk_sample install is invalid. "
                            f"See {build_log}"
                        )
                core.log("Shadow workspace rebuilt successfully after cache reset.")
                return
        raise RuntimeError(f"Failed to build patched shadow odin_ros_driver. See {build_log}")
    if not _valid_shadow_binary(SHADOW_HOST_SDK_BIN):
        if not _repair_shadow_host_sdk_install():
            raise RuntimeError(
                "Shadow odin_ros_driver build finished but host_sdk_sample install is invalid. "
                f"See {build_log}"
            )
        core.log(f"Shadow host_sdk_sample install repaired after build: {SHADOW_HOST_SDK_BIN}")
    core.log(f"Patched shadow odin_ros_driver built successfully. Build log: {build_log}")


def write_odin_config(mode: int, *, map_path: Path | None = None, map_name: str = "", recorddata: bool = False) -> Path:
    sync_shadow_package()
    text = SHADOW_ODIN_CONFIG.read_text(encoding="utf-8")
    text = _replace_yaml_scalar(text, "custom_map_mode", str(mode))
    text = _replace_yaml_scalar(text, "recorddata", "1" if recorddata else "0")
    text = _replace_yaml_scalar(text, "use_host_ros_time", "0")
    text = _replace_yaml_scalar(text, "sendimu", "1")
    text = _replace_yaml_scalar(text, "enable_imu_smooth", "0")
    text = _replace_yaml_scalar(text, "imu_smooth_frequency", "400")
    text = _replace_yaml_scalar(text, "showpath", "1")
    text = _replace_yaml_scalar(text, "showcamerapose", "1")
    if mode == 1:
        map_dir = PROJECT_ROOT / "maps" / map_name
        map_dir.mkdir(parents=True, exist_ok=True)
        text = _replace_yaml_scalar(text, "relocalization_map_abs_path", '""')
        text = _replace_yaml_scalar(text, "mapping_result_dest_dir", f'"{map_dir}"')
        text = _replace_yaml_scalar(text, "mapping_result_file_name", f'"{map_name}.bin"')
        text = _replace_yaml_scalar(text, "resetalgo", "1")
    elif mode == 2:
        if map_path is None:
            raise RuntimeError("Relocalization mode requires a map path.")
        text = _replace_yaml_scalar(text, "relocalization_map_abs_path", f'"{map_path}"')
        text = _replace_yaml_scalar(text, "mapping_result_dest_dir", '""')
        text = _replace_yaml_scalar(text, "mapping_result_file_name", '""')
        # Force the device SLAM/relocalization state to reset before each
        # relocalization session. Without this, repeated relocalization in the
        # same boot can get stuck waiting for a valid pose until the device is
        # power-cycled.
        text = _replace_yaml_scalar(text, "resetalgo", "1")
    temp_path = TEMP_CONFIG_DIR / f"odin_mode_{mode}_{int(time.time() * 1000)}.yaml"
    temp_path.write_text(text, encoding="utf-8")
    return temp_path


def set_shadow_recorddata(enabled: bool) -> None:
    sync_shadow_package()
    text = SHADOW_ODIN_CONFIG.read_text(encoding="utf-8")
    text = _replace_yaml_scalar(text, "recorddata", "1" if enabled else "0")
    SHADOW_ODIN_CONFIG.write_text(text, encoding="utf-8")
    core.log(f"Shadow Odin recorddata set to {'1' if enabled else '0'}.")


def request_map_save(target_file: Path, timeout_sec: float = 25.0, raw_log: Path | None = None) -> bool:
    command_file = Path("/tmp/odin_command.txt")

    def _send_save_command() -> None:
        command_file.write_text("set save_map 1\n", encoding="utf-8")

    _send_save_command()
    core.log("Sent Odin save_map=1 command.")
    deadline = time.monotonic() + timeout_sec
    last_size = -1
    stable_hits = 0
    last_send_at = time.monotonic()
    command_ack = False
    transfer_started = False
    last_status_log_at = 0.0
    resend_interval_sec = 4.0
    post_ack_resend_grace_sec = 6.0
    while time.monotonic() < deadline:
        now = time.monotonic()
        if raw_log is not None and raw_log.exists():
            try:
                log_text = raw_log.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                log_text = ""
            if "Successfully set save_map = 1" in log_text:
                command_ack = True
            if "Map is saved on device" in log_text or "map get start success" in log_text:
                transfer_started = True
        if not command_ack and not command_file.exists():
            command_ack = True
            core.log("Odin consumed the save_map command file.")
        if not command_ack and (now - last_send_at) >= 3.0:
            _send_save_command()
            last_send_at = now
            core.log("Resent Odin save_map=1 command.")
        elif command_ack and not transfer_started and (now - last_send_at) >= post_ack_resend_grace_sec:
            _send_save_command()
            last_send_at = now
            core.log("Save command was acknowledged but map export has not started yet. Resending save_map=1.")
        if target_file.exists():
            size = target_file.stat().st_size
            if size > 0 and size == last_size:
                stable_hits += 1
            else:
                stable_hits = 0
            last_size = size
            if size > 0 and stable_hits >= 3:
                core.log(f"Map file saved: {target_file} ({size} bytes)")
                return True
            if transfer_started and size > 0 and stable_hits >= 1:
                core.log(f"Map file transfer started and file is present: {target_file} ({size} bytes)")
                return True
        if not transfer_started and (now - last_status_log_at) >= resend_interval_sec:
            status = "acknowledged" if command_ack else "pending"
            core.log(f"Waiting for Odin map export to start ({status}); current target: {target_file}")
            last_status_log_at = now
        time.sleep(1.0)
    return target_file.exists() and target_file.stat().st_size > 0


def cleanup_stale_odin_processes() -> None:
    patterns = [
        "install/lidar_pkg/lib/lidar_pkg/lidar_node",
        " lidar_node",
        "host_sdk_sample",
        "odin1_ros2.launch.py",
        "pcd2depth_node",
        "pcd2depth_ros2_node",
        "cloud_reprojection_node",
        "cloud_reprojection_ros2_node",
        "image_overlay_node",
        "rviz2",
    ]
    found: list[str] = []
    for pattern in patterns:
        result = subprocess.run(
            ["pgrep", "-af", pattern],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        lines = [
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip() and "pgrep -af" not in line and "autorun_final/backend.py" not in line
        ]
        if not lines:
            continue
        found.append(pattern)
        core.log(f"Cleaning stale process pattern '{pattern}': {len(lines)} match(es)")
        subprocess.run(["pkill", "-f", pattern], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if found:
        time.sleep(2.0)
        core.log("Stale Odin/ROS processes were cleaned before startup.")
    else:
        core.log("No stale Odin/ROS processes were found before startup.")


class OdinConfigOverride:
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self.target_config = SHADOW_ODIN_CONFIG

    def apply(self) -> None:
        if not self.target_config.exists():
            raise RuntimeError(f"Shadow Odin config file does not exist: {self.target_config}")
        if not os.access(self.target_config, os.W_OK):
            raise RuntimeError(f"Shadow Odin config file is not writable: {self.target_config}")
        shutil.copy2(self.config_path, self.target_config)
        core.log(f"Odin config override applied: {self.config_path}")

    def restore(self) -> None:
        return


class OdinLaunchProcess(core.ManagedProcess):
    def __init__(self, config_path: Path, log_path: Path) -> None:
        del config_path
        setup_bash = SHADOW_COLCON_INSTALL
        cmd = [
            "/bin/bash",
            "-lc",
            (
                f"source /opt/ros/humble/setup.bash && "
                f"source {setup_bash} && "
                f"export ROS_LOG_DIR={ROS_LOG_DIR} && "
                f"exec ros2 launch odin_ros_driver odin1_ros2.launch.py "
                f"config_file:={SHADOW_ODIN_CONFIG} start_rviz:=false"
            ),
        ]
        super().__init__(cmd, PROJECT_ROOT, log_path)


class LocalizationSession:
    def __init__(self, map_path: Path, *, viz: str = "off") -> None:
        del viz
        self.map_path = map_path
        self.raw_log = LOG_DIR / f"localization_{time.strftime('%Y%m%d_%H%M%S')}.log"
        self.config_path = write_odin_config(2, map_path=map_path)
        self.override = OdinConfigOverride(self.config_path)
        self.proc = OdinLaunchProcess(self.config_path, self.raw_log)

    def start(self) -> None:
        cleanup_stale_odin_processes()
        self.override.apply()
        self.proc.start()
        core.log(f"Localization process started. Raw log: {self.raw_log}")

    def poll(self) -> int | None:
        return self.proc.poll()

    def stop(self) -> None:
        self.proc.stop()
        self.override.restore()
        self.config_path.unlink(missing_ok=True)


core.LocalizationSession = LocalizationSession


def cmd_map(args: argparse.Namespace) -> int:
    map_name = args.map_name or core.generated_name("map")
    log_path = LOG_DIR / f"mapping_{time.strftime('%Y%m%d_%H%M%S')}.log"
    recorddata = bool(getattr(args, "recorddata", False))
    config_path = write_odin_config(1, map_name=map_name, recorddata=recorddata)
    override = OdinConfigOverride(config_path)
    proc = OdinLaunchProcess(config_path, log_path)
    target_file = PROJECT_ROOT / "maps" / map_name / f"{map_name}.bin"
    core.log(f"Starting Odin SLAM session: {map_name}")
    core.log(f"Target map file: {target_file}")
    if recorddata:
        core.log("MindCloud recorddata capture enabled for this mapping session.")
    cleanup_stale_odin_processes()
    override.apply()
    proc.start()
    core.log(f"Mapping raw log: {log_path}")
    try:
        while not core.STOP_REQUESTED:
            code = proc.poll()
            if code is not None:
                return code
            time.sleep(0.2)
        core.log("Stop requested. Saving the current map before shutdown...")
        saved = request_map_save(target_file, timeout_sec=60.0, raw_log=log_path)
        if saved:
            core.log("Map save completed successfully.")
        else:
            core.log("Map save did not complete before timeout. Check USB link speed and Odin logs.")
        return 0
    finally:
        proc.stop()
        try:
            set_shadow_recorddata(False)
            core.log("MindCloud recorddata capture disabled after mapping stop.")
        except Exception as exc:
            core.log(f"Warning: failed to force recorddata back to 0: {exc}")
        override.restore()
        config_path.unlink(missing_ok=True)


@dataclass
class LineStatus:
    state: str = "INIT"
    found: bool = False
    obstacle_blocked: bool = False
    lost_frames: int = 0
    reverse: bool = False
    drive_enable: bool = False
    fresh: bool = False
    updated_at: float = 0.0
    payload: dict[str, Any] | None = None


@dataclass
class TwistCommand:
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    updated_at: float = 0.0
    fresh: bool = False


def _load_local_lidar_gui_config() -> dict[str, Any]:
    try:
        data = json.loads(LOCAL_ROW_CONFIG.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _config_float(config: dict[str, Any], key: str, default: float) -> float:
    try:
        value = config.get(key, default)
        return float(value)
    except Exception:
        return float(default)


def _config_str(config: dict[str, Any], key: str, default: str) -> str:
    value = config.get(key, default)
    text = str(value).strip()
    return text if text else str(default)


def _local_lidar_resolution(config: dict[str, Any], default_width: int, default_height: int) -> tuple[int, int]:
    text = _config_str(config, "resolution", f"{int(default_width)}x{int(default_height)}").lower()
    if "x" not in text:
        return int(default_width), int(default_height)
    width_text, height_text = text.split("x", 1)
    try:
        width = int(float(width_text.strip()))
        height = int(float(height_text.strip()))
    except Exception:
        return int(default_width), int(default_height)
    return max(1, width), max(1, height)


def _local_lidar_drive_settings(config: dict[str, Any], *, reverse: bool, args: argparse.Namespace) -> tuple[float, float, float]:
    speed = abs(_config_float(config, "cruise_vx", float(args.line_cruise_vx)))
    if reverse:
        offset = _config_float(
            config,
            "reverse_target_center_offset_px",
            _config_float(config, "target_center_offset_px", float(args.line_target_center_offset_px)),
        )
        angle = _config_float(
            config,
            "reverse_vehicle_direction_angle_deg",
            _config_float(config, "vehicle_direction_angle_deg", float(args.line_vehicle_direction_angle_deg)),
        )
    else:
        offset = _config_float(
            config,
            "forward_target_center_offset_px",
            _config_float(config, "target_center_offset_px", float(args.line_target_center_offset_px)),
        )
        angle = _config_float(
            config,
            "forward_vehicle_direction_angle_deg",
            _config_float(config, "vehicle_direction_angle_deg", float(args.line_vehicle_direction_angle_deg)),
        )
    return speed, offset, angle


class LocalLidarRosBridge:
    def __init__(self, cmd_vel_topic: str, status_topic: str, drive_mode_topic: str) -> None:
        self._status_lock = threading.Lock()
        self._cmd_lock = threading.Lock()
        self._status = LineStatus()
        self._cmd = TwistCommand()
        self._bridge = Ros2NodeThread("autorun_final_bridge")
        self._mode_pub = self._bridge.node.create_publisher(String, drive_mode_topic, 10)
        self._status_sub = self._bridge.node.create_subscription(String, status_topic, self._on_status, 10)
        self._cmd_sub = self._bridge.node.create_subscription(Twist, cmd_vel_topic, self._on_cmd_vel, 10)

    def _on_status(self, msg) -> None:
        now = time.monotonic()
        try:
            payload = json.loads(msg.data)
        except Exception:
            payload = {}
        state = str(payload.get("state", "UNKNOWN"))
        found = state == "TRACK" or str(payload.get("found", "")).lower() == "true"
        blocked = str(payload.get("obstacle_blocked", "False")).lower() == "true" or state == "BLOCKED"
        lost_frames = int(float(payload.get("lost_frames", 0) or 0))
        reverse = str(payload.get("reverse", "False")).lower() == "true"
        drive_enable = str(payload.get("drive_enable", "False")).lower() == "true"
        with self._status_lock:
            self._status = LineStatus(
                state=state,
                found=found,
                obstacle_blocked=blocked,
                lost_frames=lost_frames,
                reverse=reverse,
                drive_enable=drive_enable,
                fresh=True,
                updated_at=now,
                payload=payload if isinstance(payload, dict) else {},
            )

    def _on_cmd_vel(self, msg) -> None:
        with self._cmd_lock:
            self._cmd = TwistCommand(
                vx=float(msg.linear.x),
                vy=float(msg.linear.y),
                wz=float(msg.angular.z),
                updated_at=time.monotonic(),
                fresh=True,
            )

    def publish_mode(
        self,
        *,
        enable: bool,
        reverse: bool,
        cruise_vx: float,
        gear: str,
        low_beam: bool,
        target_center_offset_px: float = 0.0,
        vehicle_direction_angle_deg: float = 0.0,
    ) -> None:
        payload = {
            "enable": bool(enable),
            "reverse": bool(reverse),
            "cruise_vx": abs(float(cruise_vx)),
            "gear": str(gear),
            "low_beam": bool(low_beam),
            "target_center_offset_px": float(target_center_offset_px),
            "vehicle_direction_angle_deg": float(vehicle_direction_angle_deg),
        }
        self._mode_pub.publish(String(data=json.dumps(payload, ensure_ascii=True)))

    def status_snapshot(self) -> LineStatus:
        with self._status_lock:
            status = self._status
        age = time.monotonic() - status.updated_at if status.updated_at > 0.0 else 1e9
        if age > 1.0:
            status = LineStatus(
                state=status.state,
                found=status.found,
                obstacle_blocked=status.obstacle_blocked,
                lost_frames=status.lost_frames,
                reverse=status.reverse,
                drive_enable=status.drive_enable,
                fresh=False,
                updated_at=status.updated_at,
                payload=status.payload,
            )
        return status

    def cmd_snapshot(self) -> TwistCommand:
        with self._cmd_lock:
            cmd = self._cmd
        age = time.monotonic() - cmd.updated_at if cmd.updated_at > 0.0 else 1e9
        if age > 0.5:
            return TwistCommand(vx=cmd.vx, vy=cmd.vy, wz=cmd.wz, updated_at=cmd.updated_at, fresh=False)
        return cmd

    def close(self) -> None:
        self._bridge.close()


class LocalLidarProcess(core.ManagedProcess):
    def __init__(self, args: argparse.Namespace, log_path: Path, local_lidar_config: dict[str, Any] | None = None) -> None:
        config = local_lidar_config or {}
        cmd = [
            str(Path(sys.executable)),
            str(LOCAL_ROW_RUNNER),
            "--scan-topic",
            str(args.lidar_scan_topic),
            "--cmd-vel-topic",
            args.ros_cmd_vel_topic,
            "--status-topic",
            args.ros_status_topic,
            "--drive-mode-topic",
            args.ros_drive_mode_topic,
            "--min-speed",
            str(args.lidar_min_speed),
            "--max-wz-deg",
            str(args.lidar_max_wz_deg),
            "--max-heading-wz-deg",
            str(args.lidar_max_heading_wz_deg),
            "--k-lat",
            str(args.lidar_k_lat),
            "--k-heading",
            str(args.lidar_k_heading),
            "--row-width",
            str(args.lidar_row_width),
            "--min-row-width",
            str(args.lidar_min_row_width),
            "--max-row-width",
            str(args.lidar_max_row_width),
            "--lookahead-x",
            str(args.lidar_lookahead_x),
            "--forward-lookahead-x",
            str(args.lidar_forward_lookahead_x),
            "--reverse-lookahead-x",
            str(args.lidar_reverse_lookahead_x),
            "--forward-min",
            str(args.lidar_forward_min),
            "--forward-max",
            str(args.lidar_forward_max),
            "--lateral-limit",
            str(args.lidar_lateral_limit),
            "--range-min",
            str(args.lidar_range_min),
            "--range-max",
            str(args.lidar_range_max),
            "--bin-size",
            str(args.lidar_bin_size),
            "--min-points",
            str(args.lidar_min_points),
            "--min-bins",
            str(args.lidar_min_bins),
            "--min-line-bins",
            str(args.lidar_min_line_bins),
            "--min-side-points-per-bin",
            str(args.lidar_min_side_points_per_bin),
            "--boundary-width-tolerance-m",
            str(args.lidar_boundary_width_tolerance_m),
            "--center-deadband",
            str(args.lidar_center_deadband),
            "--left-percentile",
            str(args.lidar_left_percentile),
            "--right-percentile",
            str(args.lidar_right_percentile),
            "--sensor-yaw-deg",
            str(args.lidar_sensor_yaw_deg),
            "--lidar-yaw-correction-deg",
            str(args.lidar_yaw_correction_deg),
            "--lidar-x-offset-m",
            str(args.lidar_x_offset_m),
            "--lidar-y-offset-m",
            str(args.lidar_y_offset_m),
            "--vehicle-width",
            str(args.lidar_vehicle_width),
            "--center-y-target",
            str(args.lidar_center_y_target),
            "--heading-conflict-error-y",
            str(args.lidar_heading_conflict_error_y),
            "--heading-conflict-scale",
            str(args.lidar_heading_conflict_scale),
            "--slow-error-y",
            str(args.lidar_slow_error_y),
            "--stop-error-y",
            str(args.lidar_stop_error_y),
            "--slow-heading-rad",
            str(args.lidar_slow_heading_rad),
            "--control-deadband-y",
            str(args.lidar_control_deadband_y),
            "--safety-margin",
            str(args.lidar_safety_margin),
            "--center-jump-reject",
            str(args.lidar_center_jump_reject),
            "--one-side-center-jump-reject",
            str(args.lidar_one_side_center_jump_reject),
            "--center-y-reject-abs",
            str(args.lidar_center_y_reject_abs),
            "--raw-center-out-of-range",
            str(args.lidar_raw_center_out_of_range),
            "--one-side-raw-center-out-of-range",
            str(args.lidar_one_side_raw_center_out_of_range),
            "--history-window-s",
            str(args.lidar_history_window_s),
            "--min-center-history",
            str(args.lidar_min_center_history),
            "--center-y-alpha",
            str(args.lidar_center_y_alpha),
            "--center-y-max-jump",
            str(args.lidar_center_y_max_jump),
            "--one-side-stop-error-y",
            str(args.lidar_one_side_stop_error_y),
            "--one-side-safety-stop-band",
            str(args.lidar_one_side_safety_stop_band),
            "--control-period",
            str(args.lidar_control_period),
            "--status-period",
            str(args.lidar_status_period),
            "--scan-timeout",
            str(args.lidar_scan_timeout),
            "--forward-lost-hold-sec",
            str(args.lidar_forward_lost_hold_sec),
            "--forward-lost-stop-sec",
            str(args.lidar_forward_lost_stop_sec),
            "--forward-lost-hold-wz-scale",
            str(args.lidar_forward_lost_hold_wz_scale),
            "--forward-lost-hold-max-wz-deg",
            str(args.lidar_forward_lost_hold_max_wz_deg),
            "--reverse-min-speed",
            str(args.lidar_reverse_min_speed),
            "--reverse-one-side-speed",
            str(args.lidar_reverse_one_side_speed),
            "--reverse-both-sides-speed",
            str(args.lidar_reverse_both_sides_speed),
            "--reverse-min-wz-deg",
            str(args.lidar_reverse_min_wz_deg),
            "--reverse-max-wz-deg",
            str(args.lidar_reverse_max_wz_deg),
            "--reverse-one-side-max-wz-deg",
            str(args.lidar_reverse_one_side_max_wz_deg),
            "--reverse-wz-enable-error-y",
            str(args.lidar_reverse_wz_enable_error_y),
            "--reverse-wz-enable-heading-deg",
            str(args.lidar_reverse_wz_enable_heading_deg),
            "--reverse-min-wz-error-y",
            str(args.lidar_reverse_min_wz_error_y),
            "--reverse-sign-flip-guard-error-y",
            str(args.lidar_reverse_sign_flip_guard_error_y),
            "--reverse-sign-flip-guard-last-wz-deg",
            str(args.lidar_reverse_sign_flip_guard_last_wz_deg),
            "--reverse-sign-hold-error-y",
            str(args.lidar_reverse_sign_hold_error_y),
            "--reverse-both-sides-k-lat",
            str(args.lidar_reverse_both_sides_k_lat),
            "--reverse-both-sides-k-heading",
            str(args.lidar_reverse_both_sides_k_heading),
            "--reverse-one-side-k-lat",
            str(args.lidar_reverse_one_side_k_lat),
            "--k-reverse-lat",
            str(args.lidar_k_reverse_lat),
            "--k-reverse-heading",
            str(args.lidar_k_reverse_heading),
            "--reverse-steer-sign",
            str(args.lidar_reverse_steer_sign),
            "--reverse-heading-conflict-error-y",
            str(args.lidar_reverse_heading_conflict_error_y),
            "--reverse-heading-max-ratio",
            str(args.lidar_reverse_heading_max_ratio),
            "--reverse-error-stop",
            str(args.lidar_reverse_error_stop),
            "--reverse-wz-smoothing-alpha",
            str(args.lidar_reverse_wz_smoothing_alpha),
            "--reverse-lost-hold-sec",
            str(args.lidar_reverse_lost_hold_sec),
            "--reverse-lost-stop-sec",
            str(args.lidar_reverse_lost_stop_sec),
            "--reverse-lost-hold-max-wz-deg",
            str(args.lidar_reverse_lost_hold_max_wz_deg),
            "--reverse-lost-soft-max-wz-deg",
            str(args.lidar_reverse_lost_soft_max_wz_deg),
            "--max-wz-delta-deg-per-cycle",
            str(args.lidar_max_wz_delta_deg_per_cycle),
            "--speed",
            str(_config_float(config, "cruise_vx", float(args.line_cruise_vx))),
        ]
        super().__init__(cmd, PROJECT_ROOT, log_path)


class LidarDriverProcess(core.ManagedProcess):
    def __init__(self, log_path: Path) -> None:
        if not LIDAR_DRIVER_BIN.exists():
            raise RuntimeError(f"Lidar driver binary not found: {LIDAR_DRIVER_BIN}")
        cmd = [
            str(LIDAR_DRIVER_BIN),
        ]
        if LIDAR_DRIVER_PARAMS.exists():
            cmd.extend(["--ros-args", "--params-file", str(LIDAR_DRIVER_PARAMS)])
        super().__init__(cmd, LIDAR_DRIVER_ROOT, log_path)


def _wait_for_local_lidar_ready(
    ros_bridge: LocalLidarRosBridge,
    args: argparse.Namespace,
    *,
    local_lidar_config: dict[str, Any] | None = None,
    timeout_s: float = 5.0,
) -> bool:
    config = local_lidar_config or {}
    cruise_vx, offset_px, direction_angle_deg = _local_lidar_drive_settings(config, reverse=False, args=args)
    deadline = time.monotonic() + max(0.1, float(timeout_s))
    last_log = 0.0
    while time.monotonic() < deadline and not core.STOP_REQUESTED:
        ros_bridge.publish_mode(
            enable=True,
            reverse=False,
            cruise_vx=cruise_vx,
            gear="4t4d",
            low_beam=args.line_low_beam,
            target_center_offset_px=offset_px,
            vehicle_direction_angle_deg=direction_angle_deg,
        )
        status = ros_bridge.status_snapshot()
        cmd = ros_bridge.cmd_snapshot()
        if cmd.fresh:
            core.log("lidar local guidance is ready for hybrid replay.")
            return True
        now = time.monotonic()
        if now - last_log >= 1.0:
            core.log(
                f"Waiting for lidar local guidance: state={status.state} "
                f"fresh={status.fresh} cmd_fresh={cmd.fresh}"
            )
            last_log = now
        time.sleep(0.10)
    return False


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _linear_blend_amount(value: float, start: float, end: float) -> float:
    if end <= start:
        return 1.0 if value <= end else 0.0
    if value <= start:
        return 1.0
    if value >= end:
        return 0.0
    return (end - value) / (end - start)


def _reverse_entry_turn_scale(dist_since_reverse_start: float | None) -> tuple[float, float]:
    if dist_since_reverse_start is None:
        return 1.0, 8.0
    dist = max(0.0, float(dist_since_reverse_start))
    if dist >= 0.9:
        return 1.0, 8.0
    amount = dist / 0.9
    scale = 0.65 + 0.35 * amount
    max_wz = 3.5 + 4.5 * amount
    return scale, max_wz


def _snapshot_age_seconds(updated_at: float) -> float:
    if updated_at <= 0.0:
        return 1e9
    return max(0.0, time.monotonic() - updated_at)


def _line_status_is_tracking_ready(
    status: LineStatus,
    *,
    allow_stale_hold: bool = False,
    hold_s: float = 1.2,
) -> bool:
    if status.state != "TRACK":
        return False
    if status.fresh:
        return True
    if not allow_stale_hold or not status.drive_enable:
        return False
    return _snapshot_age_seconds(status.updated_at) <= max(0.05, float(hold_s))


def _local_cmd_is_usable(
    cmd: TwistCommand,
    *,
    allow_stale_hold: bool = False,
    hold_s: float = 1.2,
) -> bool:
    if cmd.fresh:
        return True
    if not allow_stale_hold:
        return False
    return _snapshot_age_seconds(cmd.updated_at) <= max(0.05, float(hold_s))


def _find_next_row_switch_index(
    motions: list[dict[str, Any]],
    start_index: int,
    *,
    limit: int = 48,
) -> int | None:
    end = min(len(motions), start_index + max(1, limit))
    for idx in range(max(0, start_index), end):
        motion = motions[idx]
        gear = core._resolve_replay_gear(motion, None)
        if gear == "crab" or abs(_safe_float(motion.get("vy"), 0.0)) > 0.05:
            return idx
    return None


def _find_next_reverse_index(
    motions: list[dict[str, Any]],
    start_index: int,
    *,
    limit: int = 48,
) -> int | None:
    end = min(len(motions), start_index + max(1, limit))
    for idx in range(max(0, start_index), end):
        if _safe_float(motions[idx].get("vx"), 0.0) < -0.03:
            return idx
    return None


def _find_next_4t4d_index(
    motions: list[dict[str, Any]],
    start_index: int,
    *,
    limit: int = 64,
) -> int | None:
    end = min(len(motions), start_index + max(1, limit))
    for idx in range(max(0, start_index), end):
        if core._resolve_replay_gear(motions[idx], None) == "4t4d":
            return idx
    return None


def _find_next_forward_4t4d_index(
    motions: list[dict[str, Any]],
    start_index: int,
    *,
    limit: int = 72,
) -> int | None:
    end = min(len(motions), start_index + max(1, limit))
    for idx in range(max(0, start_index), end):
        motion = motions[idx]
        if core._resolve_replay_gear(motion, None) != "4t4d":
            continue
        if _safe_float(motion.get("vx"), 0.0) >= -0.02:
            return idx
    return None


def _skip_hold_4t4d_cluster(
    points: list[core.Pose2D],
    motions: list[dict[str, Any]],
    start_index: int,
    *,
    limit: int = 48,
) -> int:
    if not points:
        return 0
    idx = max(0, min(start_index, len(points) - 1))
    end = min(len(points) - 1, idx + max(1, limit))
    while idx < end:
        motion = motions[min(idx, len(motions) - 1)]
        if _motion_segment_mode(motion) != "hold:4t4d":
            break
        next_idx = idx + 1
        if next_idx > end:
            break
        if points[idx].distance_to(points[next_idx]) > 0.03:
            break
        idx = next_idx
    return idx


def _find_forward_resume_index(
    points: list[core.Pose2D],
    motions: list[dict[str, Any]],
    start_index: int,
    pose: core.Pose2D,
    *,
    limit: int = 24,
) -> int:
    if not points:
        return 0
    start = max(0, min(start_index, len(points) - 1))
    start = _skip_hold_4t4d_cluster(points, motions, start)
    end = min(len(points), start + max(1, limit))
    best_idx = start
    best_dist = float("inf")
    for idx in range(start, end):
        if core._resolve_replay_gear(motions[min(idx, len(motions) - 1)], None) != "4t4d":
            continue
        cur = points[idx]
        dx = cur.x - pose.x
        dy = cur.y - pose.y
        # Keep only targets that are in front of the chassis heading so we do
        # not snap back to the first point of the next row and oscillate.
        forward_proj = math.cos(pose.yaw) * dx + math.sin(pose.yaw) * dy
        if forward_proj < -0.05:
            continue
        dist = pose.distance_to(cur)
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx


def _find_recent_reverse_index(
    motions: list[dict[str, Any]],
    start_index: int,
    *,
    limit: int = 24,
) -> int | None:
    begin = max(0, start_index - max(1, limit))
    for idx in range(start_index - 1, begin - 1, -1):
        if _safe_float(motions[idx].get("vx"), 0.0) < -0.03:
            return idx
    return None


def _find_reverse_segment_start(
    motions: list[dict[str, Any]],
    index: int,
) -> int | None:
    if index < 0 or index >= len(motions):
        return None
    if _safe_float(motions[index].get("vx"), 0.0) >= -0.03:
        return None
    start = index
    while start > 0 and _safe_float(motions[start - 1].get("vx"), 0.0) < -0.03:
        start -= 1
    return start


def _crab_reference_vy(
    points: list[core.Pose2D],
    pose: core.Pose2D,
    motions: list[dict[str, Any]],
    start_index: int,
    target_index: int,
    send_state: core.MotionSendState,
) -> float:
    if not motions:
        return 0.0
    search_start = send_state.crab_target_index if send_state.crab_target_index >= 0 else start_index
    search_end = send_state.crab_locked_until if send_state.crab_locked_until >= search_start else target_index
    search_start = max(0, min(search_start, len(motions) - 1))
    search_end = max(search_start, min(search_end, len(motions) - 1))
    best_vy = 0.0
    for idx in range(search_start, search_end + 1):
        vy = _safe_float(motions[idx].get("vy"), 0.0)
        if abs(vy) > abs(best_vy):
            best_vy = vy
    if abs(best_vy) >= 0.03:
        return best_vy
    for idx in range(max(0, start_index - 4), min(len(motions), target_index + 5)):
        vy = _safe_float(motions[idx].get("vy"), 0.0)
        if abs(vy) > abs(best_vy):
            best_vy = vy
    if abs(best_vy) >= 0.03:
        return best_vy
    if not points:
        return 0.0
    ref_index = max(0, min(start_index, len(points) - 1))
    tgt_index = max(0, min(target_index, len(points) - 1))
    ref_point = points[ref_index]
    target = points[tgt_index]
    _, target_lateral_err = core._body_frame_error(pose, target)
    if abs(target_lateral_err) >= 0.02:
        return math.copysign(0.12, target_lateral_err)
    dx = target.x - ref_point.x
    dy = target.y - ref_point.y
    segment_lateral = -math.sin(pose.yaw) * dx + math.cos(pose.yaw) * dy
    if abs(segment_lateral) >= 0.02:
        return math.copysign(0.12, segment_lateral)
    return 0.0


def _crab_axis_error(
    pose: core.Pose2D,
    ref_point: core.Pose2D,
    target: core.Pose2D,
) -> float:
    axis_dx = target.x - ref_point.x
    axis_dy = target.y - ref_point.y
    if abs(axis_dy) >= abs(axis_dx):
        return target.y - pose.y
    return target.x - pose.x


def _crab_motion_active(motion: dict[str, Any]) -> bool:
    return (
        abs(_safe_float(motion.get("vy"), 0.0)) >= 0.05
        or abs(_safe_float(motion.get("vx"), 0.0)) >= 0.05
        or abs(_safe_float(motion.get("wz"), 0.0)) >= 3.0
    )


def _skip_crab_tail_index(
    motions: list[dict[str, Any]],
    start_index: int,
) -> int:
    if not motions:
        return 0
    idx = max(0, min(start_index, len(motions) - 1))
    motion = motions[idx]
    if core._resolve_replay_gear(motion, None) != "crab":
        return idx
    if _crab_motion_active(motion):
        return idx
    crab_end = core._find_gear_segment_end(motions, idx, "crab")
    return min(len(motions) - 1, crab_end + 1)


def _compute_hybrid_weights(
    *,
    args: argparse.Namespace,
    row_switch_mode: bool,
    reversing_mode: bool,
    near_finish: bool,
    local_status: LineStatus,
    local_cmd: TwistCommand,
    dist_to_row_switch: float | None,
    dist_to_reverse_start: float | None,
    dist_since_reverse_start: float | None,
) -> tuple[float, float, str, bool]:
    configured_local_weight = max(0.0, min(1.0, float(args.local_weight_in_row)))
    configured_global_weight = max(0.0, min(1.0, float(args.global_weight_in_row)))
    local_enabled_by_weight = configured_local_weight > 1e-6
    local_only_base = configured_global_weight <= 1e-6 and local_enabled_by_weight and not row_switch_mode
    local_only_requested = (
        local_only_base
        and (reversing_mode or not near_finish)
    )
    allow_stale_local_hold = (
        reversing_mode
        and local_only_requested
    )

    if row_switch_mode:
        return 1.0, 0.0, "global-row-switch", False
    if not local_enabled_by_weight:
        return 1.0, 0.0, "global-only", False
    if local_status.obstacle_blocked:
        if local_only_requested and reversing_mode:
            return 0.0, 1.0, "local-block-stop", False
        return 1.0, 0.0, "global-near-finish-blocked", False

    total = configured_global_weight + configured_local_weight
    if total <= 1e-6:
        base_global = 1.0
        base_local = 0.0
    else:
        base_global = configured_global_weight / total
        base_local = configured_local_weight / total

    local_cmd_ready = _local_cmd_is_usable(
        local_cmd,
        allow_stale_hold=allow_stale_local_hold,
    )
    using_stale_local_hold = allow_stale_local_hold and (
        not local_cmd.fresh and local_cmd_ready
    )
    if local_only_requested:
        if not local_cmd_ready:
            return 0.0, 1.0, "local-only-wait", allow_stale_local_hold
        return 0.0, 1.0, ("local-only-hold" if using_stale_local_hold else "local-only"), using_stale_local_hold

    local_tracking_ready = _line_status_is_tracking_ready(
        local_status,
        allow_stale_hold=allow_stale_local_hold,
    )
    if not (local_tracking_ready and local_cmd_ready):
        return 1.0, 0.0, "global-fallback", False

    if dist_to_row_switch is None and dist_to_reverse_start is None:
        return base_global, base_local, ("blend-row-local-hold" if using_stale_local_hold else "blend-row"), using_stale_local_hold

    full_global_dist = max(0.05, float(getattr(args, "row_switch_full_global_dist", 1.50)))
    blend_start_dist = max(full_global_dist, float(getattr(args, "row_switch_blend_start_dist", 3.0)))
    anticipation = 0.0
    if dist_to_row_switch is not None:
        anticipation = max(anticipation, _linear_blend_amount(dist_to_row_switch, full_global_dist, blend_start_dist))
    if anticipation <= 1e-6:
        return base_global, base_local, "blend-row", False

    global_weight = base_global + (1.0 - base_global) * anticipation
    local_weight = base_local * (1.0 - anticipation)
    total = global_weight + local_weight
    if total <= 1e-6:
        return 1.0, 0.0, "global-anticipation", False
    mode_name = "blend-approach-maneuver-local-hold" if using_stale_local_hold else "blend-approach-maneuver"
    return global_weight / total, local_weight / total, mode_name, using_stale_local_hold


def _compute_global_command(
    pose: core.Pose2D,
    ref_point: core.Pose2D,
    target: core.Pose2D,
    motion: dict[str, Any],
    gear: str,
    send_state: core.MotionSendState,
    tracking_heading: float,
) -> tuple[float, float, float, float, float]:
    dist = pose.distance_to(target)
    forward_err, lateral_err = core._body_frame_error(pose, target)
    heading_err = core.normalize_angle(tracking_heading - pose.yaw)
    heading_err_deg, lateral_err = core._smooth_tracking_errors(
        send_state,
        math.degrees(heading_err),
        lateral_err,
        alpha=0.24,
    )
    if abs(heading_err_deg) < 2.0:
        heading_err_deg = 0.0
    if abs(lateral_err) < 0.02:
        lateral_err = 0.0
    vx_cap = max(abs(_safe_float(motion.get("vx"), 0.0)), 0.22)
    wz_cap = max(abs(_safe_float(motion.get("wz"), 0.0)), 20.0)
    cmd_vx = core._clamp(forward_err * 0.75, -vx_cap, vx_cap)
    cmd_vy = 0.0
    if gear == "crab":
        recorded_vy = _safe_float(motion.get("vy"), 0.0)
        recorded_vy_abs = abs(recorded_vy)
        vy_cap = max(recorded_vy_abs, 0.20)
        cmd_vx = 0.0
        axis_dx = target.x - ref_point.x
        axis_dy = target.y - ref_point.y
        axis_err = (target.y - pose.y) if abs(axis_dy) >= abs(axis_dx) else (target.x - pose.x)
        axis_sign = math.copysign(1.0, recorded_vy) if recorded_vy_abs >= 0.03 else math.copysign(1.0, axis_err if abs(axis_err) > 1e-6 else 1.0)
        cmd_vy = core._clamp(abs(axis_err) * 1.35, 0.0, vy_cap) * axis_sign
        if abs(axis_err) < 0.03:
            cmd_vy = 0.0
        if abs(heading_err_deg) > 18.0:
            cmd_vy = core._clamp(cmd_vy, -0.08, 0.08)
        if abs(cmd_vy) < 0.06 and abs(axis_err) > 0.12:
            crab_floor = min(vy_cap, max(0.12, min(recorded_vy_abs if recorded_vy_abs > 0.03 else 0.16, 0.18)))
            cmd_vy = math.copysign(crab_floor, axis_sign)
        # Keep row-switch motion as pure lateral travel. Mixing a large body yaw
        # rate into FW-mini crab mode makes the wheel angle hunt between 0/90 deg.
        cmd_wz = 0.0
    else:
        cmd_vx = core._clamp(cmd_vx, -0.16, 0.16)
        reversing = _safe_float(motion.get("vx"), 0.0) < -0.03
        if reversing:
            forward_err, lateral_err = _path_frame_error(pose, target, tracking_heading)
            heading_err_deg = math.degrees(core.normalize_angle(tracking_heading - pose.yaw))
            cmd_vx = core._clamp(forward_err * 0.75, -0.16, 0.16)
        heading_gain = 0.72
        lateral_gain = 18.0
        lateral_term = lateral_err * lateral_gain
        wz_limit = min(wz_cap, 12.0)
        cmd_wz = core._clamp(
            heading_err_deg * heading_gain + lateral_term,
            -wz_limit,
            wz_limit,
        )
        if abs(heading_err_deg) > 45.0:
            cmd_vx = core._clamp(cmd_vx, -0.04, 0.04)
        if abs(heading_err_deg) > 65.0:
            cmd_vx = core._signed_crawl(forward_err, 0.025)
        if gear == "4t4d":
            cmd_wz = core._limit_4t4d_turn_rate(cmd_vx, cmd_wz)
    return cmd_vx, cmd_vy, cmd_wz, dist, heading_err_deg


def _build_body_command(
    gear: str,
    global_cmd: tuple[float, float, float],
    local_cmd: TwistCommand,
    local_status: LineStatus,
    global_weight: float,
    local_weight: float,
    allow_stale_local_hold: bool = False,
) -> core.BodyCommand:
    g_vx, g_vy, g_wz = global_cmd
    if gear == "crab":
        return core.BodyCommand(gear=gear, vx=g_vx, vy=g_vy, wz=g_wz)

    def _limit_4t4d_wz(vx: float, wz: float) -> float:
        if gear != "4t4d":
            return wz
        # Keep the hybrid path closer to the standalone local follower feel by
        # reapplying the chassis-side 4t4d turn-rate limiter on the final
        # angular velocity command. This only affects 4t4d forward/reverse
        # motion and does not touch crab/lateral row-switch motion.
        wz_deg = math.degrees(float(wz))
        limited_deg = core._limit_4t4d_turn_rate(float(vx), wz_deg)
        return math.radians(limited_deg)

    if local_weight <= 1e-6:
        return core.BodyCommand(gear=gear, vx=g_vx, vy=0.0, wz=_limit_4t4d_wz(g_vx, g_wz))
    if local_status.obstacle_blocked and global_weight <= 1e-6:
        return core.BodyCommand(gear=gear, vx=0.0, vy=0.0, wz=0.0)
    local_cmd_ready = _local_cmd_is_usable(
        local_cmd,
        allow_stale_hold=allow_stale_local_hold,
    )
    if global_weight <= 1e-6:
        if not local_cmd_ready:
            return core.BodyCommand(gear=gear, vx=0.0, vy=0.0, wz=0.0)
        return core.BodyCommand(
            gear=gear,
            vx=local_cmd.vx,
            vy=0.0,
            wz=_limit_4t4d_wz(local_cmd.vx, local_cmd.wz),
        )
    # Once hybrid logic decides the local controller should participate, keep
    # its cmd_vel contribution as direct as possible. We only fall back to
    # global when the local cmd itself is unusable.
    if local_cmd_ready:
        vx = local_weight * local_cmd.vx + global_weight * g_vx
        wz = local_weight * local_cmd.wz + global_weight * g_wz
        return core.BodyCommand(gear=gear, vx=vx, vy=0.0, wz=_limit_4t4d_wz(vx, wz))
    local_tracking_ready = _line_status_is_tracking_ready(
        local_status,
        allow_stale_hold=allow_stale_local_hold,
    )
    if not (local_tracking_ready and local_cmd_ready):
        return core.BodyCommand(gear=gear, vx=g_vx, vy=0.0, wz=_limit_4t4d_wz(g_vx, g_wz))
    vx = local_weight * local_cmd.vx + global_weight * g_vx
    wz = local_weight * local_cmd.wz + global_weight * g_wz
    return core.BodyCommand(gear=gear, vx=vx, vy=0.0, wz=_limit_4t4d_wz(vx, wz))


def _motion_segment_mode(motion: dict[str, Any], current_gear: str | None = None) -> str:
    gear = core._resolve_replay_gear(motion, current_gear)
    if gear == "crab":
        return "crab"
    vx = _safe_float(motion.get("vx"), 0.0)
    if vx < -0.03:
        return "reverse"
    if vx > 0.03:
        return "forward"
    return f"hold:{gear}"


def _find_tracking_index_same_mode(
    points: list[core.Pose2D],
    motions: list[dict[str, Any]],
    current_index: int,
    pose: core.Pose2D,
    current_gear: str | None,
    window: int = 12,
) -> int:
    if not points:
        return 0
    start = max(0, current_index)
    end = min(len(points), current_index + max(2, window))
    mode = _motion_segment_mode(motions[min(start, len(motions) - 1)], current_gear)
    best_same_index = current_index
    best_same_dist = float("inf")
    best_any_index = current_index
    best_any_dist = float("inf")
    for idx in range(start, end):
        dist = pose.distance_to(points[idx])
        if dist < best_any_dist:
            best_any_dist = dist
            best_any_index = idx
        if _motion_segment_mode(motions[min(idx, len(motions) - 1)], current_gear) != mode:
            continue
        if dist < best_same_dist:
            best_same_dist = dist
            best_same_index = idx
    if best_same_dist < float("inf"):
        return best_same_index
    return best_any_index


def _segment_progress_ratio(start: core.Pose2D, end: core.Pose2D, pose: core.Pose2D) -> float:
    dx = end.x - start.x
    dy = end.y - start.y
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-9:
        return 0.0
    px = pose.x - start.x
    py = pose.y - start.y
    return (px * dx + py * dy) / length_sq


def _advance_dense_tracking_index(
    points: list[core.Pose2D],
    motions: list[dict[str, Any]],
    current_index: int,
    pose: core.Pose2D,
    current_gear: str | None,
    *,
    max_skip: int = 6,
) -> int:
    if not points:
        return 0
    idx = max(0, min(current_index, len(points) - 1))
    mode = _motion_segment_mode(motions[min(idx, len(motions) - 1)], current_gear)
    skipped = 0
    while idx + 1 < len(points) and skipped < max_skip:
        next_idx = idx + 1
        next_mode = _motion_segment_mode(motions[min(next_idx, len(motions) - 1)], current_gear)
        if next_mode != mode:
            break
        spacing = points[idx].distance_to(points[next_idx])
        motion = motions[min(idx, len(motions) - 1)]
        vx = abs(_safe_float(motion.get("vx"), 0.0))
        vy = abs(_safe_float(motion.get("vy"), 0.0))
        wz = abs(_safe_float(motion.get("wz"), 0.0))
        low_motion = max(vx, vy) < 0.05 and wz < 6.0
        cur_dist = pose.distance_to(points[idx])
        next_dist = pose.distance_to(points[next_idx])
        # When recording very slowly, neighboring mission samples can be so dense
        # that replay keeps targeting points the chassis has effectively already
        # passed. Prefer the next sample if it is almost coincident and not farther.
        if spacing <= 0.035 and low_motion and next_dist <= cur_dist + 0.02:
            idx = next_idx
            skipped += 1
            continue
        if cur_dist <= 0.08 and next_dist <= 0.14 and next_dist <= cur_dist + 0.03:
            idx = next_idx
            skipped += 1
            continue
        break
    return idx


def _advance_tracking_progress_index(
    points: list[core.Pose2D],
    motions: list[dict[str, Any]],
    current_index: int,
    pose: core.Pose2D,
    current_gear: str | None,
    *,
    max_skip: int = 24,
) -> int:
    if not points:
        return 0
    idx = max(0, min(current_index, len(points) - 1))
    skipped = 0
    while idx + 1 < len(points) and skipped < max_skip:
        next_idx = idx + 1
        mode = _motion_segment_mode(motions[min(idx, len(motions) - 1)], current_gear)
        next_mode = _motion_segment_mode(motions[min(next_idx, len(motions) - 1)], current_gear)
        if next_mode != mode:
            break
        cur = points[idx]
        nxt = points[next_idx]
        spacing = cur.distance_to(nxt)
        progress = _segment_progress_ratio(cur, nxt, pose)
        cur_dist = pose.distance_to(cur)
        next_dist = pose.distance_to(nxt)
        if spacing <= 1e-4:
            idx = next_idx
            skipped += 1
            continue
        if progress >= 0.70:
            idx = next_idx
            skipped += 1
            continue
        if progress >= 0.45 and next_dist <= cur_dist + 0.01:
            idx = next_idx
            skipped += 1
            continue
        if spacing <= 0.06 and progress >= 0.20 and next_dist <= max(0.16, cur_dist + 0.02):
            idx = next_idx
            skipped += 1
            continue
        break
    return idx


def _advance_reverse_progress_index(
    points: list[core.Pose2D],
    motions: list[dict[str, Any]],
    current_index: int,
    pose: core.Pose2D,
    current_gear: str | None,
    *,
    max_skip: int = 6,
) -> int:
    if not points:
        return 0
    idx = max(0, min(current_index, len(points) - 1))
    skipped = 0
    while idx + 1 < len(points) and skipped < max_skip:
        if _motion_segment_mode(motions[min(idx, len(motions) - 1)], current_gear) != "reverse":
            break
        next_idx = idx + 1
        if _motion_segment_mode(motions[min(next_idx, len(motions) - 1)], current_gear) != "reverse":
            break
        cur = points[idx]
        nxt = points[next_idx]
        spacing = cur.distance_to(nxt)
        progress = _segment_progress_ratio(cur, nxt, pose)
        cur_dist = pose.distance_to(cur)
        next_dist = pose.distance_to(nxt)
        if spacing <= 1e-4:
            idx = next_idx
            skipped += 1
            continue
        if progress >= 0.55:
            idx = next_idx
            skipped += 1
            continue
        if next_dist + 0.03 < cur_dist:
            idx = next_idx
            skipped += 1
            continue
        if spacing <= 0.08 and next_dist <= max(0.18, cur_dist + 0.02):
            idx = next_idx
            skipped += 1
            continue
        break
    return idx


def _tracking_lookahead_distance(motion: dict[str, Any], gear: str) -> float:
    vx = abs(_safe_float(motion.get("vx"), 0.0))
    vy = abs(_safe_float(motion.get("vy"), 0.0))
    wz = abs(_safe_float(motion.get("wz"), 0.0))
    planar_speed = math.hypot(vx, vy)
    if gear == "crab":
        lookahead = 0.14 + 0.35 * min(planar_speed, 0.20)
        if wz > 8.0:
            lookahead -= 0.02
        return core._clamp(lookahead, 0.12, 0.22)
    if vx < -0.03:
        lookahead = 0.18 + 0.45 * min(planar_speed, 0.20)
        if wz > 10.0:
            lookahead -= 0.03
        return core._clamp(lookahead, 0.16, 0.28)
    lookahead = 0.24 + 0.55 * min(planar_speed, 0.25)
    if wz > 14.0:
        lookahead -= 0.05
    elif wz > 8.0:
        lookahead -= 0.03
    return core._clamp(lookahead, 0.20, 0.40)


def _select_reverse_target_index(
    points: list[core.Pose2D],
    motions: list[dict[str, Any]],
    start_index: int,
    current_gear: str | None,
    *,
    lookahead_m: float = 0.18,
    max_span: int = 8,
) -> int:
    if not points:
        return 0
    start = max(0, min(start_index, len(points) - 1))
    target = start
    accum = 0.0
    for _ in range(max_span):
        if target + 1 >= len(points):
            break
        next_idx = target + 1
        if _motion_segment_mode(motions[min(next_idx, len(motions) - 1)], current_gear) != "reverse":
            break
        accum += points[target].distance_to(points[next_idx])
        target = next_idx
        if accum >= lookahead_m:
            break
    return max(target, min(len(points) - 1, start + 1))


def _select_lookahead_target_index(
    points: list[core.Pose2D],
    motions: list[dict[str, Any]],
    start_index: int,
    current_gear: str | None,
    *,
    lookahead_m: float | None = None,
    max_span: int = 24,
) -> int:
    if not points:
        return 0
    start = max(0, min(start_index, len(points) - 1))
    motion_here = motions[min(start, len(motions) - 1)]
    mode = _motion_segment_mode(motion_here, current_gear)
    target = start
    accum = 0.0
    desired = lookahead_m
    if desired is None:
        desired = _tracking_lookahead_distance(motion_here, core._resolve_replay_gear(motion_here, current_gear))
    if mode == "forward" and start <= 3:
        desired = max(float(desired), 0.30)
    for _ in range(max_span):
        if target + 1 >= len(points):
            break
        next_idx = target + 1
        next_mode = _motion_segment_mode(motions[min(next_idx, len(motions) - 1)], current_gear)
        if next_mode != mode:
            break
        accum += points[target].distance_to(points[next_idx])
        target = next_idx
        if accum >= desired:
            break
    if target == start and start + 1 < len(points):
        next_mode = _motion_segment_mode(motions[min(start + 1, len(motions) - 1)], current_gear)
        if next_mode == mode:
            return start + 1
    return target


def _path_frame_error(pose: core.Pose2D, target: core.Pose2D, heading: float) -> tuple[float, float]:
    dx = target.x - pose.x
    dy = target.y - pose.y
    c = math.cos(heading)
    s = math.sin(heading)
    along = c * dx + s * dy
    cross = -s * dx + c * dy
    return along, cross


def cmd_hybrid_autorun(args: argparse.Namespace) -> int:
    core.ensure_can_ready(args.channel, args.bitrate)
    mission = json.loads(Path(args.mission).read_text(encoding="utf-8"))
    mission_projection = mission.get("ground_projection", {})
    projection = core.GroundProjection(
        enabled=bool(mission_projection.get("enabled", False)),
        sensor_height_m=float(mission_projection.get("sensor_height_m", getattr(args, "sensor_height_m", 0.0)) or 0.0),
        body_x_offset_m=float(mission_projection.get("body_x_offset_m", getattr(args, "body_x_offset_m", 0.0)) or 0.0),
        body_y_offset_m=float(mission_projection.get("body_y_offset_m", getattr(args, "body_y_offset_m", 0.0)) or 0.0),
        roll_gain=float(mission_projection.get("roll_gain", getattr(args, "roll_gain", 0.65)) or 0.65),
        pitch_gain=float(mission_projection.get("pitch_gain", getattr(args, "pitch_gain", 1.0)) or 1.0),
        anchor_roll_rad=float(mission_projection.get("anchor_roll_rad", 0.0) or 0.0),
        anchor_pitch_rad=float(mission_projection.get("anchor_pitch_rad", 0.0) or 0.0),
    )
    if not projection.enabled:
        projection = core.projection_from_args(args)
    bound_map_db = str(mission.get("bound_map_db") or "").strip()
    selected_map_db = str(Path(args.db).resolve())
    if bound_map_db and not core.same_map_identity(bound_map_db, selected_map_db):
        raise RuntimeError(
            "Mission map mismatch. "
            f"This mission was recorded on: {Path(bound_map_db).name}, "
            f"but the selected replay map is: {Path(selected_map_db).name}."
        )
    samples = mission.get("samples", [])
    if len(samples) < 2:
        raise RuntimeError("Mission has too few samples.")

    controller = core.FWMiniController(args.interface, args.channel, args.bitrate)
    can_reader = core.CANFeedbackReader(args.interface, args.channel, args.bitrate)
    points = [core._sample_pose(sample) for sample in samples]
    sample_period = float(mission.get("sample_period", 0.2) or 0.2)
    motions, used_fallback = core._repair_missing_motion(samples, sample_period)
    crab_fix_count = core._stabilize_crab_segments(motions)
    soften_count = core._soften_low_speed_turns(motions)
    start_index = core._choose_start_index(points, motions)
    current_gear: str | None = None
    send_state = core.MotionSendState.create()
    send_state.unlock_request_active = True
    for _ in range(2):
        send_state.queue_unlock_sequence()
    core.log("Hybrid autorun queued startup unlock pulses.")
    session: LocalizationSession | None = None
    tracker: core.TFPoseTracker | None = None
    lidar_driver_proc: LidarDriverProcess | None = None
    local_lidar_proc: LocalLidarProcess | None = None
    ros_bridge: LocalLidarRosBridge | None = None
    lidar_driver_log = LOG_DIR / f"lidar_driver_{time.strftime('%Y%m%d_%H%M%S')}.log"
    local_lidar_log = LOG_DIR / f"lidar_local_{time.strftime('%Y%m%d_%H%M%S')}.log"
    use_local_guidance = max(0.0, float(args.local_weight_in_row)) > 1e-6
    try:
        if getattr(args, "reuse_localization", False):
            core.log("Hybrid autorun requested. Reusing the active localization session.")
            tracker = core.create_tracker_with_retry(
                args.map_frame,
                args.base_frame,
                "autorun_final_follow_reuse",
                args.localization_wait_sec,
                "Hybrid autorun",
            )
            pose_raw = core.wait_for_stable_pose(tracker, None, args.localization_wait_sec, "Hybrid autorun")
        else:
            core.log("Hybrid autorun requested. Waiting for localization to succeed first.")
            session, tracker, pose_raw = core.localize_with_retry(
                Path(args.db),
                args.map_frame,
                args.base_frame,
                "autorun_final_follow",
                args.localization_wait_sec,
                "Hybrid autorun",
            )
        if projection.enabled and abs(projection.anchor_roll_rad) < 1e-9 and abs(projection.anchor_pitch_rad) < 1e-9:
            projection = core.anchor_projection_to_pose(projection, pose_raw)
        pose = core.project_pose_to_ground(pose_raw, projection)
        local_lidar_config = _load_local_lidar_gui_config()
        if use_local_guidance:
            lidar_driver_proc = LidarDriverProcess(lidar_driver_log)
            lidar_driver_proc.start()
            core.log(f"lidar driver subprocess started. Raw log: {lidar_driver_log}")
            time.sleep(1.0)
            local_lidar_proc = LocalLidarProcess(args, local_lidar_log, local_lidar_config)
            local_lidar_proc.start()
            core.log(f"lidar local subprocess started. Raw log: {local_lidar_log}")
            ros_bridge = LocalLidarRosBridge(args.ros_cmd_vel_topic, args.ros_status_topic, args.ros_drive_mode_topic)
            if local_lidar_config:
                core.log(f"Hybrid autorun uses autorunlida config: {LOCAL_ROW_CONFIG}")
            if not _wait_for_local_lidar_ready(ros_bridge, args, local_lidar_config=local_lidar_config):
                core.log("lidar local guidance was not ready before replay; hybrid will keep waiting instead of using global when global weight is 0.")
        else:
            core.log("Hybrid autorun local weight is 0. lidar local guidance is disabled; running pure global replay.")
        if used_fallback:
            core.log("Hybrid autorun fallback: mission motion data was missing, using path-derived motion estimates.")
        if crab_fix_count:
            core.log(f"Hybrid autorun cleanup: stabilized {crab_fix_count} crab transition samples from the mission.")
        if soften_count:
            core.log(f"Hybrid autorun cleanup: softened {soften_count} low-speed turns from the mission.")
        core._approach_start_point(tracker, controller, can_reader, points, start_index, send_state)
        core.log(f"Hybrid autorun starts replay from mission sample #{start_index}.")
        last_cmd_log = 0.0
        last_feedback_log = 0.0
        current_mode = "global"
        current_gear = None
        while not core.STOP_REQUESTED and start_index < len(points):
            assert tracker is not None
            pose_raw = tracker.lookup()
            if pose_raw is None:
                time.sleep(0.05)
                continue
            pose = core.project_pose_to_ground(pose_raw, projection)
            can_reader.poll(timeout=0.0, limit=20)
            snapshot = can_reader.snapshot()
            io_state = snapshot.get("io", {})
            if bool(io_state.get("remote_control", False)):
                if not send_state.remote_paused:
                    send_state.remote_paused = True
                    send_state.reset_motion()
                    core.log("Remote controller took over. Hybrid autorun paused and progress is preserved.")
                if ros_bridge is not None:
                    cruise_vx, offset_px, direction_angle_deg = _local_lidar_drive_settings(local_lidar_config, reverse=False, args=args)
                    ros_bridge.publish_mode(
                        enable=False,
                        reverse=False,
                        cruise_vx=cruise_vx,
                        gear="4t4d",
                        low_beam=args.line_low_beam,
                        target_center_offset_px=offset_px,
                        vehicle_direction_angle_deg=direction_angle_deg,
                    )
                time.sleep(0.10)
                continue
            if send_state.remote_paused:
                send_state.remote_paused = False
                send_state.reset_motion()
                nearest_resume = _find_tracking_index_same_mode(
                    points,
                    motions,
                    start_index,
                    pose,
                    current_gear,
                    window=24,
                )
                start_index = max(start_index, nearest_resume)
                current_gear = None
                core.log(f"Remote controller released. Hybrid autorun resumed from nearest mission sample #{start_index}.")

            crab_locked_active = send_state.crab_locked_until >= start_index and send_state.crab_target_index >= 0
            if crab_locked_active:
                nearest_index = start_index
            else:
                nearest_index = _find_tracking_index_same_mode(
                    points,
                    motions,
                    start_index,
                    pose,
                    current_gear,
                    window=18,
                )
                start_index = max(start_index, nearest_index)
            motion_here = motions[min(start_index, len(motions) - 1)]
            crab_tail_release_index = _skip_crab_tail_index(motions, start_index)
            if crab_tail_release_index > start_index:
                start_index = crab_tail_release_index
                current_gear = None
                send_state.last_sent_gear = None
                motion_here = motions[min(start_index, len(motions) - 1)]
            reversing_here = _safe_float(motion_here.get("vx"), 0.0) < -0.03
            if reversing_here:
                start_index = _advance_reverse_progress_index(points, motions, start_index, pose, current_gear)
                motion_here = motions[min(start_index, len(motions) - 1)]
                reversing_here = _safe_float(motion_here.get("vx"), 0.0) < -0.03
            else:
                upcoming_crab_for_progress = core._find_future_gear_start(motions, start_index, "crab", limit=20)
                crab_is_near = (
                    upcoming_crab_for_progress is not None
                    and pose.distance_to(points[upcoming_crab_for_progress]) <= 0.45
                )
                if not crab_is_near:
                    start_index = _advance_dense_tracking_index(points, motions, start_index, pose, current_gear)
                    start_index = _advance_tracking_progress_index(points, motions, start_index, pose, current_gear)
                    motion_here = motions[min(start_index, len(motions) - 1)]
                    reversing_here = _safe_float(motion_here.get("vx"), 0.0) < -0.03
            if not reversing_here:
                imminent_reverse_index = _find_next_reverse_index(motions, start_index, limit=3)
                if imminent_reverse_index is not None and imminent_reverse_index <= start_index + 2:
                    reverse_entry_dist = pose.distance_to(points[imminent_reverse_index])
                    if reverse_entry_dist <= 0.25:
                        start_index = imminent_reverse_index
                        motion_here = motions[min(start_index, len(motions) - 1)]
                        reversing_here = _safe_float(motion_here.get("vx"), 0.0) < -0.03
                        if reversing_here:
                            core.log(
                                f"Hybrid autorun snapped into reverse segment at sample #{start_index} "
                                f"(entry_dist={reverse_entry_dist:.2f}m)."
                            )
            if send_state.crab_locked_until >= start_index and send_state.crab_target_index >= 0:
                gear = "crab"
                target_index = core._select_crab_progress_target(
                    start_index,
                    send_state.crab_locked_until,
                    max(start_index + 1, send_state.crab_target_index),
                )
                target = points[target_index]
                motion = motions[target_index]
                if core._resolve_replay_gear(motion, None) != "crab":
                    send_state.crab_locked_until = -1
                    send_state.crab_target_index = -1
                    gear = core._resolve_replay_gear(motion_here, current_gear)
                    if reversing_here:
                        target_index = _select_reverse_target_index(points, motions, start_index, current_gear)
                    else:
                        target_index = _select_lookahead_target_index(
                            points,
                            motions,
                            start_index,
                            current_gear,
                        )
                    target = points[target_index]
                    motion = motions[target_index]
            else:
                if reversing_here:
                    target_index = _select_reverse_target_index(points, motions, start_index, current_gear)
                else:
                    target_index = _select_lookahead_target_index(
                        points,
                        motions,
                        start_index,
                        current_gear,
                    )
                target = points[target_index]
                motion = motions[target_index]
                gear = core._resolve_replay_gear(motion, current_gear)
            if gear == "crab" and (
                send_state.crab_locked_until < start_index or send_state.crab_target_index < 0
            ):
                crab_end = core._find_gear_segment_end(motions, start_index, "crab")
                crab_active = core._find_first_active_crab_index(motions, start_index, crab_end)
                crab_active_end = core._find_last_active_crab_index(motions, crab_active, crab_end)
                send_state.crab_locked_until = crab_active_end
                send_state.crab_target_index = crab_active
                send_state.crab_best_dist = float("inf")
                send_state.crab_diverge_count = 0
                target_index = core._select_crab_progress_target(
                    start_index,
                    crab_active_end,
                    crab_active,
                )
                target = points[target_index]
                motion = motions[target_index]
            if gear != "crab":
                upcoming_crab = core._find_future_gear_start(motions, start_index, "crab", limit=20)
                if upcoming_crab is not None:
                    crab_entry = points[upcoming_crab]
                    if pose.distance_to(crab_entry) <= 0.50:
                        crab_end = core._find_gear_segment_end(motions, upcoming_crab, "crab")
                        crab_active = core._find_first_active_crab_index(motions, upcoming_crab, crab_end)
                        crab_active_end = core._find_last_active_crab_index(motions, crab_active, crab_end)
                        send_state.crab_locked_until = crab_active_end
                        send_state.crab_target_index = crab_active
                        send_state.crab_best_dist = float("inf")
                        send_state.crab_diverge_count = 0
                        target_index = core._select_crab_progress_target(
                            start_index,
                            crab_active_end,
                            crab_active,
                        )
                        target = points[target_index]
                        motion = motions[target_index]
                        gear = "crab"
            dist = pose.distance_to(target)
            row_switch_mode = gear == "crab" or abs(_safe_float(motion.get("vy"), 0.0)) > 0.05
            reversing_mode = gear == "reverse" or _safe_float(motion.get("vx"), 0.0) < -0.03
            tracking_heading = core._tracking_heading(points, motions, target_index, gear)
            upcoming_row_switch_index = None if row_switch_mode else _find_next_row_switch_index(motions, start_index + 1)
            dist_to_row_switch = None
            if upcoming_row_switch_index is not None:
                dist_to_row_switch = pose.distance_to(points[upcoming_row_switch_index])
            if reversing_mode and upcoming_row_switch_index is not None and dist_to_row_switch is not None and dist_to_row_switch <= 0.80:
                start_index = max(start_index + 1, upcoming_row_switch_index)
                current_gear = None
                send_state.last_sent_gear = None
                send_state.crab_best_dist = float("inf")
                send_state.crab_diverge_count = 0
                core.log(f"Hybrid autorun reverse-to-row-switch handoff at sample #{start_index}.")
                continue
            next_reverse_index = None if reversing_mode else _find_next_reverse_index(motions, start_index + 1)
            dist_to_reverse_start = None
            next_reverse_start_index = None
            if next_reverse_index is not None:
                next_reverse_start_index = _find_reverse_segment_start(motions, next_reverse_index)
                if next_reverse_start_index is not None:
                    dist_to_reverse_start = pose.distance_to(points[next_reverse_start_index])
            dist_since_reverse_start = None
            if reversing_mode:
                reverse_start_index = _find_reverse_segment_start(motions, start_index)
                if reverse_start_index is not None:
                    dist_since_reverse_start = pose.distance_to(points[reverse_start_index])
            near_finish = target_index >= len(points) - 4

            if gear != current_gear:
                core.log(f"Hybrid autorun mode switched to {gear}.")
                current_gear = gear

            if gear == "crab":
                if dist + 0.03 < send_state.crab_best_dist:
                    send_state.crab_best_dist = dist
                    send_state.crab_diverge_count = 0
                elif dist > send_state.crab_best_dist + 0.20:
                    send_state.crab_diverge_count += 1
                else:
                    send_state.crab_diverge_count = max(0, send_state.crab_diverge_count - 1)
                if send_state.crab_diverge_count >= 6:
                    core.log(
                        f"Hybrid autorun crab fallback: target distance kept increasing "
                        f"(best={send_state.crab_best_dist:.2f}, now={dist:.2f}). Returning to 4t4d tracking."
                    )
                    send_state.crab_locked_until = -1
                    send_state.crab_target_index = -1
                    send_state.crab_best_dist = float("inf")
                    send_state.crab_diverge_count = 0
                    start_index = _skip_crab_tail_index(motions, start_index)
                    current_gear = None
                    send_state.last_sent_gear = None
                    continue
                dist = pose.distance_to(target)
                ref_index = max(0, min(start_index, len(points) - 1))
                axis_err = _crab_axis_error(pose, points[ref_index], target)
                reached_crab_exit = dist <= 0.20 or abs(axis_err) <= 0.20
                if reached_crab_exit:
                    resume_seed = min(
                        len(points) - 1,
                        max(send_state.crab_locked_until + 1, start_index + 1),
                    )
                    next_forward_4t4d = _find_next_forward_4t4d_index(
                        motions,
                        resume_seed,
                        limit=72,
                    )
                    if next_forward_4t4d is not None:
                        resume_seed = next_forward_4t4d
                    resume_seed = _skip_hold_4t4d_cluster(points, motions, resume_seed)
                    start_index = _find_forward_resume_index(
                        points,
                        motions,
                        resume_seed,
                        pose,
                        limit=24,
                    )
                    send_state.crab_locked_until = -1
                    send_state.crab_target_index = -1
                    send_state.crab_best_dist = float("inf")
                    send_state.crab_diverge_count = 0
                    current_gear = None
                    send_state.last_sent_gear = None
                    core.log(f"Hybrid autorun crab segment finished near sample #{start_index - 1}. Returning to 4t4d tracking.")
                    continue
                if near_finish and dist < 0.10:
                    core._hold_current_gear_stop(controller, send_state, current_gear)
                    core.log(f"Hybrid autorun reached final area near mission end (dist={dist:.2f}). Stopping without final alignment.")
                    break
            else:
                dist = pose.distance_to(target)
                tracking_heading = core._tracking_heading(points, motions, target_index, gear)
                yaw_err = core.normalize_angle(tracking_heading - pose.yaw)
                path_along_err, path_cross_err = _path_frame_error(pose, target, tracking_heading)
                passed_target = (not reversing_mode) and path_along_err < -0.03 and abs(path_cross_err) < 0.12
                reverse_heading_ok = abs(math.degrees(yaw_err)) < 12.0
                reverse_cross_ok = abs(path_cross_err) < 0.12
                reverse_passed_target = reversing_mode and path_along_err > 0.03 and reverse_cross_ok
                reverse_reached_target = reversing_mode and dist < 0.12 and reverse_cross_ok and reverse_heading_ok
                if (
                    (not reversing_mode and dist < 0.12 and abs(math.degrees(yaw_err)) < 12.0)
                    or passed_target
                    or reverse_passed_target
                    or reverse_reached_target
                ):
                    send_state.crab_best_dist = float("inf")
                    send_state.crab_diverge_count = 0
                    if reversing_mode:
                        start_index = min(len(points) - 1, max(start_index + 1, min(target_index, start_index + 6)))
                    else:
                        start_index = min(len(points) - 1, target_index + 1)
                    continue
                _, lateral_err = core._body_frame_error(pose, target)
                if (not reversing_mode) and abs(lateral_err) > 0.07:
                    protected_target_index = _select_lookahead_target_index(
                        points,
                        motions,
                        start_index,
                        current_gear,
                        lookahead_m=0.14,
                        max_span=12,
                    )
                    if protected_target_index != target_index:
                        target_index = protected_target_index
                        target = points[target_index]
                        motion = motions[target_index]
                        near_finish = target_index >= len(points) - 4
                        tracking_heading = core._tracking_heading(points, motions, target_index, gear)
                        yaw_err = core.normalize_angle(tracking_heading - pose.yaw)
                        dist = pose.distance_to(target)
                if not reversing_mode:
                    advanced_start = _skip_hold_4t4d_cluster(points, motions, start_index)
                    if advanced_start > start_index:
                        start_index = advanced_start
                        target_index = _select_lookahead_target_index(
                            points,
                            motions,
                            start_index,
                            current_gear,
                        )
                        target = points[target_index]
                        motion = motions[target_index]
                        tracking_heading = core._tracking_heading(points, motions, target_index, gear)
                        yaw_err = core.normalize_angle(tracking_heading - pose.yaw)
                        dist = pose.distance_to(target)
            path_cross_for_log = 0.0
            if gear != "crab":
                _, path_cross_for_log = _path_frame_error(pose, target, tracking_heading)

            command_motion = motion
            if gear == "crab" and abs(_safe_float(command_motion.get("vy"), 0.0)) < 0.03:
                crab_vy = _crab_reference_vy(points, pose, motions, start_index, target_index, send_state)
                if abs(crab_vy) >= 0.03:
                    command_motion = dict(motion)
                    command_motion["vy"] = crab_vy

            g_vx, g_vy, g_wz, dist, heading_err_deg = _compute_global_command(
                pose,
                points[start_index],
                target,
                command_motion,
                gear,
                send_state,
                tracking_heading,
            )
            if near_finish and dist < 0.10:
                core._hold_current_gear_stop(controller, send_state, current_gear)
                core.log(f"Hybrid autorun reached final area near mission end (dist={dist:.2f}). Stopping without final alignment.")
                break

            local_enabled_by_weight = max(0.0, min(1.0, float(args.local_weight_in_row))) > 1e-6
            local_enable = (not row_switch_mode) and local_enabled_by_weight
            if ros_bridge is not None:
                cruise_vx, offset_px, direction_angle_deg = _local_lidar_drive_settings(
                    local_lidar_config,
                    reverse=reversing_here,
                    args=args,
                )
                ros_bridge.publish_mode(
                    enable=local_enable,
                    reverse=reversing_here,
                    cruise_vx=cruise_vx,
                    gear="4t4d",
                    low_beam=args.line_low_beam,
                    target_center_offset_px=offset_px,
                    vehicle_direction_angle_deg=direction_angle_deg,
                )
                local_status = ros_bridge.status_snapshot()
                local_cmd = ros_bridge.cmd_snapshot()
            else:
                local_status = LineStatus()
                local_cmd = TwistCommand()

            if (
                local_status.obstacle_blocked
                and not reversing_mode
                and next_reverse_start_index is not None
                and dist_to_reverse_start is not None
                and dist_to_reverse_start <= max(0.20, float(getattr(args, "row_switch_blend_start_dist", 3.0)))
            ):
                start_index = max(start_index + 1, next_reverse_start_index)
                current_gear = None
                send_state.last_sent_gear = None
                core.log(f"Hybrid autorun row end detected by lidar local guidance. Switching to reverse mission sample #{start_index}.")
                continue

            global_weight, local_weight, mode_name, allow_stale_local_hold = _compute_hybrid_weights(
                args=args,
                row_switch_mode=row_switch_mode,
                reversing_mode=reversing_mode,
                near_finish=near_finish,
                local_status=local_status,
                local_cmd=local_cmd,
                dist_to_row_switch=dist_to_row_switch,
                dist_to_reverse_start=dist_to_reverse_start,
                dist_since_reverse_start=dist_since_reverse_start,
            )

            body_cmd = _build_body_command(
                gear,
                (g_vx, g_vy, g_wz),
                local_cmd,
                local_status,
                global_weight,
                local_weight,
                allow_stale_local_hold,
            )
            body_cmd.gear = gear
            if (
                gear == "crab"
                and dist < 0.16
                and abs(body_cmd.vy) < 0.10
                and target_index >= max(start_index + 1, send_state.crab_locked_until - 1)
            ):
                start_index = min(len(points) - 1, target_index + 1)
                if start_index > send_state.crab_locked_until:
                    send_state.crab_locked_until = -1
                    send_state.crab_target_index = -1
                    send_state.crab_best_dist = float("inf")
                    send_state.crab_diverge_count = 0
                    current_gear = None
                    send_state.last_sent_gear = None
                continue
            # Preserve local row-following behavior as much as possible once the
            # hybrid stage/weight logic has decided to use it. We still keep
            # the high-level mode switching and weight blending, but avoid a
            # second round of motion smoothing here because it noticeably
            # changes the feel of the lidar local controller.
            if gear == "crab":
                send_state.last_vx = 0.0
                body_cmd.vx = 0.0
            waiting_unlock, unlock_now, sent_vx, sent_vy, sent_wz = core._send_drive(
                controller,
                send_state,
                snapshot,
                gear,
                body_cmd.vx,
                body_cmd.vy,
                body_cmd.wz,
            )
            now = time.monotonic()
            if mode_name != current_mode:
                current_mode = mode_name
                core.log(
                    f"Hybrid control switched to {mode_name}: "
                    f"local_state={local_status.state} row_switch={row_switch_mode} "
                    f"gw={global_weight:.2f} lw={local_weight:.2f} "
                    f"switch_dist={(dist_to_row_switch if dist_to_row_switch is not None else -1.0):.2f}"
                )
            if now - last_cmd_log >= 1.0:
                core.log(
                    f"Hybrid command: mode={mode_name} gear={gear} "
                    f"global_vx={g_vx:.2f} global_vy={g_vy:.2f} global_wz={g_wz:.1f} "
                    f"local_vx={local_cmd.vx:.2f} local_wz={local_cmd.wz:.2f} "
                    f"sent_vx={sent_vx:.2f} sent_vy={sent_vy:.2f} sent_wz={sent_wz:.1f} "
                    f"nearest_index={nearest_index} target_index={target_index} dist={dist:.2f} "
                    f"cross_err={path_cross_for_log:.3f} "
                    f"switch_dist={(dist_to_row_switch if dist_to_row_switch is not None else -1.0):.2f}"
                )
                if waiting_unlock or unlock_now:
                    core.log(f"Hybrid unlock: waiting_unlock={waiting_unlock} unlock_pulse={unlock_now}")
                last_cmd_log = now
            if now - last_feedback_log >= 1.0:
                core._log_feedback("Hybrid autorun", snapshot, pose)
                core.log(
                    f"Hybrid local: state={local_status.state} found={local_status.found} "
                    f"blocked={local_status.obstacle_blocked} lost_frames={local_status.lost_frames} "
                    f"fresh={local_status.fresh} heading_err_deg={heading_err_deg:.1f}"
                )
                last_feedback_log = now
            time.sleep(max(0.03, min(0.10, sample_period)))

        try:
            core._hold_current_gear_stop(controller, send_state, current_gear)
        except Exception as exc:
            core.log(f"Hybrid autorun final stop command skipped: {exc}")
        core.log("Hybrid autorun finished.")
        return 0
    finally:
        try:
            if ros_bridge is not None:
                cruise_vx, offset_px, direction_angle_deg = _local_lidar_drive_settings(local_lidar_config, reverse=False, args=args)
                ros_bridge.publish_mode(
                    enable=False,
                    reverse=False,
                    cruise_vx=cruise_vx,
                    gear="4t4d",
                    low_beam=args.line_low_beam,
                    target_center_offset_px=offset_px,
                    vehicle_direction_angle_deg=direction_angle_deg,
                )
        except Exception:
            pass
        try:
            core._hold_current_gear_stop(controller, send_state, current_gear)
        except Exception as exc:
            core.log(f"Hybrid autorun cleanup stop command skipped: {exc}")
        can_reader.close()
        controller.close()
        if ros_bridge is not None:
            ros_bridge.close()
        if local_lidar_proc is not None:
            local_lidar_proc.stop()
        if lidar_driver_proc is not None:
            lidar_driver_proc.stop()
        if session is not None:
            session.stop()


def _add_hybrid_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db", required=True)
    parser.add_argument("--mission", required=True)
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--base-frame", default="odin1_base_link")
    parser.add_argument("--localization-wait-sec", type=float, default=30.0)
    parser.add_argument("--reuse-localization", action="store_true")
    parser.add_argument("--line-cruise-vx", type=float, default=0.12)
    parser.add_argument("--line-target-center-offset-px", type=float, default=0.0)
    parser.add_argument("--line-vehicle-direction-angle-deg", type=float, default=0.0)
    parser.add_argument("--line-steer-sign", type=float, default=-1.0)
    parser.add_argument("--line-kp-offset", type=float, default=7.0)
    parser.add_argument("--line-kp-heading", type=float, default=0.08)
    parser.add_argument("--line-max-wz", type=float, default=1.6)
    parser.add_argument("--line-period-ms", type=int, default=20)
    parser.add_argument("--line-lost-stop-frames", type=int, default=12)
    parser.add_argument("--line-low-beam", action="store_true")
    parser.add_argument("--ros-cmd-vel-topic", default="/lidarun/cmd_vel")
    parser.add_argument("--ros-status-topic", default="/lidarun/status")
    parser.add_argument("--ros-drive-mode-topic", default="/lidarun/drive_mode")
    parser.add_argument("--local-weight-in-row", type=float, default=0.75)
    parser.add_argument("--global-weight-in-row", type=float, default=0.25)
    parser.add_argument("--row-switch-blend-start-dist", type=float, default=3.0)
    parser.add_argument("--row-switch-full-global-dist", type=float, default=1.5)
    parser.add_argument("--lidar-scan-topic", default="/scan")
    parser.add_argument("--lidar-min-speed", type=float, default=0.04)
    parser.add_argument("--lidar-max-wz-deg", type=float, default=1.2)
    parser.add_argument("--lidar-max-heading-wz-deg", type=float, default=0.5)
    parser.add_argument("--lidar-k-lat", type=float, default=1.2)
    parser.add_argument("--lidar-k-heading", type=float, default=0.45)
    parser.add_argument("--lidar-center-y-target", type=float, default=0.0)
    parser.add_argument("--lidar-heading-conflict-error-y", type=float, default=0.012)
    parser.add_argument("--lidar-heading-conflict-scale", type=float, default=0.0)
    parser.add_argument("--lidar-row-width", type=float, default=0.60)
    parser.add_argument("--lidar-min-row-width", type=float, default=0.48)
    parser.add_argument("--lidar-max-row-width", type=float, default=0.78)
    parser.add_argument("--lidar-lookahead-x", type=float, default=0.75)
    parser.add_argument("--lidar-forward-lookahead-x", type=float, default=0.6)
    parser.add_argument("--lidar-reverse-lookahead-x", type=float, default=-0.6)
    parser.add_argument("--lidar-forward-min", type=float, default=0.15)
    parser.add_argument("--lidar-forward-max", type=float, default=1.60)
    parser.add_argument("--lidar-lateral-limit", type=float, default=0.75)
    parser.add_argument("--lidar-range-min", type=float, default=0.05)
    parser.add_argument("--lidar-range-max", type=float, default=6.0)
    parser.add_argument("--lidar-bin-size", type=float, default=0.20)
    parser.add_argument("--lidar-min-points", type=int, default=16)
    parser.add_argument("--lidar-min-bins", type=int, default=2)
    parser.add_argument("--lidar-min-line-bins", type=int, default=4)
    parser.add_argument("--lidar-min-side-points-per-bin", type=int, default=2)
    parser.add_argument("--lidar-boundary-width-tolerance-m", type=float, default=0.0)
    parser.add_argument("--lidar-center-deadband", type=float, default=0.03)
    parser.add_argument("--lidar-left-percentile", type=float, default=20.0)
    parser.add_argument("--lidar-right-percentile", type=float, default=80.0)
    parser.add_argument("--lidar-sensor-yaw-deg", type=float, default=180.0)
    parser.add_argument("--lidar-yaw-correction-deg", type=float, default=0.0)
    parser.add_argument("--lidar-x-offset-m", type=float, default=0.0)
    parser.add_argument("--lidar-y-offset-m", type=float, default=0.0)
    parser.add_argument("--lidar-vehicle-width", type=float, default=0.40)
    parser.add_argument("--lidar-control-deadband-y", type=float, default=0.004)
    parser.add_argument("--lidar-slow-error-y", type=float, default=0.04)
    parser.add_argument("--lidar-stop-error-y", type=float, default=0.065)
    parser.add_argument("--lidar-slow-heading-rad", type=float, default=0.18)
    parser.add_argument("--lidar-safety-margin", type=float, default=0.04)
    parser.add_argument("--lidar-center-jump-reject", type=float, default=0.16)
    parser.add_argument("--lidar-one-side-center-jump-reject", type=float, default=0.30)
    parser.add_argument("--lidar-center-y-reject-abs", type=float, default=0.16)
    parser.add_argument("--lidar-raw-center-out-of-range", type=float, default=0.20)
    parser.add_argument("--lidar-one-side-raw-center-out-of-range", type=float, default=0.28)
    parser.add_argument("--lidar-history-window-s", type=float, default=0.40)
    parser.add_argument("--lidar-min-center-history", type=int, default=2)
    parser.add_argument("--lidar-center-y-alpha", type=float, default=0.75)
    parser.add_argument("--lidar-center-y-max-jump", type=float, default=0.08)
    parser.add_argument("--lidar-one-side-stop-error-y", type=float, default=0.12)
    parser.add_argument("--lidar-one-side-safety-stop-band", type=float, default=0.12)
    parser.add_argument("--lidar-control-period", type=float, default=0.05)
    parser.add_argument("--lidar-status-period", type=float, default=0.20)
    parser.add_argument("--lidar-scan-timeout", type=float, default=0.30)
    parser.add_argument("--lidar-forward-lost-hold-sec", type=float, default=0.35)
    parser.add_argument("--lidar-forward-lost-stop-sec", type=float, default=0.50)
    parser.add_argument("--lidar-forward-lost-hold-wz-scale", type=float, default=0.5)
    parser.add_argument("--lidar-forward-lost-hold-max-wz-deg", type=float, default=0.6)
    parser.add_argument("--lidar-reverse-min-speed", type=float, default=0.04)
    parser.add_argument("--lidar-reverse-one-side-speed", type=float, default=0.10)
    parser.add_argument("--lidar-reverse-both-sides-speed", type=float, default=0.12)
    parser.add_argument("--lidar-reverse-min-wz-deg", type=float, default=2.5)
    parser.add_argument("--lidar-reverse-max-wz-deg", type=float, default=5.0)
    parser.add_argument("--lidar-reverse-one-side-max-wz-deg", type=float, default=5.0)
    parser.add_argument("--lidar-reverse-wz-enable-error-y", type=float, default=0.004)
    parser.add_argument("--lidar-reverse-wz-enable-heading-deg", type=float, default=1.0)
    parser.add_argument("--lidar-reverse-min-wz-error-y", type=float, default=0.025)
    parser.add_argument("--lidar-reverse-sign-flip-guard-error-y", type=float, default=0.02)
    parser.add_argument("--lidar-reverse-sign-flip-guard-last-wz-deg", type=float, default=1.5)
    parser.add_argument("--lidar-reverse-sign-hold-error-y", type=float, default=0.0)
    parser.add_argument("--lidar-reverse-both-sides-k-lat", type=float, default=1.2)
    parser.add_argument("--lidar-reverse-both-sides-k-heading", type=float, default=0.15)
    parser.add_argument("--lidar-reverse-one-side-k-lat", type=float, default=1.0)
    parser.add_argument("--lidar-k-reverse-lat", type=float, default=25.0)
    parser.add_argument("--lidar-k-reverse-heading", type=float, default=0.03)
    parser.add_argument("--lidar-reverse-steer-sign", type=float, default=-1.0)
    parser.add_argument("--lidar-reverse-heading-conflict-error-y", type=float, default=0.01)
    parser.add_argument("--lidar-reverse-heading-max-ratio", type=float, default=0.35)
    parser.add_argument("--lidar-reverse-error-stop", type=float, default=0.28)
    parser.add_argument("--lidar-reverse-wz-smoothing-alpha", type=float, default=0.0)
    parser.add_argument("--lidar-reverse-lost-hold-sec", type=float, default=0.35)
    parser.add_argument("--lidar-reverse-lost-stop-sec", type=float, default=0.80)
    parser.add_argument("--lidar-reverse-lost-hold-max-wz-deg", type=float, default=1.5)
    parser.add_argument("--lidar-reverse-lost-soft-max-wz-deg", type=float, default=0.8)
    parser.add_argument("--lidar-max-wz-delta-deg-per-cycle", type=float, default=1.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="UGV autorunlida: Odin global localization + lidar local row guidance"
    )
    parser.add_argument("--interface", default="socketcan")
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--bitrate", type=int, default=500000)
    parser.add_argument("--sensor-height-m", type=float, default=1.2)
    parser.add_argument("--body-x-offset-m", type=float, default=0.0)
    parser.add_argument("--body-y-offset-m", type=float, default=0.0)
    parser.add_argument("--roll-gain", type=float, default=0.65)
    parser.add_argument("--pitch-gain", type=float, default=1.0)
    sub = parser.add_subparsers(dest="command", required=True)

    map_p = sub.add_parser("map", help="Start Odin SLAM mapping and save a .bin map")
    map_p.add_argument("--map-name", default="")
    map_p.add_argument("--viz", choices=["on", "off"], default="off")
    map_p.add_argument("--recorddata", action="store_true", help="Enable MindCloud-compatible recorddata during mapping")
    map_p.set_defaults(func=cmd_map)

    loc_p = sub.add_parser("localization", help="Run Odin relocalization only")
    loc_p.add_argument("--db", required=True, help="Path to the Odin .bin map file")
    loc_p.add_argument("--map-frame", default="map")
    loc_p.add_argument("--base-frame", default="odin1_base_link")
    loc_p.add_argument("--localization-wait-sec", type=float, default=30.0)
    loc_p.set_defaults(func=core.cmd_localization)

    rec_p = sub.add_parser("record", help="Relocalize first, then record a taught path")
    rec_p.add_argument("--db", required=True, help="Path to the Odin .bin map file")
    rec_p.add_argument("--mission-name", default="")
    rec_p.add_argument("--map-frame", default="map")
    rec_p.add_argument("--base-frame", default="odin1_base_link")
    rec_p.add_argument("--sample-period", type=float, default=0.2)
    rec_p.add_argument("--localization-wait-sec", type=float, default=30.0)
    rec_p.add_argument("--reuse-localization", action="store_true")
    rec_p.set_defaults(func=core.cmd_record)

    run_p = sub.add_parser("autorun", help="Hybrid replay: global mission + lidar row following")
    _add_hybrid_args(run_p)
    run_p.set_defaults(func=cmd_hybrid_autorun)

    replay_p = sub.add_parser("replay", help="Alias of autorun")
    _add_hybrid_args(replay_p)
    replay_p.set_defaults(func=cmd_hybrid_autorun)

    return parser


def main() -> int:
    install_signal_handlers()
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
