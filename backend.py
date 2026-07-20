#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
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


def _validate_odin_config_text(text: str, mode: int, *, map_path: Path | None = None) -> None:
    if not text.strip():
        raise RuntimeError("Generated Odin config is empty.")
    required_keys = (
        "custom_map_mode",
        "recorddata",
        "sendimu",
        "sendodom",
        "send_odom_baselink_tf",
        "relocalization_map_abs_path",
        "mapping_result_dest_dir",
        "mapping_result_file_name",
        "resetalgo",
    )
    missing = [key for key in required_keys if not re.search(rf"^\s*{re.escape(key)}\s*:", text, flags=re.MULTILINE)]
    if missing:
        raise RuntimeError(f"Generated Odin config is missing required key(s): {', '.join(missing)}")
    if mode == 2:
        if map_path is None:
            match = re.search(r'^\s*relocalization_map_abs_path\s*:\s*"?([^"\n]*)"?\s*$', text, flags=re.MULTILINE)
            if match is None or not match.group(1).strip():
                raise RuntimeError("Generated Odin relocalization config has an empty map path.")
        else:
            if not map_path.exists() or map_path.stat().st_size <= 0:
                raise RuntimeError(f"Relocalization map file is missing or empty: {map_path}")
            expected = f'relocalization_map_abs_path: "{map_path}"'
            if expected not in text:
                raise RuntimeError(f"Generated Odin relocalization config does not reference the selected map: {map_path}")


def _validate_odin_config_file(config_path: Path, mode: int, *, map_path: Path | None = None) -> None:
    if not config_path.exists():
        raise RuntimeError(f"Odin config file was not created: {config_path}")
    if config_path.stat().st_size <= 0:
        raise RuntimeError(f"Odin config file is empty: {config_path}")
    _validate_odin_config_text(config_path.read_text(encoding="utf-8"), mode, map_path=map_path)


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
    if not SHADOW_ODIN_CONFIG.exists() or SHADOW_ODIN_CONFIG.stat().st_size <= 0:
        raise RuntimeError(f"Shadow Odin config is missing or empty: {SHADOW_ODIN_CONFIG}")
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
        if not map_path.exists() or map_path.stat().st_size <= 0:
            raise RuntimeError(f"Relocalization map file is missing or empty: {map_path}")
        text = _replace_yaml_scalar(text, "relocalization_map_abs_path", f'"{map_path}"')
        text = _replace_yaml_scalar(text, "mapping_result_dest_dir", '""')
        text = _replace_yaml_scalar(text, "mapping_result_file_name", '""')
        # Force the device SLAM/relocalization state to reset before each
        # relocalization session. Without this, repeated relocalization in the
        # same boot can get stuck waiting for a valid pose until the device is
        # power-cycled.
        text = _replace_yaml_scalar(text, "resetalgo", "1")
    _validate_odin_config_text(text, mode, map_path=map_path)
    temp_path = TEMP_CONFIG_DIR / f"odin_mode_{mode}_{int(time.time() * 1000)}.yaml"
    temp_path.write_text(text, encoding="utf-8")
    _validate_odin_config_file(temp_path, mode, map_path=map_path)
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
        _validate_odin_config_file(self.config_path, 2 if "odin_mode_2_" in self.config_path.name else 1)
        if not self.target_config.exists():
            raise RuntimeError(f"Shadow Odin config file does not exist: {self.target_config}")
        if not os.access(self.target_config, os.W_OK):
            raise RuntimeError(f"Shadow Odin config file is not writable: {self.target_config}")
        shutil.copy2(self.config_path, self.target_config)
        if self.target_config.stat().st_size <= 0:
            raise RuntimeError(f"Shadow Odin config override produced an empty file: {self.target_config}")
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


class DirectLocalLidarController:
    def __init__(self, args: argparse.Namespace, local_lidar_config: dict[str, Any] | None = None) -> None:
        self._status_lock = threading.Lock()
        self._cmd_lock = threading.Lock()
        self._status = LineStatus()
        self._cmd = TwistCommand()
        self._module = self._load_standalone_module()
        self._node = self._build_node(args, local_lidar_config or {})
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._spin, name="autorun-direct-lidar-local", daemon=True)
        self._thread.start()

    def _load_standalone_module(self):
        standalone_dir = PROJECT_ROOT
        standalone_file = standalone_dir / "plant_lidar_centerline_follower.py"
        if not standalone_file.exists():
            raise FileNotFoundError(f"Standalone local follower not found: {standalone_file}")
        if str(standalone_dir) not in sys.path:
            sys.path.insert(0, str(standalone_dir))
        spec = importlib.util.spec_from_file_location("autorun_standalone_lidar_local", standalone_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load standalone local follower: {standalone_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("autorun_standalone_lidar_local", module)
        spec.loader.exec_module(module)
        return module

    def _build_args(self, args: argparse.Namespace, config: dict[str, Any]) -> argparse.Namespace:
        cruise_vx, _offset_px, _direction_angle_deg = _local_lidar_drive_settings(config, reverse=False, args=args)
        saved_argv = sys.argv[:]
        try:
            sys.argv = [saved_argv[0]]
            follower_args = self._module.parse_args()
        finally:
            sys.argv = saved_argv
        calibration_path = PROJECT_ROOT / "config" / "lidar_calibration.json"
        calibration: dict[str, Any] = {}
        try:
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            if not isinstance(calibration, dict):
                calibration = {}
        except Exception:
            calibration = {}
        follower_args.scan_topic = str(args.lidar_scan_topic)
        follower_args.status_topic = str(args.ros_status_topic)
        follower_args.speed = abs(float(cruise_vx))
        follower_args.min_speed = float(args.lidar_min_speed)
        follower_args.max_wz_deg = float(args.lidar_max_wz_deg)
        follower_args.max_heading_wz_deg = float(args.lidar_max_heading_wz_deg)
        follower_args.k_lat = float(args.lidar_k_lat)
        follower_args.k_heading = float(args.lidar_k_heading)
        follower_args.center_y_target = _config_float(config, "center_y_target", float(args.lidar_center_y_target))
        follower_args.heading_conflict_error_y = float(args.lidar_heading_conflict_error_y)
        follower_args.heading_conflict_scale = float(args.lidar_heading_conflict_scale)
        follower_args.row_width = float(args.lidar_row_width)
        follower_args.vehicle_width = float(args.lidar_vehicle_width)
        follower_args.min_row_width = float(args.lidar_min_row_width)
        follower_args.max_row_width = float(args.lidar_max_row_width)
        follower_args.lookahead_x = float(args.lidar_lookahead_x)
        follower_args.forward_lookahead_x = float(args.lidar_forward_lookahead_x)
        follower_args.reverse_lookahead_x = float(args.lidar_reverse_lookahead_x)
        follower_args.forward_min = float(args.lidar_forward_min)
        follower_args.forward_max = float(args.lidar_forward_max)
        follower_args.lateral_limit = float(args.lidar_lateral_limit)
        follower_args.range_min = float(args.lidar_range_min)
        follower_args.range_max = float(args.lidar_range_max)
        follower_args.bin_size = float(args.lidar_bin_size)
        follower_args.min_points = int(args.lidar_min_points)
        follower_args.min_bins = int(args.lidar_min_bins)
        follower_args.min_line_bins = int(args.lidar_min_line_bins)
        follower_args.min_side_points_per_bin = int(args.lidar_min_side_points_per_bin)
        follower_args.boundary_width_tolerance_m = float(args.lidar_boundary_width_tolerance_m)
        follower_args.center_deadband = float(args.lidar_center_deadband)
        follower_args.left_percentile = float(args.lidar_left_percentile)
        follower_args.right_percentile = float(args.lidar_right_percentile)
        follower_args.sensor_yaw_deg = float(args.lidar_sensor_yaw_deg)
        follower_args.lidar_yaw_correction_deg = float(
            calibration.get("lidar_yaw_correction_deg", float(args.lidar_yaw_correction_deg))
        )
        follower_args.lidar_x_offset_m = float(
            calibration.get("lidar_x_offset_m", float(args.lidar_x_offset_m))
        )
        follower_args.lidar_y_offset_m = float(
            calibration.get("lidar_y_offset_m", float(args.lidar_y_offset_m))
        )
        follower_args.control_deadband_y = float(args.lidar_control_deadband_y)
        follower_args.slow_error_y = float(args.lidar_slow_error_y)
        follower_args.stop_error_y = float(args.lidar_stop_error_y)
        follower_args.slow_heading_rad = float(args.lidar_slow_heading_rad)
        follower_args.safety_margin = float(args.lidar_safety_margin)
        follower_args.center_jump_reject = float(args.lidar_center_jump_reject)
        follower_args.one_side_center_jump_reject = float(args.lidar_one_side_center_jump_reject)
        follower_args.center_y_reject_abs = float(args.lidar_center_y_reject_abs)
        follower_args.raw_center_out_of_range = float(args.lidar_raw_center_out_of_range)
        follower_args.one_side_raw_center_out_of_range = float(args.lidar_one_side_raw_center_out_of_range)
        follower_args.history_window_s = float(args.lidar_history_window_s)
        follower_args.min_center_history = int(args.lidar_min_center_history)
        follower_args.center_y_alpha = float(args.lidar_center_y_alpha)
        follower_args.center_y_max_jump = float(args.lidar_center_y_max_jump)
        follower_args.one_side_stop_error_y = float(args.lidar_one_side_stop_error_y)
        follower_args.one_side_safety_stop_band = float(args.lidar_one_side_safety_stop_band)
        follower_args.control_period = float(args.lidar_control_period)
        follower_args.status_period = float(args.lidar_status_period)
        follower_args.scan_timeout = float(args.lidar_scan_timeout)
        follower_args.forward_lost_hold_sec = float(args.lidar_forward_lost_hold_sec)
        follower_args.forward_lost_stop_sec = float(args.lidar_forward_lost_stop_sec)
        follower_args.forward_lost_hold_wz_scale = float(args.lidar_forward_lost_hold_wz_scale)
        follower_args.forward_lost_hold_max_wz_deg = float(args.lidar_forward_lost_hold_max_wz_deg)
        follower_args.reverse_min_speed = float(args.lidar_reverse_min_speed)
        follower_args.reverse_one_side_speed = float(args.lidar_reverse_one_side_speed)
        follower_args.reverse_both_sides_speed = float(args.lidar_reverse_both_sides_speed)
        follower_args.reverse_min_wz_deg = float(args.lidar_reverse_min_wz_deg)
        follower_args.reverse_max_wz_deg = float(args.lidar_reverse_max_wz_deg)
        follower_args.reverse_one_side_max_wz_deg = float(args.lidar_reverse_one_side_max_wz_deg)
        follower_args.reverse_wz_enable_error_y = float(args.lidar_reverse_wz_enable_error_y)
        follower_args.reverse_wz_enable_heading_deg = float(args.lidar_reverse_wz_enable_heading_deg)
        follower_args.reverse_min_wz_error_y = float(args.lidar_reverse_min_wz_error_y)
        follower_args.reverse_sign_flip_guard_error_y = float(args.lidar_reverse_sign_flip_guard_error_y)
        follower_args.reverse_sign_flip_guard_last_wz_deg = float(args.lidar_reverse_sign_flip_guard_last_wz_deg)
        follower_args.reverse_sign_hold_error_y = float(args.lidar_reverse_sign_hold_error_y)
        follower_args.reverse_both_sides_k_lat = float(args.lidar_reverse_both_sides_k_lat)
        follower_args.reverse_both_sides_k_heading = float(args.lidar_reverse_both_sides_k_heading)
        follower_args.reverse_one_side_k_lat = float(args.lidar_reverse_one_side_k_lat)
        follower_args.k_reverse_lat = float(args.lidar_k_reverse_lat)
        follower_args.k_reverse_heading = float(args.lidar_k_reverse_heading)
        follower_args.reverse_steer_sign = float(args.lidar_reverse_steer_sign)
        follower_args.reverse_heading_conflict_error_y = float(args.lidar_reverse_heading_conflict_error_y)
        follower_args.reverse_heading_max_ratio = float(args.lidar_reverse_heading_max_ratio)
        follower_args.reverse_recenter_error_y = float(args.lidar_reverse_recenter_error_y)
        follower_args.reverse_recenter_heading_deg = float(args.lidar_reverse_recenter_heading_deg)
        follower_args.reverse_recenter_scale = float(args.lidar_reverse_recenter_scale)
        follower_args.reverse_error_stop = float(args.lidar_reverse_error_stop)
        follower_args.reverse_wz_smoothing_alpha = float(args.lidar_reverse_wz_smoothing_alpha)
        follower_args.reverse_lost_hold_sec = float(args.lidar_reverse_lost_hold_sec)
        follower_args.reverse_lost_stop_sec = float(args.lidar_reverse_lost_stop_sec)
        follower_args.reverse_lost_hold_max_wz_deg = float(args.lidar_reverse_lost_hold_max_wz_deg)
        follower_args.reverse_lost_soft_max_wz_deg = float(args.lidar_reverse_lost_soft_max_wz_deg)
        follower_args.reverse_start_lock_frames = int(args.lidar_reverse_start_lock_frames)
        follower_args.reverse_start_ramp_frames = int(args.lidar_reverse_start_ramp_frames)
        follower_args.reverse_start_max_wz_deg = float(args.lidar_reverse_start_max_wz_deg)
        follower_args.max_wz_delta_deg_per_cycle = float(args.lidar_max_wz_delta_deg_per_cycle)
        follower_args.enable_4t4d_steering_assist = bool(args.lidar_enable_4t4d_steering_assist)
        follower_args.steering_assist_wheelbase_m = float(args.lidar_steering_assist_wheelbase_m)
        follower_args.steering_assist_gain = float(args.lidar_steering_assist_gain)
        follower_args.steering_assist_max_angle_deg = float(args.lidar_steering_assist_max_angle_deg)
        follower_args.steering_assist_min_speed_mps = float(args.lidar_steering_assist_min_speed_mps)
        follower_args.steering_assist_speed_mps = float(args.lidar_steering_assist_speed_mps)
        follower_args.low_beam = bool(args.line_low_beam)
        follower_args.reverse = False
        follower_args.gear = "4t4d"
        return follower_args

    def _build_node(self, args: argparse.Namespace, config: dict[str, Any]):
        controller = self
        module = self._module
        follower_args = self._build_args(args, config)

        class _InProcessFollower(module.PlantRowFollower):
            def __init__(self, args_ns: argparse.Namespace) -> None:
                super().__init__(args_ns)
                self.drive_enable = False
                self._direct_drive_until = 0.0
                self._direct_command_active = False

            def _send_drive(self, gear: str, vx: float, wz: float, force_brake: bool = False) -> None:
                if time.monotonic() < float(self._direct_drive_until) and not bool(self._direct_command_active):
                    return
                super()._send_drive(gear, vx, wz, force_brake=force_brake)
                with controller._cmd_lock:
                    controller._cmd = TwistCommand(
                        vx=float(vx),
                        vy=0.0,
                        wz=float(wz),
                        updated_at=time.monotonic(),
                        fresh=bool(self.drive_enable),
                    )

            def send_direct_drive(self, gear: str, vx: float, wz_rad: float, force_brake: bool = False) -> None:
                previous_enable = bool(self.drive_enable)
                try:
                    self.drive_enable = True
                    self._direct_command_active = True
                    self._direct_drive_until = time.monotonic() + 0.80
                    self._send_drive(gear, float(vx), float(wz_rad), force_brake=force_brake)
                finally:
                    self._direct_command_active = False
                    self.drive_enable = previous_enable

            def send_direct_body_drive(
                self,
                gear: str,
                vx: float,
                vy: float,
                wz_rad: float,
                force_brake: bool = False,
            ) -> None:
                previous_enable = bool(self.drive_enable)
                try:
                    self.drive_enable = True
                    self._direct_command_active = True
                    self._direct_drive_until = time.monotonic() + 0.80
                    normalized_gear = self._normalized_gear() if gear in {"6", "8"} else gear
                    io_cmd = module.IOCommand(
                        light_mode="free" if self.args.low_beam else "auto",
                        low_beam=bool(self.args.low_beam),
                        brake=bool(force_brake) and not any(abs(v) > 1e-6 for v in (vx, vy, wz_rad)),
                    )
                    steering_cmd = None if normalized_gear == "4t4d" else module.SteeringCommand(
                        gear=normalized_gear,
                        speed=0.0,
                        angle=self.last_steering_angle,
                    )
                    self.sender.update(
                        module.BodyCommand(
                            gear=normalized_gear,
                            vx=float(vx),
                            vy=float(vy),
                            wz=float(wz_rad),
                        ),
                        steering_cmd,
                        io_cmd,
                    )
                    if any(abs(v) > 1e-6 for v in (vx, vy, wz_rad)):
                        self.sender.request_unlock()
                    with controller._cmd_lock:
                        controller._cmd = TwistCommand(
                            vx=float(vx),
                            vy=float(vy),
                            wz=float(wz_rad),
                            updated_at=time.monotonic(),
                            fresh=True,
                        )
                finally:
                    self._direct_command_active = False
                    self.drive_enable = previous_enable

            def hold_direct_control(self, hold_sec: float = 0.35) -> None:
                self.drive_enable = False
                self._direct_drive_until = max(float(self._direct_drive_until), time.monotonic() + max(0.05, float(hold_sec)))

            def clear_motion_history(self) -> None:
                super().clear_motion_history()

            def _send_stop(self) -> None:
                if time.monotonic() < float(self._direct_drive_until):
                    return
                super()._send_stop()

            def _publish_status(self) -> None:
                super()._publish_status()
                estimate = self.last_estimate
                center_error_m = float(estimate.center_y)
                row_width_m = float(estimate.row_width)
                half_row_width_m = 0.5 * row_width_m
                left_boundary_dist_m = center_error_m + half_row_width_m
                right_boundary_dist_m = half_row_width_m - center_error_m
                vehicle_half_width_m = 0.5 * float(self.args.vehicle_width)
                left_clearance_m = left_boundary_dist_m - vehicle_half_width_m
                right_clearance_m = right_boundary_dist_m - vehicle_half_width_m
                state = "IDLE"
                if self.drive_enable:
                    state = "TRACK" if estimate.found else "SEARCH"
                payload = {
                    "state": state,
                    "found": bool(estimate.found),
                    "obstacle_blocked": False,
                    "lost_frames": 0 if estimate.found else 1,
                    "reverse": bool(self.args.reverse),
                    "drive_enable": bool(self.drive_enable),
                    "mode": estimate.mode,
                    "raw_center_y_m": round(float(estimate.raw_center_y), 4),
                    "filtered_center_y_m": round(float(estimate.center_y), 4),
                    "center_y_m": round(float(estimate.center_y), 4),
                    "center_error_m": round(center_error_m, 4),
                    "heading_deg": round(math.degrees(float(estimate.heading_rad)), 3),
                    "row_width_m": round(row_width_m, 4),
                    "left_boundary_dist_m": round(left_boundary_dist_m, 4),
                    "right_boundary_dist_m": round(right_boundary_dist_m, 4),
                    "left_clearance_m": round(left_clearance_m, 4),
                    "right_clearance_m": round(right_clearance_m, 4),
                    "left_bins": int(estimate.left_bins),
                    "right_bins": int(estimate.right_bins),
                    "candidate_bins": int(estimate.candidate_bins),
                    "reject_reason": estimate.reject_reason,
                }
                with controller._status_lock:
                    controller._status = LineStatus(
                        state=state,
                        found=bool(estimate.found),
                        obstacle_blocked=False,
                        lost_frames=0 if estimate.found else 1,
                        reverse=bool(self.args.reverse),
                        drive_enable=bool(self.drive_enable),
                        fresh=True,
                        updated_at=time.monotonic(),
                        payload=payload,
                    )

            def feedback_snapshot(self) -> dict[str, Any]:
                try:
                    self.can_reader.poll(timeout=0.0, limit=20)
                except Exception:
                    pass
                snapshot = {}
                try:
                    snapshot = self.can_reader.snapshot()
                except Exception:
                    snapshot = {}
                try:
                    runtime = self.sender.feedback_snapshot().get("_runtime", {})
                except Exception:
                    runtime = {}
                if isinstance(snapshot, dict):
                    snapshot = dict(snapshot)
                    snapshot["_runtime"] = dict(runtime)
                return snapshot

        return _InProcessFollower(follower_args)

    def _spin(self) -> None:
        try:
            self._executor.spin()
        except Exception:
            pass

    def publish_mode(
        self,
        *,
        enable: bool,
        reverse: bool,
        cruise_vx: float,
        max_wz_deg: float | None = None,
        gear: str,
        low_beam: bool,
        target_center_offset_px: float = 0.0,
        vehicle_direction_angle_deg: float = 0.0,
    ) -> None:
        del gear, target_center_offset_px, vehicle_direction_angle_deg
        mode_changed = bool(enable) and (
            not bool(self._node.drive_enable) or bool(reverse) != bool(self._node.args.reverse)
        )
        if mode_changed:
            self._node.clear_motion_history()
        self._node.drive_enable = bool(enable)
        self._node.args.reverse = bool(reverse)
        self._node.args.low_beam = bool(low_beam)
        self._node.args.speed = abs(float(cruise_vx))
        if max_wz_deg is not None:
            self._node.args.max_wz_deg = abs(float(max_wz_deg))

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

    def feedback_snapshot(self) -> dict[str, Any]:
        try:
            return self._node.feedback_snapshot()
        except Exception:
            return {}

    def send_direct_drive(self, gear: str, vx: float, wz_rad: float, force_brake: bool = False) -> None:
        self._node.send_direct_drive(gear, float(vx), float(wz_rad), force_brake=force_brake)

    def send_direct_body_drive(
        self,
        gear: str,
        vx: float,
        vy: float,
        wz_rad: float,
        force_brake: bool = False,
    ) -> None:
        self._node.send_direct_body_drive(gear, float(vx), float(vy), float(wz_rad), force_brake=force_brake)

    def hold_direct_control(self, hold_sec: float = 0.35) -> None:
        self._node.hold_direct_control(float(hold_sec))

    def clear_motion_history(self) -> None:
        self._node.clear_motion_history()

    def close(self) -> None:
        try:
            self._node.stop()
        except Exception:
            pass
        try:
            self._executor.shutdown()
        except Exception:
            pass
        try:
            self._executor.remove_node(self._node)
        except Exception:
            pass
        try:
            self._node.destroy_node()
        except Exception:
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


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
    local_controller: DirectLocalLidarController,
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
        local_controller.publish_mode(
            enable=True,
            reverse=False,
            cruise_vx=cruise_vx,
            gear="4t4d",
            low_beam=args.line_low_beam,
            target_center_offset_px=offset_px,
            vehicle_direction_angle_deg=direction_angle_deg,
        )
        status = local_controller.status_snapshot()
        cmd = local_controller.cmd_snapshot()
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


def _row_entry_lidar_reliable(
    status: LineStatus,
    *,
    min_clearance_m: float,
    max_heading_deg: float,
) -> tuple[bool, str]:
    if not status.fresh:
        return False, "stale_status"
    if not status.found:
        return False, "centerline_not_found"
    payload = status.payload if isinstance(status.payload, dict) else {}
    if str(payload.get("mode") or "") != "both_sides":
        return False, "both_boundaries_required"
    heading_deg = abs(_safe_float(payload.get("heading_deg"), float("inf")))
    if heading_deg > max(0.0, float(max_heading_deg)):
        return False, "heading_out_of_range"
    left_clearance_m = _safe_float(payload.get("left_clearance_m"), -float("inf"))
    right_clearance_m = _safe_float(payload.get("right_clearance_m"), -float("inf"))
    if min(left_clearance_m, right_clearance_m) < max(0.0, float(min_clearance_m)):
        return False, "insufficient_clearance"
    return True, "ok"


def _forward_row_segments(points: list[Any], motions: list[dict[str, Any]]) -> list[ForwardRowSegment]:
    segments: list[ForwardRowSegment] = []
    idx = 0
    while idx < len(points) - 1:
        motion = motions[min(idx, len(motions) - 1)]
        raw_gear = str(motion.get("gear") or "")
        gear = core._resolve_replay_gear(motion, None)
        vx = _safe_float(motion.get("vx"), 0.0)
        if raw_gear == "7" or gear != "4t4d" or vx < -0.03:
            idx += 1
            continue
        start = idx
        end = idx
        while end + 1 < len(points):
            next_motion = motions[min(end + 1, len(motions) - 1)]
            next_raw_gear = str(next_motion.get("gear") or "")
            next_gear = core._resolve_replay_gear(next_motion, None)
            next_vx = _safe_float(next_motion.get("vx"), 0.0)
            # Ignore tiny speed drops or pauses inside the same row.
            # Only break the row when the recorded mode leaves 4t4d or
            # clearly turns into a reverse segment.
            if next_raw_gear == "7" or next_gear != "4t4d" or next_vx < -0.03:
                break
            end += 1
        start_point = points[start]
        end_point = points[end]
        dx = float(end_point.x - start_point.x)
        dy = float(end_point.y - start_point.y)
        length_m = math.hypot(dx, dy)
        if length_m >= 0.20:
            segments.append(
                ForwardRowSegment(
                    start_index=start,
                    end_index=end,
                    start_point=start_point,
                    end_point=end_point,
                    unit_x=dx / length_m,
                    unit_y=dy / length_m,
                    length_m=length_m,
                )
            )
        idx = end + 1
    return segments


def _segment_for_index(segments: list[ForwardRowSegment], index: int) -> ForwardRowSegment | None:
    for segment in segments:
        if segment.start_index <= index <= segment.end_index:
            return segment
    return None


def _next_forward_segment(
    segments: list[ForwardRowSegment],
    index: int,
) -> ForwardRowSegment | None:
    for segment in segments:
        if segment.end_index < index:
            continue
        if segment.start_index >= index or segment.start_index <= index <= segment.end_index:
            return segment
    return None


def _segment_lateral_error(segment: ForwardRowSegment, pose: Any) -> float:
    rel_x = float(pose.x - segment.start_point.x)
    rel_y = float(pose.y - segment.start_point.y)
    return -rel_x * segment.unit_y + rel_y * segment.unit_x


def _choose_start_index_from_offset(
    points: list[Any],
    motions: list[dict[str, Any]],
    start_index: int,
) -> int:
    if not points:
        return 0
    start = max(0, min(int(start_index), len(points) - 1))
    rel_index = core._choose_start_index(points[start:], motions[start:])
    return max(start, min(len(points) - 1, start + int(rel_index)))


def _next_forward_segment_start_index(
    segments: list[ForwardRowSegment],
    motions: list[dict[str, Any]],
    fallback_index: int,
) -> int:
    segment = _next_forward_segment(segments, fallback_index)
    if segment is not None:
        start = int(segment.start_index)
        end = int(segment.end_index)
        for idx in range(start, end + 1):
            motion = motions[min(idx, len(motions) - 1)]
            gear = core._resolve_replay_gear(motion, None)
            vx = _safe_float(motion.get("vx"), 0.0)
            if gear == "4t4d" and vx > 0.03:
                return idx
        return start
    return int(fallback_index)


@dataclass
class RowEndReverseState:
    active: bool = False
    stop_until: float = 0.0
    triggered_at_index: int = -1
    hard_stop_sent: bool = False
    reverse_start_index: int = -1
    reverse_end_index: int = -1
    pending_next_index: int = -1
    row_change_sync_until: float = 0.0
    row_change_sync_sent: bool = False
    post_row_change_lock_until: float = 0.0
    forward_mode_sync_until: float = 0.0
    forward_mode_sync_logged: bool = False
    force_global_only: bool = False
    start_index_floor: int = 0
    last_global_lateral_err: float = 0.0
    last_global_heading_err_deg: float = 0.0
    runaway_same_side_count: int = 0
    reverse_global_progress_m: float = 0.0


@dataclass
class RowEntryAssistState:
    segment_start_index: int = -1
    active: bool = False
    handed_off: bool = False
    force_global_only: bool = False
    start_along_m: float = 0.0
    start_along_valid: bool = False
    fresh_start_reset_done: bool = False
    lidar_pending: bool = False
    lidar_tracking: bool = False
    settle_until: float = 0.0
    stable_frames: int = 0
    last_status_at: float = 0.0
    last_log_at: float = 0.0
    lidar_start_along_m: float = 0.0
    lidar_start_along_valid: bool = False


@dataclass
class ForwardRowSegment:
    start_index: int
    end_index: int
    start_point: Any
    end_point: Any
    unit_x: float
    unit_y: float
    length_m: float


def _segment_target_index(
    points: list[Any],
    segment: ForwardRowSegment,
    desired_along_m: float,
) -> int:
    desired = max(0.0, min(float(desired_along_m), float(segment.length_m)))
    best_index = segment.end_index
    for idx in range(segment.start_index, segment.end_index + 1):
        point = points[idx]
        rel_x = float(point.x - segment.start_point.x)
        rel_y = float(point.y - segment.start_point.y)
        along = rel_x * segment.unit_x + rel_y * segment.unit_y
        if along >= desired:
            best_index = idx
            break
    return max(segment.start_index, min(segment.end_index, best_index))


def _nearest_forward_target_index(
    points: list[Any],
    segment: ForwardRowSegment,
    pose: Any,
    *,
    min_ahead_m: float = 0.8,
) -> int:
    rel_x = float(pose.x - segment.start_point.x)
    rel_y = float(pose.y - segment.start_point.y)
    along_pose = rel_x * segment.unit_x + rel_y * segment.unit_y
    desired_along = max(0.0, min(float(segment.length_m), along_pose + max(0.1, float(min_ahead_m))))
    return _segment_target_index(points, segment, desired_along)


def _nearest_index_in_range(
    points: list[Any],
    pose: Any,
    start_index: int,
    end_index: int,
) -> tuple[int, float]:
    start = max(0, min(int(start_index), len(points) - 1))
    end = max(start, min(int(end_index), len(points) - 1))
    best_index = start
    best_dist = float("inf")
    for idx in range(start, end + 1):
        dist = pose.distance_to(points[idx])
        if dist < best_dist:
            best_dist = dist
            best_index = idx
    return best_index, best_dist


def _path_progress_in_range(
    points: list[Any],
    pose: Any,
    start_index: int,
    end_index: int,
) -> tuple[float, int, float]:
    start = max(0, min(int(start_index), len(points) - 1))
    end = max(start, min(int(end_index), len(points) - 1))
    nearest_index, nearest_dist = _nearest_index_in_range(points, pose, start, end)
    progress_m = 0.0
    if nearest_index > start:
        for idx in range(start, nearest_index):
            progress_m += points[idx].distance_to(points[idx + 1])
    return progress_m, nearest_index, nearest_dist


def _sync_start_index_from_pose(
    points: list[Any],
    pose: Any,
    start_index: int,
    *,
    look_back: int = 3,
    look_ahead: int = 24,
    max_snap_dist: float = 1.8,
) -> int:
    if not points:
        return 0
    nearest_index, nearest_dist = _nearest_index_in_range(
        points,
        pose,
        max(0, int(start_index) - max(0, int(look_back))),
        min(len(points) - 1, int(start_index) + max(1, int(look_ahead))),
    )
    if nearest_dist <= max(0.10, float(max_snap_dist)):
        return nearest_index
    return start_index


def _linear_blend_amount(value: float, start: float, end: float) -> float:
    if end <= start:
        return 1.0 if value <= end else 0.0
    if value <= start:
        return 1.0
    if value >= end:
        return 0.0
    return (end - value) / (end - start)


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


def _append_hybrid_log(log_path: Path | None, message: str) -> None:
    if log_path is None:
        return
    try:
        timestamp = time.strftime("%H:%M:%S")
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"{timestamp} {message}\n")
    except Exception:
        pass


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

    points = [core._sample_pose(sample) for sample in samples]
    sample_period = float(mission.get("sample_period", 0.2) or 0.2)
    motions, _used_fallback = core._repair_missing_motion(samples, sample_period)
    start_index = core._choose_start_index(points, motions)
    forward_segments = _forward_row_segments(points, motions)
    controller = core.FWMiniController(args.interface, args.channel, args.bitrate)
    send_state = core.MotionSendState.create()
    send_state.unlock_request_active = True
    for _ in range(2):
        send_state.queue_unlock_sequence()

    session: LocalizationSession | None = None
    tracker: core.TFPoseTracker | None = None
    lidar_driver_proc: LidarDriverProcess | None = None
    direct_local_controller: DirectLocalLidarController | None = None
    local_controller: DirectLocalLidarController | None = None
    lidar_driver_log = LOG_DIR / f"lidar_driver_{time.strftime('%Y%m%d_%H%M%S')}.log"
    hybrid_run_log = LOG_DIR / f"hybrid_autorun_{time.strftime('%Y%m%d_%H%M%S')}.log"
    local_lidar_config = _load_local_lidar_gui_config()

    # Align autorunlida with the standalone lidarun calibration file so the
    # lidar flip, yaw correction, and offsets match the user's working setup.
    calib_path = PROJECT_ROOT / "config" / "lidar_calibration.json"
    if calib_path.exists():
        try:
            calib = json.loads(calib_path.read_text(encoding="utf-8"))
            args.lidar_yaw_correction_deg = float(calib.get("lidar_yaw_correction_deg", args.lidar_yaw_correction_deg))
            args.lidar_x_offset_m = float(calib.get("lidar_x_offset_m", args.lidar_x_offset_m))
            args.lidar_y_offset_m = float(calib.get("lidar_y_offset_m", args.lidar_y_offset_m))
            core.log(
                "Loaded lidarun calibration: "
                f"yaw_corr={args.lidar_yaw_correction_deg:.3f} "
                f"x_offset={args.lidar_x_offset_m:.3f} "
                f"y_offset={args.lidar_y_offset_m:.3f}"
            )
        except Exception as exc:
            core.log(f"Warning: failed to load lidarun calibration file: {exc}")

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
        _pose = core.project_pose_to_ground(pose_raw, projection)

        lidar_driver_proc = LidarDriverProcess(lidar_driver_log)
        lidar_driver_proc.start()
        core.log(f"lidar driver subprocess started. Raw log: {lidar_driver_log}")
        time.sleep(1.0)

        direct_local_controller = DirectLocalLidarController(args, local_lidar_config)
        local_controller = direct_local_controller
        if not _wait_for_local_lidar_ready(local_controller, args, local_lidar_config=local_lidar_config):
            raise RuntimeError("Pure local lidar control was not ready before autorun start.")

        core.log("Hybrid autorun now runs pure lidarun local control. No global control logic is applied.")
        _append_hybrid_log(hybrid_run_log, f"Hybrid autorun log started: {hybrid_run_log}")
        _append_hybrid_log(hybrid_run_log, f"Mission: {args.mission}")
        _append_hybrid_log(hybrid_run_log, f"Map DB: {args.db}")
        last_cmd_log = 0.0
        last_feedback_log = 0.0
        last_crab_align_log = 0.0
        last_reverse_state: bool | None = None
        current_gear = "4t4d"
        current_cmd_gear = "4t4d"
        row_end_reverse = RowEndReverseState()
        row_entry_assist = RowEntryAssistState()
        reverse_stop_pause_s = 0.8

        def _clear_drive_boundary(reason: str, next_gear: str = "4t4d") -> None:
            """Hard-clear command state when switching reverse/crab/row-entry phases."""
            assert local_controller is not None
            core._hold_current_gear_stop(controller, send_state, current_gear)
            local_controller.send_direct_body_drive("crab", 0.0, 0.0, 0.0, force_brake=True)
            local_controller.send_direct_drive("4t4d", 0.0, 0.0, force_brake=True)
            local_controller.clear_motion_history()
            send_state.reset_motion()
            send_state.crab_locked_until = -1
            send_state.crab_target_index = -1
            send_state.crab_best_dist = float("inf")
            send_state.crab_diverge_count = 0
            send_state.last_sent_gear = next_gear
            clear_msg = f"Hybrid drive boundary cleared: {reason}, next_gear={next_gear}."
            core.log(clear_msg)
            _append_hybrid_log(hybrid_run_log, clear_msg)

        while not core.STOP_REQUESTED and start_index < len(points):
            assert tracker is not None
            pose_raw = tracker.lookup()
            if pose_raw is None:
                time.sleep(0.05)
                continue
            pose = core.project_pose_to_ground(pose_raw, projection)
            if not row_end_reverse.active:
                start_index = _sync_start_index_from_pose(
                    points,
                    pose,
                    start_index,
                    look_back=4,
                    look_ahead=28,
                    max_snap_dist=2.2,
                )
            start_index = max(int(start_index), int(row_end_reverse.start_index_floor))
            snapshot = local_controller.feedback_snapshot()
            io_state = snapshot.get("io", {})
            if bool(io_state.get("remote_control", False)):
                local_controller.publish_mode(
                    enable=False,
                    reverse=False,
                    cruise_vx=abs(float(args.line_cruise_vx)),
                    gear="4t4d",
                    low_beam=args.line_low_beam,
                    target_center_offset_px=0.0,
                    vehicle_direction_angle_deg=0.0,
                )
                time.sleep(0.10)
                continue

            nearest_index = core._find_tracking_index(points, start_index, pose, window=14)
            start_index = max(start_index, nearest_index)
            start_index = max(int(start_index), int(row_end_reverse.start_index_floor))
            motion_here = motions[min(start_index, len(motions) - 1)]
            mission_wants_reverse = (
                core._resolve_replay_gear(motion_here, current_gear) == "reverse"
                or _safe_float(motion_here.get("vx"), 0.0) < -0.03
            )
            reversing_here = mission_wants_reverse or row_end_reverse.active
            target_index = min(len(points) - 1, max(start_index, start_index + 1))
            dist = pose.distance_to(points[target_index])
            current_segment = _segment_for_index(forward_segments, start_index)
            if (
                current_segment is None
                and not reversing_here
                and row_entry_assist.force_global_only
                and send_state.crab_locked_until < start_index
            ):
                start_index = _choose_start_index_from_offset(points, motions, start_index)
                current_segment = _segment_for_index(forward_segments, start_index)
            in_row_end_zone = False
            forward_global_window = False
            row_entry_global_window = False
            row_change_global_window = False
            row_entry_handoff_ready = False
            if current_segment is not None:
                rel_x = float(pose.x - current_segment.start_point.x)
                rel_y = float(pose.y - current_segment.start_point.y)
                along = rel_x * current_segment.unit_x + rel_y * current_segment.unit_y
                lateral = -rel_x * current_segment.unit_y + rel_y * current_segment.unit_x
                remaining_along = current_segment.length_m - along
                if row_entry_assist.segment_start_index != current_segment.start_index:
                    row_entry_assist.segment_start_index = current_segment.start_index
                    lidar_entry_staged = row_entry_assist.lidar_pending or row_entry_assist.lidar_tracking
                    row_entry_assist.active = not reversing_here and not lidar_entry_staged
                    row_entry_assist.handed_off = reversing_here or lidar_entry_staged
                    row_entry_assist.fresh_start_reset_done = False
                    if not reversing_here and row_entry_assist.force_global_only and not lidar_entry_staged:
                        row_entry_assist.active = True
                        row_entry_assist.handed_off = False
                        row_entry_assist.start_along_valid = False
                        row_entry_assist.fresh_start_reset_done = False
                in_row_end_zone = (
                    abs(lateral) <= 0.80
                    and remaining_along <= 0.35
                    and remaining_along >= -0.20
                )
                forward_global_window = (
                    not reversing_here
                    and abs(lateral) <= 0.80
                    and remaining_along <= 2.5
                    and remaining_along >= -0.30
                )
                row_entry_global_window = (
                    row_entry_assist.active
                    and not row_entry_assist.handed_off
                    and not reversing_here
                    and not row_entry_assist.lidar_pending
                    and not row_entry_assist.lidar_tracking
                )
                if row_entry_global_window:
                    if not row_entry_assist.start_along_valid:
                        row_entry_assist.start_along_m = along
                        row_entry_assist.start_along_valid = True
                    row_entry_progress_m = max(0.0, along - row_entry_assist.start_along_m)
                    row_entry_global_window = row_entry_progress_m < 2.5
                else:
                    row_entry_progress_m = 0.0
                if time.monotonic() < row_end_reverse.post_row_change_lock_until:
                    forward_global_window = False
                    row_entry_global_window = False
                row_entry_handoff_ready = row_entry_assist.start_along_valid and row_entry_progress_m >= 2.5
                if row_entry_handoff_ready:
                    row_entry_assist.active = False
                    row_entry_assist.handed_off = True
                    row_entry_assist.force_global_only = False
                    row_entry_assist.start_along_valid = False
                    row_entry_assist.fresh_start_reset_done = False
                    row_entry_global_window = False
            else:
                row_entry_assist.active = False
                row_entry_assist.handed_off = False
                row_entry_assist.start_along_valid = False
                row_entry_assist.fresh_start_reset_done = False
                row_change_global_window = (
                    not reversing_here
                    and not row_entry_assist.lidar_pending
                    and not row_entry_assist.lidar_tracking
                )
            if (
                row_entry_assist.force_global_only
                and not reversing_here
                and not row_entry_assist.lidar_pending
                and not row_entry_assist.lidar_tracking
            ):
                row_entry_assist.active = True
                row_entry_assist.handed_off = False
            if (
                not row_end_reverse.active
                and not mission_wants_reverse
                and current_segment is not None
                and in_row_end_zone
            ):
                row_end_reverse.active = True
                row_end_reverse.stop_until = time.monotonic() + reverse_stop_pause_s
                row_end_reverse.triggered_at_index = start_index
                row_end_reverse.hard_stop_sent = False
                row_end_reverse.pending_next_index = -1
                row_end_reverse.force_global_only = True
                row_end_reverse.reverse_global_progress_m = 0.0
                reverse_start_index = current_segment.end_index + 1
                reverse_end_index = reverse_start_index
                while reverse_end_index + 1 < len(points):
                    next_motion = motions[min(reverse_end_index + 1, len(motions) - 1)]
                    next_gear = core._resolve_replay_gear(next_motion, current_gear)
                    next_vx = _safe_float(next_motion.get("vx"), 0.0)
                    if next_gear != "reverse" and next_vx >= -0.03:
                        break
                    reverse_end_index += 1
                row_end_reverse.reverse_start_index = reverse_start_index
                row_end_reverse.reverse_end_index = max(reverse_start_index, reverse_end_index)
                row_end_msg = (
                    f"Global row-end trigger: reached end zone of current forward row "
                    f"(segment={current_segment.start_index}-{current_segment.end_index}, dist={dist:.2f}, "
                    f"reverse_segment={row_end_reverse.reverse_start_index}-{row_end_reverse.reverse_end_index}). "
                    "Stopping first, then switching local lidarun to reverse."
                )
                core.log(row_end_msg)
                _append_hybrid_log(hybrid_run_log, row_end_msg)

            if row_end_reverse.active and not row_end_reverse.hard_stop_sent:
                core._hold_current_gear_stop(controller, send_state, current_gear)
                row_end_reverse.hard_stop_sent = True

            if last_reverse_state is None or last_reverse_state != reversing_here:
                reverse_msg = f"Pure local lidar mode switched to {'reverse' if reversing_here else 'forward'}."
                core.log(reverse_msg)
                _append_hybrid_log(hybrid_run_log, reverse_msg)
                last_reverse_state = reversing_here

            reverse_row_global_window = False
            if (
                row_end_reverse.active
                and row_end_reverse.force_global_only
                and row_end_reverse.pending_next_index < 0
                and time.monotonic() >= row_end_reverse.stop_until
                and row_end_reverse.reverse_start_index >= 0
                and row_end_reverse.reverse_start_index < len(points)
            ):
                reverse_progress_m, _reverse_progress_index, _reverse_progress_dist = _path_progress_in_range(
                    points,
                    pose,
                    row_end_reverse.reverse_start_index,
                    row_end_reverse.reverse_end_index,
                )
                row_end_reverse.reverse_global_progress_m = float(reverse_progress_m)
                reverse_row_global_window = reverse_progress_m < 1.0
                if not reverse_row_global_window:
                    row_end_reverse.force_global_only = False
            global_control_active = (
                row_entry_global_window
                or forward_global_window
                or row_change_global_window
                or reverse_row_global_window
            )
            post_row_change_locked = time.monotonic() < row_end_reverse.post_row_change_lock_until
            force_global_entry_only = bool(row_entry_assist.force_global_only and row_entry_global_window)

            cruise_vx, offset_px, direction_angle_deg = _local_lidar_drive_settings(
                local_lidar_config,
                reverse=reversing_here,
                args=args,
            )
            if row_end_reverse.active and time.monotonic() < row_end_reverse.stop_until:
                local_controller.publish_mode(
                    enable=False,
                    reverse=False,
                    cruise_vx=cruise_vx,
                    gear="4t4d",
                    low_beam=args.line_low_beam,
                    target_center_offset_px=offset_px,
                    vehicle_direction_angle_deg=direction_angle_deg,
                )
                time.sleep(0.05)
                continue

            if row_end_reverse.active and row_end_reverse.pending_next_index >= 0:
                next_index = min(len(points) - 1, row_end_reverse.pending_next_index)
                finish_msg = f"Reverse stop pause finished. Switching to next mission stage at index {next_index}."
                core.log(finish_msg)
                _append_hybrid_log(hybrid_run_log, finish_msg)
                _clear_drive_boundary("reverse_stop_pause_finished", next_gear="crab")
                row_end_reverse.active = False
                row_end_reverse.stop_until = 0.0
                row_end_reverse.triggered_at_index = -1
                row_end_reverse.hard_stop_sent = False
                row_end_reverse.reverse_start_index = -1
                row_end_reverse.reverse_end_index = -1
                row_end_reverse.pending_next_index = -1
                row_end_reverse.force_global_only = False
                row_end_reverse.last_global_lateral_err = 0.0
                row_end_reverse.last_global_heading_err_deg = 0.0
                row_end_reverse.runaway_same_side_count = 0
                row_end_reverse.reverse_global_progress_m = 0.0
                row_end_reverse.start_index_floor = max(int(row_end_reverse.start_index_floor), int(next_index))
                row_end_reverse.row_change_sync_until = time.monotonic() + 0.45
                row_end_reverse.row_change_sync_sent = False
                # Reset any previous entry phase before the crab segment. The
                # crab handoff will arm the stopped lidar-entry acquisition.
                row_end_reverse.post_row_change_lock_until = 0.0
                row_entry_assist.force_global_only = False
                row_entry_assist.active = False
                row_entry_assist.handed_off = False
                row_entry_assist.segment_start_index = -1
                row_entry_assist.start_along_m = 0.0
                row_entry_assist.start_along_valid = False
                row_entry_assist.fresh_start_reset_done = False
                row_entry_assist.lidar_pending = False
                row_entry_assist.lidar_tracking = False
                row_entry_assist.settle_until = 0.0
                row_entry_assist.stable_frames = 0
                row_entry_assist.last_status_at = 0.0
                row_entry_assist.last_log_at = 0.0
                row_entry_assist.lidar_start_along_m = 0.0
                row_entry_assist.lidar_start_along_valid = False
                start_index = _choose_start_index_from_offset(points, motions, next_index)
                last_reverse_state = None
                time.sleep(0.05)
                continue

            local_max_wz_deg = abs(float(args.lidar_max_wz_deg))
            if row_entry_assist.lidar_pending or row_entry_assist.lidar_tracking:
                now = time.monotonic()
                entry_status = local_controller.status_snapshot()
                if reversing_here or current_segment is None:
                    local_controller.publish_mode(
                        enable=False,
                        reverse=False,
                        cruise_vx=cruise_vx,
                        max_wz_deg=local_max_wz_deg,
                        gear="4t4d",
                        low_beam=args.line_low_beam,
                        target_center_offset_px=offset_px,
                        vehicle_direction_angle_deg=direction_angle_deg,
                    )
                    if now - row_entry_assist.last_log_at >= 1.0:
                        wait_msg = "Lidar row entry waiting: next forward row segment is not available."
                        core.log(wait_msg)
                        _append_hybrid_log(hybrid_run_log, wait_msg)
                        row_entry_assist.last_log_at = now
                    time.sleep(0.05)
                    continue

                if row_entry_assist.lidar_pending:
                    local_controller.publish_mode(
                        enable=False,
                        reverse=False,
                        cruise_vx=cruise_vx,
                        max_wz_deg=local_max_wz_deg,
                        gear="4t4d",
                        low_beam=args.line_low_beam,
                        target_center_offset_px=offset_px,
                        vehicle_direction_angle_deg=direction_angle_deg,
                    )
                    if now < row_entry_assist.settle_until:
                        time.sleep(0.05)
                        continue

                    reliability_reason = "waiting_for_new_status"
                    if entry_status.updated_at > row_entry_assist.last_status_at:
                        row_entry_assist.last_status_at = entry_status.updated_at
                        reliable, reliability_reason = _row_entry_lidar_reliable(
                            entry_status,
                            min_clearance_m=float(args.lidar_row_entry_min_clearance),
                            max_heading_deg=float(args.lidar_row_entry_max_heading_deg),
                        )
                        if reliable:
                            row_entry_assist.stable_frames += 1
                        else:
                            row_entry_assist.stable_frames = 0

                    required_frames = max(1, int(args.lidar_row_entry_stable_frames))
                    if row_entry_assist.stable_frames < required_frames:
                        if now - row_entry_assist.last_log_at >= 1.0:
                            payload = entry_status.payload if isinstance(entry_status.payload, dict) else {}
                            wait_msg = (
                                "Lidar row entry waiting for stable centerline: "
                                f"frames={row_entry_assist.stable_frames}/{required_frames} "
                                f"reason={reliability_reason} mode={payload.get('mode', '')} "
                                f"left_clearance={_safe_float(payload.get('left_clearance_m'), -1.0):.3f} "
                                f"right_clearance={_safe_float(payload.get('right_clearance_m'), -1.0):.3f}."
                            )
                            core.log(wait_msg)
                            _append_hybrid_log(hybrid_run_log, wait_msg)
                            row_entry_assist.last_log_at = now
                        time.sleep(0.05)
                        continue

                    row_entry_assist.lidar_pending = False
                    row_entry_assist.lidar_tracking = True
                    row_entry_assist.lidar_start_along_m = float(along)
                    row_entry_assist.lidar_start_along_valid = True
                    row_entry_assist.last_log_at = now
                    local_controller.clear_motion_history()
                    entry_msg = (
                        "Lidar row entry centerline locked. Starting constrained entry: "
                        f"speed={abs(float(args.lidar_row_entry_speed)):.2f}m/s "
                        f"distance={max(0.0, float(args.lidar_row_entry_distance)):.2f}m "
                        f"max_wz={abs(float(args.lidar_row_entry_max_wz_deg)):.1f}deg/s."
                    )
                    core.log(entry_msg)
                    _append_hybrid_log(hybrid_run_log, entry_msg)

                if row_entry_assist.lidar_tracking:
                    if entry_status.updated_at > row_entry_assist.last_status_at:
                        row_entry_assist.last_status_at = entry_status.updated_at
                        reliable, reliability_reason = _row_entry_lidar_reliable(
                            entry_status,
                            min_clearance_m=float(args.lidar_row_entry_min_clearance),
                            max_heading_deg=float(args.lidar_row_entry_max_heading_deg),
                        )
                        if not reliable:
                            local_controller.publish_mode(
                                enable=False,
                                reverse=False,
                                cruise_vx=cruise_vx,
                                max_wz_deg=local_max_wz_deg,
                                gear="4t4d",
                                low_beam=args.line_low_beam,
                                target_center_offset_px=offset_px,
                                vehicle_direction_angle_deg=direction_angle_deg,
                            )
                            local_controller.send_direct_drive("4t4d", 0.0, 0.0, force_brake=True)
                            local_controller.clear_motion_history()
                            row_entry_assist.lidar_pending = True
                            row_entry_assist.lidar_tracking = False
                            row_entry_assist.settle_until = now + 0.15
                            row_entry_assist.stable_frames = 0
                            row_entry_assist.lidar_start_along_valid = False
                            row_entry_assist.last_log_at = now
                            stop_msg = (
                                "Lidar row entry stopped and returned to centerline acquisition: "
                                f"reason={reliability_reason}."
                            )
                            core.log(stop_msg)
                            _append_hybrid_log(hybrid_run_log, stop_msg)
                            time.sleep(0.05)
                            continue

                    entry_progress_m = (
                        max(0.0, float(along) - row_entry_assist.lidar_start_along_m)
                        if row_entry_assist.lidar_start_along_valid
                        else 0.0
                    )
                    if entry_progress_m >= max(0.0, float(args.lidar_row_entry_distance)):
                        row_entry_assist.lidar_tracking = False
                        row_entry_assist.active = False
                        row_entry_assist.handed_off = True
                        row_entry_assist.lidar_start_along_valid = False
                        row_entry_assist.stable_frames = 0
                        finish_msg = (
                            "Lidar constrained row entry finished. "
                            f"progress={entry_progress_m:.2f}m; restoring normal lidar cruise."
                        )
                        core.log(finish_msg)
                        _append_hybrid_log(hybrid_run_log, finish_msg)
                    else:
                        cruise_vx = min(
                            abs(float(cruise_vx)),
                            abs(float(args.lidar_row_entry_speed)),
                        )
                        local_max_wz_deg = min(
                            local_max_wz_deg,
                            abs(float(args.lidar_row_entry_max_wz_deg)),
                        )

            reverse_global_only_active = bool(reversing_here and reverse_row_global_window)

            if global_control_active:
                motion_fb = snapshot.get("motion", {}) if isinstance(snapshot, dict) else {}
                steering_fb = snapshot.get("steering", {}) if isinstance(snapshot, dict) else {}
                row_end_reverse.forward_mode_sync_until = 0.0
                row_end_reverse.forward_mode_sync_logged = False
                local_controller.publish_mode(
                    enable=False,
                    reverse=reversing_here,
                    cruise_vx=cruise_vx,
                    gear="4t4d",
                    low_beam=args.line_low_beam,
                    target_center_offset_px=offset_px,
                    vehicle_direction_angle_deg=direction_angle_deg,
                )
                if row_entry_global_window and current_segment is not None:
                    if not row_entry_assist.fresh_start_reset_done:
                        _clear_drive_boundary("row_entry_global_start", next_gear="4t4d")
                        row_end_reverse.row_change_sync_until = 0.0
                        row_end_reverse.row_change_sync_sent = False
                        row_end_reverse.forward_mode_sync_until = 0.0
                        row_end_reverse.forward_mode_sync_logged = False
                        row_entry_assist.fresh_start_reset_done = True
                    if send_state.crab_locked_until >= 0:
                        send_state.crab_locked_until = -1
                        send_state.crab_target_index = -1
                        send_state.crab_best_dist = float("inf")
                        send_state.crab_diverge_count = 0
                    target_index = _nearest_forward_target_index(
                        points,
                        current_segment,
                        pose,
                        min_ahead_m=max(0.8, min(1.6, abs(float(args.line_cruise_vx)) * 8.0)),
                    )
                elif row_change_global_window:
                    motion_here_gear = core._resolve_replay_gear(motion_here, current_gear)
                    if send_state.crab_locked_until >= start_index and send_state.crab_target_index >= 0:
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
                            target_index = min(len(points) - 1, start_index + 2)
                    else:
                        if motion_here_gear == "crab":
                            crab_end = core._find_gear_segment_end(motions, start_index, "crab")
                            crab_active = core._find_first_active_crab_index(motions, start_index, crab_end)
                            crab_active_end = core._find_last_active_crab_index(motions, crab_active, crab_end)
                            send_state.crab_locked_until = crab_active_end
                            send_state.crab_target_index = crab_active
                            target_index = core._select_crab_progress_target(
                                start_index,
                                crab_active_end,
                                crab_active,
                            )
                        else:
                            upcoming_crab = core._find_future_gear_start(motions, start_index, "crab", limit=20)
                            if upcoming_crab is not None and pose.distance_to(points[upcoming_crab]) <= 0.45:
                                crab_end = core._find_gear_segment_end(motions, upcoming_crab, "crab")
                                crab_active = core._find_first_active_crab_index(motions, upcoming_crab, crab_end)
                                crab_active_end = core._find_last_active_crab_index(motions, crab_active, crab_end)
                                send_state.crab_locked_until = crab_active_end
                                send_state.crab_target_index = crab_active
                                target_index = core._select_crab_progress_target(
                                    start_index,
                                    crab_active_end,
                                    crab_active,
                                )
                            else:
                                target_index = min(len(points) - 1, start_index + 2)
                elif reverse_row_global_window:
                    reverse_nearest_index, _reverse_nearest_dist = _nearest_index_in_range(
                        points,
                        pose,
                        row_end_reverse.reverse_start_index,
                        row_end_reverse.reverse_end_index,
                    )
                    target_index = min(
                        row_end_reverse.reverse_end_index,
                        max(row_end_reverse.reverse_start_index, reverse_nearest_index + 2),
                    )
                target = points[target_index]
                motion = motions[min(target_index, len(motions) - 1)]
                tracking_heading = core._tracking_heading(points, motions, target_index, "crab" if row_change_global_window else "4t4d")
                forward_err, lateral_err = core._body_frame_error(pose, target)
                heading_err = core.normalize_angle(tracking_heading - pose.yaw)
                if (
                    global_control_active
                    and current_segment is not None
                    and not reversing_here
                    and not row_change_global_window
                ):
                    rel_x = float(pose.x - current_segment.start_point.x)
                    rel_y = float(pose.y - current_segment.start_point.y)
                    segment_lateral = -rel_x * current_segment.unit_y + rel_y * current_segment.unit_x
                    lateral_err = core._clamp(-segment_lateral, -0.05, 0.05)
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
                cmd_vy = 0.0
                cmd_gear = "4t4d"
                if reversing_here:
                    cmd_vx = -min(abs(float(args.line_cruise_vx)), 0.16)
                    if reverse_row_global_window:
                        # At row end, reverse the first 1m straight back only.
                        # Do not let global path tracking inject steering here.
                        cmd_wz = 0.0
                        row_end_reverse.last_global_lateral_err = 0.0
                        row_end_reverse.last_global_heading_err_deg = 0.0
                        row_end_reverse.runaway_same_side_count = 0
                    else:
                        # Non-row-end reverse path keeps the existing global reverse formula.
                        reverse_lateral_err = -lateral_err
                        cmd_wz = core._clamp(
                            -(heading_err_deg * 0.72 + reverse_lateral_err * 18.0),
                            -12.0,
                            12.0,
                        )
                        row_end_reverse.last_global_lateral_err = float(reverse_lateral_err)
                        row_end_reverse.last_global_heading_err_deg = float(heading_err_deg)
                        row_end_reverse.runaway_same_side_count = 0
                else:
                    row_end_reverse.last_global_lateral_err = 0.0
                    row_end_reverse.last_global_heading_err_deg = 0.0
                    row_end_reverse.runaway_same_side_count = 0
                    if row_change_global_window:
                        cmd_gear = "crab"
                        dist = pose.distance_to(target)
                        if dist + 0.03 < send_state.crab_best_dist:
                            send_state.crab_best_dist = dist
                            send_state.crab_diverge_count = 0
                        elif dist > send_state.crab_best_dist + 0.20:
                            send_state.crab_diverge_count += 1
                        else:
                            send_state.crab_diverge_count = max(0, send_state.crab_diverge_count - 1)
                        if send_state.crab_diverge_count >= 6:
                            fallback_msg = (
                                f"Hybrid row-change crab fallback: target distance kept increasing "
                                f"(best={send_state.crab_best_dist:.2f}, now={dist:.2f})."
                            )
                            core.log(fallback_msg)
                            _append_hybrid_log(hybrid_run_log, fallback_msg)
                            send_state.crab_locked_until = -1
                            send_state.crab_target_index = -1
                            send_state.crab_best_dist = float("inf")
                            send_state.crab_diverge_count = 0
                            time.sleep(0.05)
                            continue
                        ref_point = points[start_index]
                        axis_dx = target.x - ref_point.x
                        axis_dy = target.y - ref_point.y
                        if abs(axis_dy) >= abs(axis_dx):
                            axis_reached = core._axis_progress_reached(pose.y, target.y, axis_dy, 0.05)
                        else:
                            axis_reached = core._axis_progress_reached(pose.x, target.x, axis_dx, 0.05)
                        next_segment = _next_forward_segment(forward_segments, target_index + 1)
                        has_next_forward_segment = next_segment is not None
                        next_lateral_err = (
                            abs(_segment_lateral_error(next_segment, pose))
                            if has_next_forward_segment
                            else float("inf")
                        )
                        strict_crab_finish = (
                            axis_reached
                            and dist <= 0.12
                            and next_lateral_err <= 0.08
                        )
                        # After crab row-change, lateral alignment to the next
                        # forward row is the important part. Lidar entry will
                        # verify both boundaries before any forward movement.
                        aligned_crab_finish = has_next_forward_segment and next_lateral_err <= 0.06
                        crab_finish_ready = strict_crab_finish or aligned_crab_finish
                        if time.monotonic() - last_crab_align_log >= 1.0:
                            align_msg = (
                                f"Hybrid crab align: axis_reached={axis_reached} "
                                f"target_dist={dist:.2f} next_lateral_err={next_lateral_err:.2f} "
                                f"ready={crab_finish_ready} target_index={target_index}"
                            )
                            core.log(align_msg)
                            _append_hybrid_log(hybrid_run_log, align_msg)
                            last_crab_align_log = time.monotonic()
                        if crab_finish_ready:
                            next_forward_start = min(
                                len(points) - 1,
                                _next_forward_segment_start_index(forward_segments, motions, target_index + 1),
                            )
                            if next_forward_start <= start_index:
                                align_msg = (
                                    f"Hybrid crab handoff skipped at mission tail: "
                                    f"target_index={target_index} next_start_index={next_forward_start}"
                                )
                                core.log(align_msg)
                                _append_hybrid_log(hybrid_run_log, align_msg)
                                start_index = len(points)
                                time.sleep(0.03)
                                continue
                            align_msg = (
                                f"Hybrid crab handoff: target_index={target_index} "
                                f"next_start_index={next_forward_start}"
                            )
                            core.log(align_msg)
                            _append_hybrid_log(hybrid_run_log, align_msg)
                            _clear_drive_boundary("crab_to_lidar_row_entry", next_gear="4t4d")
                            start_index = next_forward_start
                            row_entry_assist.force_global_only = False
                            row_entry_assist.active = False
                            row_entry_assist.handed_off = True
                            row_entry_assist.segment_start_index = -1
                            row_entry_assist.start_along_m = 0.0
                            row_entry_assist.start_along_valid = False
                            row_entry_assist.fresh_start_reset_done = False
                            row_entry_assist.lidar_pending = True
                            row_entry_assist.lidar_tracking = False
                            row_entry_assist.settle_until = time.monotonic() + max(
                                0.0,
                                float(args.lidar_row_entry_settle_sec),
                            )
                            row_entry_assist.stable_frames = 0
                            row_entry_assist.last_status_at = 0.0
                            row_entry_assist.last_log_at = 0.0
                            row_entry_assist.lidar_start_along_m = 0.0
                            row_entry_assist.lidar_start_along_valid = False
                            if start_index > send_state.crab_locked_until:
                                send_state.crab_locked_until = -1
                                send_state.crab_target_index = -1
                                send_state.crab_best_dist = float("inf")
                                send_state.crab_diverge_count = 0
                            time.sleep(0.03)
                            continue
                        vy_cap = 0.25
                        cmd_vx = 0.0
                        cmd_vy = core._clamp(lateral_err * 1.15, -vy_cap, vy_cap)
                        if abs(lateral_err) < 0.04:
                            cmd_vy = 0.0
                        cmd_wz = 0.0
                    else:
                        send_state.crab_best_dist = float("inf")
                        send_state.crab_diverge_count = 0
                        cmd_vx = min(abs(float(args.line_cruise_vx)), 0.15)
                        cmd_wz = core._clamp(
                            heading_err_deg * 0.72 + lateral_err * 18.0,
                            -12.0,
                            12.0,
                        )
                if reverse_global_only_active:
                    local_controller.hold_direct_control(0.90)

                if cmd_gear == "crab":
                    local_controller.hold_direct_control(0.60)
                    if (
                        row_end_reverse.row_change_sync_until > time.monotonic()
                        and not row_end_reverse.row_change_sync_sent
                    ):
                        local_controller.send_direct_body_drive("crab", 0.0, 0.0, 0.0, force_brake=True)
                        row_end_reverse.row_change_sync_sent = True
                        current_cmd_gear = "crab"
                        local_status = local_controller.status_snapshot()
                        local_cmd = TwistCommand(vx=0.0, vy=0.0, wz=0.0, updated_at=time.monotonic(), fresh=True)
                        time.sleep(0.08)
                        continue
                    local_controller.send_direct_body_drive("crab", cmd_vx, cmd_vy, math.radians(cmd_wz))
                else:
                    row_end_reverse.row_change_sync_until = 0.0
                    row_end_reverse.row_change_sync_sent = False
                    local_controller.send_direct_drive("4t4d", cmd_vx, math.radians(cmd_wz))
                current_cmd_gear = cmd_gear
                local_status = local_controller.status_snapshot()
                local_cmd = TwistCommand(vx=cmd_vx, vy=cmd_vy, wz=math.radians(cmd_wz), updated_at=time.monotonic(), fresh=True)
            else:
                if row_end_reverse.force_global_only:
                    row_end_reverse.force_global_only = False
                    row_end_reverse.last_global_lateral_err = 0.0
                    row_end_reverse.last_global_heading_err_deg = 0.0
                    row_end_reverse.runaway_same_side_count = 0
                    local_controller.clear_motion_history()
                row_end_reverse.row_change_sync_until = 0.0
                row_end_reverse.row_change_sync_sent = False
                row_end_reverse.forward_mode_sync_until = 0.0
                row_end_reverse.forward_mode_sync_logged = False
                if send_state.crab_locked_until >= 0 and not reversing_here:
                    send_state.crab_locked_until = -1
                    send_state.crab_target_index = -1
                    send_state.crab_best_dist = float("inf")
                    send_state.crab_diverge_count = 0
                local_controller.publish_mode(
                    enable=True,
                    reverse=reversing_here,
                    cruise_vx=cruise_vx,
                    max_wz_deg=local_max_wz_deg,
                    gear="4t4d",
                    low_beam=args.line_low_beam,
                    target_center_offset_px=offset_px,
                    vehicle_direction_angle_deg=direction_angle_deg,
                )
                local_status = local_controller.status_snapshot()
                local_cmd = local_controller.cmd_snapshot()
                current_cmd_gear = "4t4d"

            if not local_cmd.fresh:
                time.sleep(0.05)
                continue
            current_gear = current_cmd_gear

            if row_end_reverse.active and row_end_reverse.reverse_end_index >= row_end_reverse.reverse_start_index:
                reverse_nearest_index, reverse_nearest_dist = _nearest_index_in_range(
                    points,
                    pose,
                    row_end_reverse.reverse_start_index,
                    row_end_reverse.reverse_end_index,
                )
                reverse_target = points[min(row_end_reverse.reverse_end_index, len(points) - 1)]
                reverse_dist = pose.distance_to(reverse_target)
                reverse_tail_index = max(
                    row_end_reverse.reverse_start_index,
                    row_end_reverse.reverse_end_index - 2,
                )
                if (
                    reverse_nearest_index >= reverse_tail_index
                    or reverse_dist <= 0.45
                ):
                    next_index = min(len(points) - 1, row_end_reverse.reverse_end_index + 1)
                    end_reverse_msg = (
                        f"Reverse segment finished near mission index {row_end_reverse.reverse_end_index} "
                        f"(target_dist={reverse_dist:.2f}, nearest_idx={reverse_nearest_index}, "
                        f"nearest_dist={reverse_nearest_dist:.2f}, tail_idx={reverse_tail_index}). "
                        "Stopping first before switching to the next mission stage."
                    )
                    core.log(end_reverse_msg)
                    _append_hybrid_log(hybrid_run_log, end_reverse_msg)
                    _clear_drive_boundary("reverse_segment_finished", next_gear="4t4d")
                    row_end_reverse.stop_until = time.monotonic() + reverse_stop_pause_s
                    row_end_reverse.hard_stop_sent = True
                    row_end_reverse.pending_next_index = next_index
                    time.sleep(0.05)
                    continue

            if not row_end_reverse.active and dist < 0.20:
                start_index = min(len(points) - 1, start_index + 1)

            now = time.monotonic()
            if now - last_cmd_log >= 1.0:
                cmd_log = (
                    f"Hybrid command: mode=lidar-only gear={current_cmd_gear} "
                    f"local_vx={local_cmd.vx:.2f} local_vy={local_cmd.vy:.2f} local_wz={local_cmd.wz:.2f} "
                    f"sent_vx={_safe_float(snapshot.get('motion', {}).get('vx_mps', 0.0)):.2f} "
                    f"sent_vy={_safe_float(snapshot.get('motion', {}).get('vy_mps', 0.0)):.2f} "
                    f"sent_wz={_safe_float(snapshot.get('motion', {}).get('wz_dps', 0.0)):.2f} "
                    f"track_index={start_index} row_end_zone={in_row_end_zone} "
                    f"entry_window={row_entry_global_window} end_window={forward_global_window} "
                    f"row_change_window={row_change_global_window} "
                    f"global_window={global_control_active} dist={dist:.2f} "
                    f"post_row_change_lock={post_row_change_locked} "
                    f"force_global_entry_only={force_global_entry_only} "
                    f"entry_handoff={row_entry_handoff_ready} "
                    f"lidar_entry_pending={row_entry_assist.lidar_pending} "
                    f"lidar_entry_tracking={row_entry_assist.lidar_tracking} "
                    f"reverse_global_only={reverse_global_only_active} "
                    f"reverse={reversing_here}"
                )
                core.log(cmd_log)
                _append_hybrid_log(hybrid_run_log, cmd_log)
                runtime = snapshot.get("_runtime", {}) if isinstance(snapshot, dict) else {}
                waiting_unlock = bool(runtime.get("waiting_unlock", False))
                unlock_now = bool(runtime.get("unlock_now", False))
                if waiting_unlock or unlock_now:
                    unlock_log = f"Hybrid unlock: waiting_unlock={waiting_unlock} unlock_pulse={unlock_now}"
                    core.log(unlock_log)
                    _append_hybrid_log(hybrid_run_log, unlock_log)
                last_cmd_log = now
            if now - last_feedback_log >= 1.0:
                motion = snapshot.get("motion", {})
                steering = snapshot.get("steering", {})
                io_fb = snapshot.get("io", {})
                err_fb = snapshot.get("error", {})
                feedback_log = (
                    f"Hybrid autorun feedback: body_gear={motion.get('gear', '--')} "
                    f"steer_gear={steering.get('gear', '--')} "
                    f"steer_speed={steering.get('wheel_speed_mps', '--')} "
                    f"steer_angle={steering.get('wheel_angle_deg', '--')} "
                    f"unlock_ok={io_fb.get('unlock_ok', '--')} remote_control={io_fb.get('remote_control', '--')} "
                    f"estop={io_fb.get('estop', '--')} error={err_fb.get('level', '--')}/{err_fb.get('type', '--')}"
                )
                pose_log = f"Hybrid autorun pose: x={pose.x:.3f} y={pose.y:.3f} yaw_deg={math.degrees(pose.yaw):.1f}"
                local_log = (
                    f"Hybrid local: state={local_status.state} found={local_status.found} "
                    f"blocked={local_status.obstacle_blocked} lost_frames={local_status.lost_frames} "
                    f"fresh={local_status.fresh}"
                )
                core._log_feedback("Hybrid autorun", snapshot, pose)
                core.log(local_log)
                _append_hybrid_log(hybrid_run_log, feedback_log)
                _append_hybrid_log(hybrid_run_log, pose_log)
                _append_hybrid_log(hybrid_run_log, local_log)
                last_feedback_log = now
            time.sleep(max(0.03, min(0.10, sample_period)))

        core.log("Hybrid autorun finished.")
        _append_hybrid_log(hybrid_run_log, "Hybrid autorun finished.")
        return 0
    finally:
        try:
            if local_controller is not None:
                cruise_vx, offset_px, direction_angle_deg = _local_lidar_drive_settings(local_lidar_config, reverse=False, args=args)
                local_controller.publish_mode(
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
        if local_controller is not None:
            local_controller.close()
        try:
            core._hold_current_gear_stop(controller, send_state, current_gear)
        except Exception:
            pass
        controller.close()
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
    parser.add_argument("--row-centering-trigger-error-m", type=float, default=0.09)
    parser.add_argument("--row-centering-trigger-min-clearance-m", type=float, default=0.035)
    parser.add_argument("--row-centering-global-weight", type=float, default=0.55)
    parser.add_argument("--row-switch-blend-start-dist", type=float, default=3.0)
    parser.add_argument("--row-switch-full-global-dist", type=float, default=1.5)
    parser.add_argument("--lidar-row-entry-settle-sec", type=float, default=0.55)
    parser.add_argument("--lidar-row-entry-stable-frames", type=int, default=4)
    parser.add_argument("--lidar-row-entry-speed", type=float, default=0.07)
    parser.add_argument("--lidar-row-entry-distance", type=float, default=1.0)
    parser.add_argument("--lidar-row-entry-max-wz-deg", type=float, default=1.0)
    parser.add_argument("--lidar-row-entry-max-heading-deg", type=float, default=10.0)
    parser.add_argument("--lidar-row-entry-min-clearance", type=float, default=0.04)
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
    parser.add_argument("--lidar-reverse-sign-flip-guard-error-y", type=float, default=0.06)
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
    parser.add_argument("--lidar-reverse-recenter-error-y", type=float, default=0.06)
    parser.add_argument("--lidar-reverse-recenter-heading-deg", type=float, default=6.0)
    parser.add_argument("--lidar-reverse-recenter-scale", type=float, default=0.45)
    parser.add_argument("--lidar-reverse-error-stop", type=float, default=0.28)
    parser.add_argument("--lidar-reverse-wz-smoothing-alpha", type=float, default=0.0)
    parser.add_argument("--lidar-reverse-lost-hold-sec", type=float, default=0.35)
    parser.add_argument("--lidar-reverse-lost-stop-sec", type=float, default=0.80)
    parser.add_argument("--lidar-reverse-lost-hold-max-wz-deg", type=float, default=1.5)
    parser.add_argument("--lidar-reverse-lost-soft-max-wz-deg", type=float, default=0.8)
    parser.add_argument("--lidar-reverse-start-lock-frames", type=int, default=3)
    parser.add_argument("--lidar-reverse-start-ramp-frames", type=int, default=8)
    parser.add_argument("--lidar-reverse-start-max-wz-deg", type=float, default=0.8)
    parser.add_argument("--lidar-max-wz-delta-deg-per-cycle", type=float, default=1.0)
    parser.add_argument("--lidar-enable-4t4d-steering-assist", action="store_true")
    parser.add_argument("--lidar-steering-assist-wheelbase-m", type=float, default=0.85)
    parser.add_argument("--lidar-steering-assist-gain", type=float, default=1.0)
    parser.add_argument("--lidar-steering-assist-max-angle-deg", type=float, default=12.0)
    parser.add_argument("--lidar-steering-assist-min-speed-mps", type=float, default=0.05)
    parser.add_argument("--lidar-steering-assist-speed-mps", type=float, default=0.20)


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
