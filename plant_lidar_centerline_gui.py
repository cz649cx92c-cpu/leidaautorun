#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

ROOT = Path(__file__).resolve().parent
CONTROL_ROOT = ROOT.parent / "control"
ROS_SETUP = Path("/opt/ros/humble/setup.bash")
WORKSPACE_SETUP = ROOT.parent / "install" / "setup.bash"
FOLLOWER_PATH = ROOT / "plant_lidar_centerline_follower.py"
CALIBRATION_PATH = ROOT / "config" / "lidar_calibration.json"


def _merge_ros_env() -> None:
    if not ROS_SETUP.exists():
        return
    parts = [f"source {ROS_SETUP}"]
    if WORKSPACE_SETUP.exists():
        parts.append(f"source {WORKSPACE_SETUP}")
    parts.append("env")
    try:
        result = subprocess.run(
            ["bash", "-lc", " && ".join(parts)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        return
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in {"LD_LIBRARY_PATH", "PYTHONPATH", "AMENT_PREFIX_PATH", "CMAKE_PREFIX_PATH", "PATH"}:
            os.environ[key] = value


_merge_ros_env()
if str(CONTROL_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTROL_ROOT))

from fw_mini_controller import FWMiniController  # noqa: E402
from row_geometry import RowDebugData, RowEstimate, RowFollowerConfig, blend_line, estimate_row  # noqa: E402

import rclpy  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from sensor_msgs.msg import LaserScan  # noqa: E402


DISPLAY_RANGE_FRONT_M = 3.0
DISPLAY_RANGE_BACK_M = 1.1
DISPLAY_RANGE_SIDE_M = 3.0


def ensure_can_interface_up(channel: str, bitrate: int) -> tuple[bool, list[str]]:
    logs: list[str] = []
    operstate_path = Path(f"/sys/class/net/{channel}/operstate")
    try:
        state = operstate_path.read_text(encoding="utf-8").strip().lower()
    except Exception:
        logs.append(f"{channel} was not found on this system.")
        return False, logs
    if state in {"up", "unknown"}:
        logs.append(f"{channel} is already up.")
        return True, logs
    logs.append(f"{channel} is {state}. Bringing it up at {bitrate} bitrate...")
    cmds = [
        ["sudo", "ip", "link", "set", channel, "down"],
        ["sudo", "ip", "link", "set", channel, "up", "type", "can", "bitrate", str(int(bitrate))],
    ]
    for cmd in cmds:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
        if result.returncode != 0:
            logs.append(f"Failed to run: {' '.join(cmd)}")
            output = (result.stdout or "").strip()
            if output:
                logs.append(output)
            return False, logs
    logs.append(f"{channel} is now up.")
    return True, logs


class ScanProbe(Node):
    def __init__(self, topic: str) -> None:
        super().__init__("plant_lidar_gui_probe")
        self._lock = threading.Lock()
        self.last_scan: LaserScan | None = None
        self.last_time = 0.0
        self.create_subscription(LaserScan, topic, self._on_scan, 10)

    def _on_scan(self, msg: LaserScan) -> None:
        with self._lock:
            self.last_scan = msg
            self.last_time = time.monotonic()

    def snapshot(self) -> tuple[LaserScan | None, float]:
        with self._lock:
            return self.last_scan, self.last_time


class PlantLidarGUI:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self._load_calibration()
        self.root = tk.Tk()
        self.root.title("Plant Lidar Centerline")
        self.root.geometry("980x680")
        self.root.minsize(820, 560)

        self.auto_process: subprocess.Popen | None = None
        self.last_auto_exit_code: int | None = None
        self.last_auto_exit_logged = False
        self.auto_log_lock = threading.Lock()
        self.auto_log_lines: list[str] = []
        self.lidar_process: subprocess.Popen | None = None
        self.lidar_log_lock = threading.Lock()
        self.lidar_log_lines: list[str] = []
        self.scan_probe: ScanProbe | None = None
        self.scan_executor: SingleThreadedExecutor | None = None
        self.scan_thread: threading.Thread | None = None
        self.scan_topic_active = ""
        self.can_controller: FWMiniController | None = None
        self.status_after_id: str | None = None

        self.latest_status_payload: dict[str, object] = {}
        self.latest_debug_payload: dict[str, object] = {}
        self.latest_estimate = RowEstimate(found=False)
        self.latest_debug = RowDebugData(
            raw_points=[],
            web_points=[],
            points=[],
            left_points=[],
            right_points=[],
            center_points=[],
            virtual_left_points=[],
            virtual_right_points=[],
            virtual_center_points=[],
            left_line=None,
            right_line=None,
            center_line=None,
        )
        self.last_good_row_width = float(args.row_width)
        self.preview_center_line: tuple[float, float] | None = None
        self.preview_left_line: tuple[float, float] | None = None
        self.preview_right_line: tuple[float, float] | None = None
        self.preview_center_y = 0.0
        self.preview_heading_deg = 0.0
        self.last_display_points = []
        self.last_display_left_points = []
        self.last_display_right_points = []
        self.last_display_center_points = []
        self.last_display_virtual_left_points = []
        self.last_display_virtual_right_points = []
        self.last_display_left_line: tuple[float, float] | None = None
        self.last_display_right_line: tuple[float, float] | None = None
        self.last_display_center_line: tuple[float, float] | None = None
        self.last_display_time = 0.0

        self.speed_var = tk.StringVar(value=f"{args.auto_speed:.2f}")
        self.center_target_var = tk.StringVar(value=f"{getattr(args, 'center_y_target', 0.0):.3f}")
        self.yaw_corr_var = tk.StringVar(value=f"{args.lidar_yaw_correction_deg:.2f}")
        self.x_offset_var = tk.StringVar(value=f"{args.lidar_x_offset_m:.3f}")
        self.y_offset_var = tk.StringVar(value=f"{args.lidar_y_offset_m:.3f}")
        self.can_state_var = tk.StringVar(value="CAN disconnected")
        self.lidar_state_var = tk.StringVar(value="Lidar stopped")
        self.auto_state_var = tk.StringVar(value="stopped")
        self.mode_state_var = tk.StringVar(value="mode: -")
        self.error_state_var = tk.StringVar(value="error_y: -")
        self.direct_error_state_var = tk.StringVar(value="center_y_direct_error: -")
        self.heading_state_var = tk.StringVar(value="heading_deg: -")
        self.reverse_steer_sign_state_var = tk.StringVar(value="reverse_steer_sign: -")
        self.frame_state_var = tk.StringVar(value="display/control frame: - / -")
        self.error_source_state_var = tk.StringVar(value="error_source: -")
        self.reverse_error_state_var = tk.StringVar(value="reverse_line_error_y: -")
        self.suggested_center_state_var = tk.StringVar(value="suggested_center_y_target: -")
        self.cmd_state_var = tk.StringVar(value="cmd_vx: -   cmd_wz: -")
        self.fb_state_var = tk.StringVar(value="fb_vx: -   fb_wz: -")
        self.display_state_var = tk.StringVar(value="display_hold: false   points: 0   last points age: -")
        self.calib_state_var = tk.StringVar(value=self._calibration_status_text())
        self.warning_state_var = tk.StringVar(value="warning: -")

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._auto_initialize)
        self._schedule_refresh()

    def _build_ui(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)

        top = ttk.Frame(outer)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        top.columnconfigure(1, weight=0)
        ttk.Label(top, text="Speed:").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.speed_var, width=10).grid(row=0, column=1, sticky="w", padx=(6, 12))
        ttk.Label(top, text="Center target m:").grid(row=0, column=2, sticky="e")
        ttk.Entry(top, textvariable=self.center_target_var, width=10).grid(row=0, column=3, sticky="w", padx=(6, 12))
        ttk.Button(top, text="Auto Forward", command=self.start_auto_forward).grid(row=0, column=4, padx=4)
        ttk.Button(top, text="Auto Reverse", command=self.start_auto_reverse).grid(row=0, column=5, padx=4)
        ttk.Button(top, text="Stop", command=self.stop_auto_follow).grid(row=0, column=6, padx=4)
        ttk.Label(top, text="Yaw corr deg:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(top, textvariable=self.yaw_corr_var, width=10).grid(row=1, column=1, sticky="w", padx=(6, 12), pady=(8, 0))
        ttk.Label(top, text="Vertical offset m:").grid(row=1, column=2, sticky="e", pady=(8, 0))
        ttk.Entry(top, textvariable=self.x_offset_var, width=10).grid(row=1, column=3, sticky="w", pady=(8, 0))
        ttk.Label(top, text="Horizontal offset m:").grid(row=1, column=4, sticky="e", pady=(8, 0))
        ttk.Entry(top, textvariable=self.y_offset_var, width=10).grid(row=1, column=5, sticky="w", pady=(8, 0), padx=(6, 12))
        ttk.Button(top, text="Apply Calib", command=self.apply_calibration).grid(row=1, column=6, padx=4, pady=(8, 0))
        ttk.Button(top, text="Save Calib", command=self.save_calibration).grid(row=1, column=7, padx=4, pady=(8, 0))

        center = ttk.Panedwindow(outer, orient=tk.HORIZONTAL)
        center.grid(row=1, column=0, sticky="nsew")

        plot_frame = ttk.Frame(center)
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)
        self.plot_canvas = tk.Canvas(plot_frame, background="#101418", highlightthickness=0)
        self.plot_canvas.grid(row=0, column=0, sticky="nsew")
        center.add(plot_frame, weight=4)

        status_frame = ttk.LabelFrame(center, text="Status", padding=8)
        status_frame.columnconfigure(0, weight=1)
        for idx, var in enumerate(
            [
                self.can_state_var,
                self.lidar_state_var,
                self.auto_state_var,
                self.mode_state_var,
                self.error_state_var,
                self.direct_error_state_var,
                self.heading_state_var,
                self.reverse_steer_sign_state_var,
                self.frame_state_var,
                self.error_source_state_var,
                self.reverse_error_state_var,
                self.suggested_center_state_var,
                self.cmd_state_var,
                self.fb_state_var,
                self.display_state_var,
                self.calib_state_var,
                self.warning_state_var,
            ]
        ):
            ttk.Label(status_frame, textvariable=var, anchor="w").grid(row=idx, column=0, sticky="ew", pady=2)
        center.add(status_frame, weight=1)

        log_frame = ttk.LabelFrame(outer, text="Log", padding=6)
        log_frame.grid(row=2, column=0, sticky="nsew", pady=(8, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(1, weight=1)
        ttk.Button(log_frame, text="Copy All", command=self._copy_all_logs).grid(row=0, column=0, sticky="e", pady=(0, 6))
        self.log_text = tk.Text(log_frame, height=5, wrap="word")
        self.log_text.grid(row=1, column=0, sticky="nsew")
        self.log_text.insert("end", "GUI ready.\n")
        self.log_text.configure(state="disabled")

    def _auto_initialize(self) -> None:
        # The follower process owns CAN while automatic motion is active.
        # Keeping a second GUI-side ROS/CAN client caused environment errors
        # and can contend with the actual command sender.
        self.can_state_var.set("CAN managed by auto follower")
        self.start_lidar()
        self._start_scan_probe()

    def _schedule_refresh(self) -> None:
        self._refresh_status()
        self.status_after_id = self.root.after(250, self._schedule_refresh)

    def _refresh_status(self) -> None:
        self._ensure_scan_probe_topic()
        self._refresh_lidar_state()
        self._poll_can_feedback()
        self._flush_auto_logs()
        self._flush_lidar_logs()
        self._refresh_scan_plot()

        if self.auto_process is not None and self.auto_process.poll() is not None:
            self.last_auto_exit_code = self.auto_process.returncode
            if not self.last_auto_exit_logged:
                self._log(f"Auto exited with code {self.auto_process.returncode}.")
                self.last_auto_exit_logged = True
            self.auto_state_var.set(f"Auto exited code={self.auto_process.returncode}")
            self.auto_process = None
        if self.auto_state_var.get() == "stopped":
            self.cmd_state_var.set("cmd_vx: 0.00   cmd_wz: 0.00")

    def _load_calibration(self) -> None:
        if not CALIBRATION_PATH.exists():
            return
        try:
            payload = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
        except Exception:
            return
        self.args.lidar_yaw_correction_deg = float(payload.get("lidar_yaw_correction_deg", 0.0))
        self.args.lidar_x_offset_m = float(payload.get("lidar_x_offset_m", 0.0))
        self.args.lidar_y_offset_m = float(payload.get("lidar_y_offset_m", 0.0))

    def _calibration_status_text(self) -> str:
        return (
            f"yaw_corr_deg: {float(self.args.lidar_yaw_correction_deg):+.2f}   "
            f"vertical_offset_m: {float(self.args.lidar_x_offset_m):+.3f}   "
            f"horizontal_offset_m: {float(self.args.lidar_y_offset_m):+.3f}"
        )

    def apply_calibration(self) -> None:
        self.args.lidar_yaw_correction_deg = float(self.yaw_corr_var.get().strip())
        self.args.lidar_x_offset_m = float(self.x_offset_var.get().strip())
        self.args.lidar_y_offset_m = float(self.y_offset_var.get().strip())
        self.calib_state_var.set(self._calibration_status_text())
        self._log("Calibration applied.")

    def save_calibration(self) -> None:
        self.apply_calibration()
        CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "lidar_yaw_correction_deg": float(self.args.lidar_yaw_correction_deg),
            "lidar_x_offset_m": float(self.args.lidar_x_offset_m),
            "lidar_y_offset_m": float(self.args.lidar_y_offset_m),
        }
        CALIBRATION_PATH.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        self._log(f"Calibration saved to {CALIBRATION_PATH}.")

    def connect_can(self) -> None:
        if self.can_controller is not None:
            self.can_state_var.set("CAN connected")
            return
        try:
            ok, logs = ensure_can_interface_up(self.args.channel, self.args.bitrate)
            for line in logs:
                self._log(line)
            if not ok:
                self.can_state_var.set("CAN disconnected")
                return
            self.can_controller = FWMiniController(self.args.interface, self.args.channel, self.args.bitrate)
            self.can_state_var.set("CAN connected")
        except Exception as exc:
            self.can_state_var.set(f"CAN error: {exc}")
            self._log(f"CAN error: {exc}")

    def _poll_can_feedback(self) -> None:
        if self.can_controller is None:
            return
        try:
            feedback = {}
            for item in self.can_controller.poll():
                feedback[item.get("name", "")] = item.get("data", {})
            ctrl = feedback.get("ctrl_fb", {})
            self.fb_state_var.set(
                f"fb_vx: {float(ctrl.get('vx_mps', ctrl.get('vx', 0.0))):+.3f}   "
                f"fb_wz: {float(ctrl.get('wz_dps', ctrl.get('wz', 0.0))):+.3f}"
            )
        except Exception as exc:
            self.can_state_var.set(f"CAN error: {exc}")

    def _auto_command(self, reverse: bool) -> list[str]:
        speed = abs(float(self.speed_var.get().strip()))
        cmd = (
            f"source {ROS_SETUP}"
            + (f" && source {WORKSPACE_SETUP}" if WORKSPACE_SETUP.exists() else "")
            + f" && /usr/bin/python3 {FOLLOWER_PATH}"
            + f" --scan-topic {self.args.front_scan_topic}"
            + f" --front-scan-topic {self.args.front_scan_topic}"
            + f" --rear-scan-topic {self.args.rear_scan_topic}"
            + f" --status-topic {self.args.status_topic}"
            + f" --interface {self.args.interface}"
            + f" --channel {self.args.channel}"
            + f" --bitrate {int(self.args.bitrate)}"
            + f" --gear {self.args.gear}"
            + f" --speed {speed}"
            + f" --row-width {float(self.args.row_width)}"
            + f" --boundary-width-tolerance-m 0.0"
            + f" --center-y-target {float(self.center_target_var.get().strip())}"
            + f" --sensor-yaw-deg {float(self.args.sensor_yaw_deg)}"
            + f" --lidar-yaw-correction-deg {float(self.args.lidar_yaw_correction_deg)}"
            + f" --lidar-x-offset-m {float(self.args.lidar_x_offset_m)}"
            + f" --lidar-y-offset-m {float(self.args.lidar_y_offset_m)}"
            + f" --front-sensor-yaw-deg {float(self.args.front_sensor_yaw_deg)}"
            + f" --rear-sensor-yaw-deg {float(self.args.rear_sensor_yaw_deg)}"
            + f" --rear-extrinsics-confirmed"
            + f" --forward-lookahead-x 0.6"
            + f" --control-deadband-y 0.001"
            + f" --forward-lost-hold-sec 0.35"
            + f" --forward-lost-stop-sec 0.50"
            + f" --forward-lost-hold-wz-scale 0.5"
            + f" --forward-lost-hold-max-wz-deg 0.6"
            + f" --k-heading 0.05"
            + f" --reverse-max-wz-deg 5.0"
            + f" --reverse-steer-sign -1.0"
            + (" --reverse" if reverse else "")
        )
        return ["bash", "-lc", cmd]

    def start_auto_forward(self) -> None:
        self._start_auto(reverse=False)

    def start_auto_reverse(self) -> None:
        self._start_auto(reverse=True)

    def _start_auto(self, reverse: bool) -> None:
        self.stop_auto_follow(log_stop=False)
        self.start_lidar()
        self._switch_scan_probe_topic(self.args.rear_scan_topic if reverse else self.args.front_scan_topic)
        try:
            self.auto_process = subprocess.Popen(
                self._auto_command(reverse),
                cwd=str(ROOT.parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            self.auto_state_var.set(f"Auto start failed: {exc}")
            self._log(f"Auto start failed: {exc}")
            return
        self.last_auto_exit_code = None
        self.last_auto_exit_logged = False
        self.auto_state_var.set("Auto reverse" if reverse else "Auto forward")
        self._log("Auto reverse started." if reverse else "Auto forward started.")
        threading.Thread(target=self._read_auto_output, daemon=True).start()

    def stop_auto_follow(self, log_stop: bool = True) -> None:
        if self.auto_process is not None and self.auto_process.poll() is None:
            try:
                self.auto_process.terminate()
                self.auto_process.wait(timeout=2.0)
            except Exception:
                try:
                    self.auto_process.kill()
                except Exception:
                    pass
        if self.auto_process is not None:
            self.last_auto_exit_code = self.auto_process.poll()
        self.auto_process = None
        self.auto_state_var.set("stopped")
        self.cmd_state_var.set("cmd_vx: 0.00   cmd_wz: 0.00")
        if log_stop:
            self._log("Auto stopped.")

    def _read_auto_output(self) -> None:
        process = self.auto_process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            text = line.rstrip()
            with self.auto_log_lock:
                self.auto_log_lines.append(text)
                self.auto_log_lines = self.auto_log_lines[-1000:]
        returncode = process.poll()
        with self.auto_log_lock:
            self.auto_log_lines.append(f"[auto] process exited code={returncode}")
            self.auto_log_lines = self.auto_log_lines[-1000:]

    def start_lidar(self) -> None:
        # Dual drivers are provided by autorunlida; do not start legacy /dev/lidar.
        self.lidar_state_var.set("Dual lidar drivers provided by autorunlida")
        return
        if self.lidar_process is not None and self.lidar_process.poll() is None:
            self.lidar_state_var.set("Lidar running")
            return
        port_name = self._detect_lidar_port()
        if not port_name:
            self.lidar_state_var.set("Lidar stopped: port not found")
            self._log("Lidar port not found.")
            return
        setup_cmd = f"source {ROS_SETUP}"
        if WORKSPACE_SETUP.exists():
            setup_cmd += f" && source {WORKSPACE_SETUP}"
        cmd = [
            "bash",
            "-lc",
            (
                f"{setup_cmd} && "
                "ros2 run lidar_pkg lidar_node "
                f"--ros-args -p port_name:={port_name} -p frame_id:=laser"
            ),
        ]
        try:
            self.lidar_process = subprocess.Popen(
                cmd,
                cwd=str(ROOT.parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
        except Exception as exc:
            self.lidar_state_var.set(f"Lidar error: {exc}")
            self._log(f"Lidar error: {exc}")
            self.lidar_process = None
            return
        self.lidar_state_var.set("Lidar running")
        threading.Thread(target=self._read_lidar_output, daemon=True).start()

    def stop_lidar(self) -> None:
        if self.lidar_process is None:
            self.lidar_state_var.set("Lidar stopped")
            return
        process = self.lidar_process
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=2.0)
            except Exception:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except Exception:
                    try:
                        process.kill()
                    except Exception:
                        pass
        self.lidar_process = None
        self.lidar_state_var.set("Lidar stopped")

    def _read_lidar_output(self) -> None:
        process = self.lidar_process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            text = line.rstrip()
            with self.lidar_log_lock:
                self.lidar_log_lines.append(text)
                self.lidar_log_lines = self.lidar_log_lines[-100:]

    def _flush_auto_logs(self) -> None:
        with self.auto_log_lock:
            lines = self.auto_log_lines[:]
            self.auto_log_lines.clear()
        for line in lines:
            self._log(line)
            self._update_status_from_line(line)

    def _flush_lidar_logs(self) -> None:
        with self.lidar_log_lock:
            lines = self.lidar_log_lines[:]
            self.lidar_log_lines.clear()
        for line in lines:
            if line:
                self._log(f"[lidar] {line}")

    def _update_status_from_line(self, line: str) -> None:
        text = line.strip()
        if not text:
            return
        if text.startswith("[debug] "):
            try:
                self.latest_debug_payload = json.loads(text[8:])
            except Exception:
                pass
            return
        if text.startswith("{") and text.endswith("}"):
            try:
                payload = json.loads(text)
            except Exception:
                return
            self.latest_status_payload = payload
            self.mode_state_var.set(f"mode: {payload.get('mode', '-')}")
            self.error_state_var.set(f"error_y_used_for_control: {self.latest_debug_payload.get('error_y_used_for_control', '-')}")
            self.direct_error_state_var.set(f"center_y_direct_error: {self.latest_debug_payload.get('center_y_direct_error', '-')}")
            self.heading_state_var.set(f"heading_deg: {payload.get('heading_deg', '-')}")
            self.reverse_steer_sign_state_var.set(
                f"reverse_steer_sign: {self.latest_debug_payload.get('reverse_steer_sign', '-')}"
            )
            self.frame_state_var.set(
                "display/control frame: "
                f"{self.latest_debug_payload.get('display_frame', '-')} / "
                f"{self.latest_debug_payload.get('control_frame', '-')}"
            )
            self.error_source_state_var.set(
                f"error_source: {self.latest_debug_payload.get('error_source', '-')}"
            )
            self.reverse_error_state_var.set(
                f"reverse_line_error_y: {self.latest_debug_payload.get('reverse_line_error_y', '-')}"
            )
            self.suggested_center_state_var.set(
                "suggested_center_y_target: "
                f"{self.latest_debug_payload.get('suggested_center_y_target', '-')}"
            )
            self.cmd_state_var.set(
                f"cmd_vx: {payload.get('cmd_vx_mps', '-')}   cmd_wz: {payload.get('cmd_wz_deg_s', '-')}"
            )
            self.warning_state_var.set(f"warning: {self.latest_debug_payload.get('warning', '-')}")

    def _detect_lidar_port(self) -> str:
        preferred = ["/dev/lidar"]
        fallback_patterns = ["/dev/ttyACM*", "/dev/ttyUSB*"]
        for path in preferred:
            if Path(path).exists():
                return path
        candidates: list[str] = []
        for pattern in fallback_patterns:
            candidates.extend(sorted(glob.glob(pattern)))
        return candidates[0] if candidates else ""

    def _start_scan_probe(self) -> None:
        if self.scan_probe is not None:
            return
        if not rclpy.ok():
            rclpy.init()
        self.scan_topic_active = self.args.scan_topic
        self.scan_probe = ScanProbe(self.scan_topic_active)
        self.scan_executor = SingleThreadedExecutor()
        self.scan_executor.add_node(self.scan_probe)
        self.scan_thread = threading.Thread(target=self.scan_executor.spin, daemon=True)
        self.scan_thread.start()

    def _switch_scan_probe_topic(self, topic: str) -> None:
        topic = str(topic or "").strip()
        if not topic or topic == self.scan_topic_active:
            return
        # A topic switch must never render or filter data from the previous lidar.
        # Clear every preview/display cache before subscribing to the new source.
        self._stop_scan_probe()
        self.args.scan_topic = topic
        self.last_display_points = []
        self.last_display_left_points = []
        self.last_display_right_points = []
        self.last_display_center_points = []
        self.last_display_virtual_left_points = []
        self.last_display_virtual_right_points = []
        self.last_display_left_line = None
        self.last_display_right_line = None
        self.last_display_center_line = None
        self.last_display_time = 0.0
        self.latest_estimate = RowEstimate(found=False)
        self.latest_debug = RowDebugData(
            raw_points=[], web_points=[], points=[], left_points=[], right_points=[],
            center_points=[], virtual_left_points=[], virtual_right_points=[],
            virtual_center_points=[],
            left_line=None, right_line=None, center_line=None,
        )
        self._start_scan_probe()
        self.preview_left_line = None
        self.preview_right_line = None
        self.preview_center_line = None
        self.preview_center_y = 0.0
        self.preview_heading_deg = 0.0
        self.lidar_state_var.set(f"Preview topic: {topic}")

    def _ensure_scan_probe_topic(self) -> None:
        if self.scan_probe is None:
            self._start_scan_probe()

    def _stop_scan_probe(self) -> None:
        if self.scan_executor is not None:
            self.scan_executor.shutdown()
        if self.scan_probe is not None:
            try:
                self.scan_probe.destroy_node()
            except Exception:
                pass
        self.scan_probe = None
        self.scan_executor = None
        if self.scan_thread is not None:
            self.scan_thread.join(timeout=1.0)
        self.scan_thread = None

    def _refresh_lidar_state(self) -> None:
        if self.lidar_process is not None and self.lidar_process.poll() is not None:
            self.lidar_state_var.set(f"Lidar stopped code={self.lidar_process.returncode}")
            self.lidar_process = None

    def _row_cfg(self) -> RowFollowerConfig:
        return RowFollowerConfig(
            row_width=float(self.args.row_width),
            min_row_width=0.48,
            max_row_width=0.78,
            lookahead_x=0.6,
            forward_min=0.25,
            forward_max=1.20,
            lateral_limit=0.60,
            range_min=0.05,
            range_max=6.0,
            bin_size=0.20,
            min_points=12,
            min_bins=2,
            min_line_bins=4,
            min_side_points_per_bin=2,
            center_deadband=0.03,
            left_percentile=20.0,
            right_percentile=80.0,
            sensor_yaw_deg=float(self.args.sensor_yaw_deg),
            lidar_yaw_correction_deg=float(self.args.lidar_yaw_correction_deg),
            lidar_x_offset_m=float(self.args.lidar_x_offset_m),
            lidar_y_offset_m=float(self.args.lidar_y_offset_m),
            boundary_max_gap_x=0.45,
            vehicle_half_width=0.20,
            safety_margin=0.04,
            center_jump_reject=0.25,
            one_side_center_jump_reject=0.30,
        )

    def _refresh_scan_plot(self) -> None:
        canvas = self.plot_canvas
        canvas.delete("all")
        width = max(1, int(canvas.winfo_width()))
        height = max(1, int(canvas.winfo_height()))
        canvas.create_rectangle(0, 0, width, height, fill="#101418", outline="")
        self._draw_axes(width, height)
        self._draw_vehicle(width, height)

        if self.scan_probe is None:
            return
        scan, scan_time = self.scan_probe.snapshot()
        if scan is None or (time.monotonic() - scan_time) > 0.5:
            self.display_state_var.set(
                f"display_hold: false   points: 0   waiting: "
                f"{self.scan_topic_active or self.args.scan_topic}"
            )
            canvas.create_text(
                width / 2,
                height / 2,
                fill="#9ca3af",
                text=f"Waiting for {self.scan_topic_active or self.args.scan_topic}",
            )
            return

        estimate, debug = estimate_row(scan, self._row_cfg(), self.last_good_row_width)
        self.latest_estimate = estimate
        self.latest_debug = debug
        if estimate.found and estimate.row_width > 0.0:
            self.last_good_row_width = float(estimate.row_width)

        if estimate.found:
            left_line = blend_line(self.preview_left_line, debug.left_line, keep_ratio=0.7)
            right_line = blend_line(self.preview_right_line, debug.right_line, keep_ratio=0.7)
            center_line = blend_line(self.preview_center_line, debug.center_line, keep_ratio=0.7)
            center_y = float(center_line[1]) if center_line is not None else float(estimate.center_y)
            heading_deg = math.degrees(math.atan(center_line[0])) if center_line is not None else float(estimate.heading_rad) * 180.0 / 3.141592653589793
            center_jump = abs(center_y - self.preview_center_y) if self.preview_center_line is not None else 0.0
            heading_jump = abs(heading_deg - self.preview_heading_deg) if self.preview_center_line is not None else 0.0
            if self.preview_center_line is not None and center_jump > 0.03:
                center_line = self.preview_center_line
                left_line = self.preview_left_line
                right_line = self.preview_right_line
            elif self.preview_center_line is not None and heading_jump > 5.0:
                center_line = self.preview_center_line
                left_line = self.preview_left_line
                right_line = self.preview_right_line
            self.preview_left_line = left_line
            self.preview_right_line = right_line
            self.preview_center_line = center_line
            self.preview_center_y = center_y
            self.preview_heading_deg = heading_deg
            debug.left_line = left_line
            debug.right_line = right_line
            debug.center_line = center_line

        now = time.monotonic()
        current_scan_points_count = len(debug.web_points) if debug.web_points is not None else 0
        display_hold = False
        lidar_display_timeout = False
        display_hold_age_sec = 0.0

        if current_scan_points_count > 0:
            self.last_display_points = debug.web_points
            self.last_display_left_points = debug.left_points
            self.last_display_right_points = debug.right_points
            self.last_display_center_points = debug.center_points
            self.last_display_virtual_left_points = debug.virtual_left_points
            self.last_display_virtual_right_points = debug.virtual_right_points
            self.last_display_left_line = debug.left_line
            self.last_display_right_line = debug.right_line
            self.last_display_center_line = debug.center_line
            self.last_display_time = now
            draw_points = debug.web_points
            draw_left_points = debug.left_points
            draw_right_points = debug.right_points
            draw_center_points = debug.center_points
            draw_virtual_left_points = debug.virtual_left_points
            draw_virtual_right_points = debug.virtual_right_points
            draw_left_line = debug.left_line
            draw_right_line = debug.right_line
            draw_center_line = debug.center_line
        else:
            display_hold_age_sec = now - self.last_display_time if self.last_display_time > 0.0 else float("inf")
            if self.last_display_time > 0.0 and display_hold_age_sec < 0.5:
                display_hold = True
                draw_points = self.last_display_points
                draw_left_points = self.last_display_left_points
                draw_right_points = self.last_display_right_points
                draw_center_points = self.last_display_center_points
                draw_virtual_left_points = self.last_display_virtual_left_points
                draw_virtual_right_points = self.last_display_virtual_right_points
                draw_left_line = self.last_display_left_line
                draw_right_line = self.last_display_right_line
                draw_center_line = self.last_display_center_line
            else:
                lidar_display_timeout = True
                draw_points = []
                draw_left_points = []
                draw_right_points = []
                draw_center_points = []
                draw_virtual_left_points = []
                draw_virtual_right_points = []
                draw_left_line = None
                draw_right_line = None
                draw_center_line = None

        last_display_points_count = len(self.last_display_points) if self.last_display_points is not None else 0
        age_text = f"{display_hold_age_sec:.2f}s" if (display_hold or lidar_display_timeout) and self.last_display_time > 0.0 else "-"
        self.display_state_var.set(
            f"display_hold: {'true' if display_hold else 'false'}   "
            f"points: {current_scan_points_count}   last points age: {age_text}"
        )

        self._draw_points(draw_points, width, height, "#475569", 2)
        self._draw_points(draw_left_points, width, height, "#22c55e", 3)
        self._draw_points(draw_right_points, width, height, "#f97316", 3)
        self._draw_points(draw_center_points, width, height, "#f43f5e", 3)
        self._draw_line(getattr(debug, "candidate_left_line", None), 0.15, 1.6, width, height, "#64748b", 1, dash=(4, 4))
        self._draw_line(getattr(debug, "candidate_right_line", None), 0.15, 1.6, width, height, "#64748b", 1, dash=(4, 4))
        self._draw_line(draw_left_line, 0.15, 1.6, width, height, "#16a34a", 2)
        self._draw_line(draw_right_line, 0.15, 1.6, width, height, "#ea580c", 2)
        self._draw_line_from_points(draw_virtual_left_points, width, height, "#86efac", 2, dash=(6, 4))
        self._draw_line_from_points(draw_virtual_right_points, width, height, "#fdba74", 2, dash=(6, 4))
        self._draw_line(draw_center_line, 0.0, 1.0, width, height, "#e11d48", 4)
        self._draw_line(draw_center_line, 0.0, -1.0, width, height, "#93c5fd", 2, dash=(6, 4))

        self.latest_debug_payload.update(
            {
                "current_scan_points_count": current_scan_points_count,
                "last_display_points_count": last_display_points_count,
                "display_hold": display_hold,
                "display_hold_age_sec": 0.0 if not display_hold else display_hold_age_sec,
                "lidar_display_timeout": lidar_display_timeout,
            }
        )

    def _draw_axes(self, width: int, height: int) -> None:
        origin_x = width * 0.5
        origin_y = self._robot_center_y(height)
        self.plot_canvas.create_line(origin_x, 16, origin_x, height - 16, fill="#334155")
        self.plot_canvas.create_line(20, origin_y, width - 20, origin_y, fill="#334155")

    def _draw_vehicle(self, width: int, height: int) -> None:
        cx, cy = self._project(0.0, 0.0, width, height)
        self.plot_canvas.create_rectangle(cx - 14, cy - 18, cx + 14, cy + 18, outline="#f8fafc", width=2)
        self.plot_canvas.create_line(cx, cy, cx, cy - 25, fill="#22d3ee", width=3)

    def _draw_points(self, points, width: int, height: int, color: str, radius: int) -> None:
        if points is None:
            return
        for point in points:
            x, y = self._project(float(point[0]), float(point[1]), width, height)
            self.plot_canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color, outline="")

    def _draw_line(
        self,
        line: tuple[float, float] | None,
        x0: float,
        x1: float,
        width: int,
        height: int,
        color: str,
        line_width: int,
        dash: tuple[int, int] | None = None,
    ) -> None:
        if line is None:
            return
        a, b = line
        p0 = self._project(x0, a * x0 + b, width, height)
        p1 = self._project(x1, a * x1 + b, width, height)
        self.plot_canvas.create_line(p0[0], p0[1], p1[0], p1[1], fill=color, width=line_width, dash=dash)

    def _draw_line_from_points(
        self,
        points,
        width: int,
        height: int,
        color: str,
        line_width: int,
        dash: tuple[int, int] | None = None,
    ) -> None:
        if points is None or len(points) < 2:
            return
        coords: list[float] = []
        for point in points:
            px, py = self._project(float(point[0]), float(point[1]), width, height)
            coords.extend([px, py])
        self.plot_canvas.create_line(*coords, fill=color, width=line_width, dash=dash)

    def _project(self, x_m: float, y_m: float, width: int, height: int) -> tuple[float, float]:
        meters_to_pixels = self._meters_to_pixels(width, height)
        robot_center_x = width * 0.5
        robot_center_y = self._robot_center_y(height)
        px = robot_center_x + y_m * meters_to_pixels
        py = robot_center_y - x_m * meters_to_pixels
        return px, py

    def _robot_center_y(self, height: int) -> float:
        usable_height = max(1.0, height - 40.0)
        meters_to_pixels = min(
            max(1.0, (self.plot_canvas.winfo_width() - 40.0) / (2.0 * DISPLAY_RANGE_SIDE_M)),
            max(1.0, usable_height / (DISPLAY_RANGE_FRONT_M + DISPLAY_RANGE_BACK_M)),
        )
        return height - 20 - DISPLAY_RANGE_BACK_M * meters_to_pixels

    def _meters_to_pixels(self, width: int, height: int) -> float:
        usable_width = max(1.0, width - 40.0)
        usable_height = max(1.0, height - 40.0)
        return min(
            usable_width / (2.0 * DISPLAY_RANGE_SIDE_M),
            usable_height / (DISPLAY_RANGE_FRONT_M + DISPLAY_RANGE_BACK_M),
        )

    def _log(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[{stamp}] {text}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _copy_all_logs(self) -> None:
        content = self.log_text.get("1.0", "end-1c")
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self.root.update_idletasks()
        self._log("Copied all logs to clipboard.")

    def _on_close(self) -> None:
        if self.status_after_id is not None:
            self.root.after_cancel(self.status_after_id)
            self.status_after_id = None
        self.stop_auto_follow(log_stop=False)
        self.stop_lidar()
        self._stop_scan_probe()
        self.root.destroy()

    def run(self) -> int:
        self.root.mainloop()
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal GUI for lidar centerline auto-follow")
    parser.add_argument("--interface", default="socketcan")
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--bitrate", type=int, default=500000)
    parser.add_argument("--gear", default="4t4d")
    parser.add_argument("--scan-topic", default="/front/scan")
    parser.add_argument("--front-scan-topic", default="/front/scan")
    parser.add_argument("--rear-scan-topic", default="/rear/scan")
    parser.add_argument("--status-topic", default="/plant_row/status")
    parser.add_argument("--row-width", type=float, default=0.60)
    parser.add_argument("--auto-speed", type=float, default=0.12)
    parser.add_argument("--sensor-yaw-deg", type=float, default=180.0)
    parser.add_argument("--front-sensor-yaw-deg", type=float, default=180.0)
    parser.add_argument("--rear-sensor-yaw-deg", type=float, default=180.0)
    parser.add_argument("--lidar-yaw-correction-deg", type=float, default=0.0)
    parser.add_argument("--lidar-x-offset-m", type=float, default=0.0)
    parser.add_argument("--lidar-y-offset-m", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    return PlantLidarGUI(parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(main())
