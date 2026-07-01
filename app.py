#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import queue
import re
import shutil
import signal
import socket
import subprocess
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

import cv2
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage as RosCompressedImage
from sensor_msgs.msg import Image as RosImage
from sensor_msgs.msg import PointCloud2 as RosPointCloud2
from sensor_msgs_py import point_cloud2
from nav_msgs.msg import Odometry as RosOdometry
from visualization_msgs.msg import MarkerArray

PROJECT_ROOT = Path(__file__).resolve().parent
ROOT = PROJECT_ROOT.parent
CONTROL_ROOT = ROOT / "control"
MAPDATA_DIR = PROJECT_ROOT / "maps"
MISSIONS_DIR = PROJECT_ROOT / "missions"
MAIN_SCRIPT = PROJECT_ROOT / "main.py"
UVC_PREVIEW_SCRIPT = PROJECT_ROOT / "uvc_preview_publisher.py"
ROS_PYTHON = "/usr/bin/python3"
ROS_MONITOR_NODE_NAME = "autorun_final_visual_monitor"
UVC_PREVIEW_TOPIC = "/autorun_final/uvc_preview/compressed"
DEFAULT_SENSOR_HEIGHT_M = "1.2"
DEFAULT_BODY_X_OFFSET_M = "0.0"
DEFAULT_BODY_Y_OFFSET_M = "0.0"
DEFAULT_ROLL_GAIN = "0.65"
DEFAULT_PITCH_GAIN = "1.0"
DEFAULT_LIDAR_YAW_CORRECTION_DEG = "-3.0"
DEFAULT_LIDAR_X_OFFSET_M = "0.035"
DEFAULT_LIDAR_Y_OFFSET_M = "0.0"
SETTINGS_PATH = PROJECT_ROOT / "gui_settings.json"


def now_text() -> str:
    return datetime.now().strftime("%H:%M:%S")


def generated_name(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def missions_for_map(map_path: Path | None) -> list[Path]:
    if map_path is None:
        return []
    local = sorted(map_path.parent.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if local:
        return local
    legacy: list[Path] = []
    for mission_path in sorted(MISSIONS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(mission_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        bound_map_db = str(payload.get("bound_map_db") or "").strip()
        if not bound_map_db:
            continue
        try:
            bound_tokens = {
                Path(bound_map_db).name.lower(),
                Path(bound_map_db).stem.lower(),
                Path(bound_map_db).parent.name.lower(),
            }
            map_tokens = {
                map_path.name.lower(),
                map_path.stem.lower(),
                map_path.parent.name.lower(),
            }
            if bound_tokens & map_tokens:
                legacy.append(mission_path)
        except Exception:
            continue
    return legacy


def normalize_device_path(source: str) -> str:
    text = str(source).strip()
    if text.isdigit():
        return f"/dev/video{text}"
    return text


def list_v4l2_devices() -> list[str]:
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        return ["/dev/video0"]

    devices: list[str] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("/dev/video"):
            devices.append(stripped)
    return devices or ["/dev/video0"]


def query_v4l2_capabilities(device: str) -> dict[str, dict[str, list[str]]]:
    result = subprocess.run(
        ["v4l2-ctl", "-d", device, "--list-formats-ext"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    capabilities: dict[str, dict[str, list[str]]] = {}
    current_format = ""
    current_size = ""
    for line in result.stdout.splitlines():
        fmt_match = re.search(r"\[\d+\]:\s+'([^']+)'", line)
        if fmt_match:
            current_format = fmt_match.group(1)
            capabilities.setdefault(current_format, {})
            continue
        size_match = re.search(r"Size:\s+Discrete\s+(\d+x\d+)", line)
        if size_match and current_format:
            current_size = size_match.group(1)
            capabilities[current_format].setdefault(current_size, [])
            continue
        fps_match = re.search(r"\(([\d.]+)\s+fps\)", line)
        if fps_match and current_format and current_size:
            fps_text = fps_match.group(1)
            fps_list = capabilities[current_format][current_size]
            if fps_text not in fps_list:
                fps_list.append(fps_text)
    if not capabilities:
        raise RuntimeError(f"No V4L2 capabilities found for {device}")
    return capabilities


def sort_resolution_text(values: list[str]) -> list[str]:
    def key(item: str) -> tuple[int, int]:
        match = re.match(r"^\s*(\d+)\s*x\s*(\d+)\s*$", item)
        if not match:
            return (0, 0)
        return (int(match.group(1)), int(match.group(2)))

    return sorted(values, key=key)


def sort_fps_text(values: list[str]) -> list[str]:
    def key(item: str) -> float:
        try:
            return float(item)
        except Exception:
            return 0.0

    return sorted(values, key=key)


def ros_master_is_ready(timeout_sec: float = 0.2) -> bool:
    del timeout_sec
    return True


class Ros2NodeThread:
    _init_lock = threading.Lock()
    _refcount = 0

    def __init__(self, node_name: str) -> None:
        os.environ.setdefault("ROS_HOME", "/tmp/autorun_final_ros2_home")
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


_ROS_MONITOR_THREAD: Ros2NodeThread | None = None


def ensure_ros_monitor_node() -> Ros2NodeThread:
    global _ROS_MONITOR_THREAD
    if _ROS_MONITOR_THREAD is None:
        _ROS_MONITOR_THREAD = Ros2NodeThread(ROS_MONITOR_NODE_NAME)
    return _ROS_MONITOR_THREAD


def quat_to_rpy(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float, float]:
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def project_ground_pose(
    raw_pose: dict[str, float],
    *,
    sensor_height_m: float,
    body_x_offset_m: float,
    body_y_offset_m: float,
    roll_gain: float,
    pitch_gain: float,
    anchor_roll_rad: float,
    anchor_pitch_rad: float,
) -> dict[str, float]:
    roll = raw_pose["roll"]
    pitch = raw_pose["pitch"]
    yaw = raw_pose["yaw"]
    rel_roll = (roll - anchor_roll_rad) * roll_gain
    rel_pitch = (pitch - anchor_pitch_rad) * pitch_gain
    ground_forward = body_x_offset_m - sensor_height_m * math.sin(rel_pitch)
    ground_left = body_y_offset_m + sensor_height_m * math.sin(rel_roll)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    dx = cy * ground_forward - sy * ground_left
    dy = sy * ground_forward + cy * ground_left
    dz = -sensor_height_m * math.cos(rel_roll) * math.cos(rel_pitch)
    return {
        "x": raw_pose["x"] + dx,
        "y": raw_pose["y"] + dy,
        "z": raw_pose["z"] + dz,
        "yaw": yaw,
        "roll": roll,
        "pitch": pitch,
    }


class ProcessWorker:
    _next_id = 1

    def __init__(self, cmd: list[str], cwd: Path, label: str, sink: queue.Queue[tuple[str, Any]]) -> None:
        self.cmd = cmd
        self.cwd = cwd
        self.label = label
        self.sink = sink
        self.worker_id = ProcessWorker._next_id
        ProcessWorker._next_id += 1
        self.process: subprocess.Popen[str] | None = None
        self.stop_requested = False
        self.stop_signal_sent_at = 0.0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        self.sink.put(("log", f"{now_text()} Starting command: {' '.join(self.cmd)}"))
        self.process = subprocess.Popen(
            self.cmd,
            cwd=str(self.cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid,
            env=env,
        )
        assert self.process.stdout is not None
        for raw_line in self.process.stdout:
            line = raw_line.rstrip()
            if line:
                self.sink.put(("log", line))
        code = self.process.wait()
        self.sink.put(("task_finished", {"code": code, "stopped": self.stop_requested, "label": self.label, "worker_id": self.worker_id}))

    def stop(self) -> None:
        self.stop_requested = True
        if self.process is None or self.process.poll() is not None:
            return
        try:
            if self.stop_signal_sent_at <= 0.0:
                self.process.send_signal(signal.SIGINT)
                self.stop_signal_sent_at = time.monotonic()
            elif (time.monotonic() - self.stop_signal_sent_at) > 8.0:
                os.killpg(self.process.pid, signal.SIGTERM)
        except Exception:
            pass


class StatusMonitor:
    def __init__(self, sink: queue.Queue[tuple[str, Any]]) -> None:
        self.sink = sink
        self.process: subprocess.Popen[str] | None = None
        self.running = False
        self.first_snapshot_logged = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.running = True
        self.thread.start()

    def _ensure_can0_up(self) -> bool:
        try:
            state = Path("/sys/class/net/can0/operstate").read_text(encoding="utf-8").strip().lower()
        except Exception:
            self.sink.put(("log", f"{now_text()} can0 was not found on this system."))
            return False
        if state in {"up", "unknown"}:
            self.sink.put(("log", f"{now_text()} can0 is already up."))
            return True
        self.sink.put(("log", f"{now_text()} can0 is {state}. Bringing it up at 500000 bitrate..."))
        cmds = [
            ["sudo", "ip", "link", "set", "can0", "down"],
            ["sudo", "ip", "link", "set", "can0", "type", "can", "bitrate", "500000"],
            ["sudo", "ip", "link", "set", "can0", "up"],
        ]
        for cmd in cmds:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
            if result.returncode != 0:
                output = (result.stdout or "").strip()
                self.sink.put(("log", f"{now_text()} Failed to run: {' '.join(cmd)}"))
                if output:
                    self.sink.put(("log", f"{now_text()} {output}"))
                return False
        self.sink.put(("log", f"{now_text()} can0 is now up."))
        return True

    def _run(self) -> None:
        self._ensure_can0_up()
        self.process = subprocess.Popen(
            [
                "sudo",
                "python3",
                str(CONTROL_ROOT / "fw_mini_status_reader.py"),
                "--interface",
                "socketcan",
                "--channel",
                "can0",
                "--bitrate",
                "500000",
                "--refresh",
                "1",
                "--json",
            ],
            cwd=str(CONTROL_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid,
        )
        assert self.process.stdout is not None
        saw_snapshot = False
        for raw_line in self.process.stdout:
            if not self.running:
                break
            line = raw_line.strip()
            if not line:
                continue
            try:
                snapshot = json.loads(line)
            except json.JSONDecodeError:
                self.sink.put(("log", f"{now_text()} {line}"))
                continue
            saw_snapshot = True
            if not self.first_snapshot_logged:
                self.sink.put(("log", f"{now_text()} Vehicle feedback is streaming on can0."))
                self.first_snapshot_logged = True
            self.sink.put(("snapshot", snapshot))
        if self.running and not saw_snapshot:
            self.sink.put(("log", f"{now_text()} No CAN feedback was received on can0."))
        self.sink.put(("status_finished", None))

    def stop(self) -> None:
        self.running = False
        if self.process is not None and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGINT)
            except Exception:
                pass


class RosImageMonitor:
    def __init__(
        self,
        sink: queue.Queue[tuple[str, Any]],
        topics: tuple[str, ...] = ("/odin1/image/compressed", "/odin1/image/undistorted", "/odin1/image"),
    ) -> None:
        self.sink = sink
        self.topics = topics
        self.running = False
        self.subscriptions: list[Any] = []
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.last_emit = 0.0
        self.has_frame = False
        self.started_at = 0.0
        self.min_emit_interval = 0.12

    def start(self) -> None:
        self.running = True
        self.started_at = time.monotonic()
        self.thread.start()

    def stop(self) -> None:
        self.running = False

    def reset(self) -> None:
        self.last_emit = 0.0
        self.has_frame = False
        self.started_at = time.monotonic()

    def _run(self) -> None:
        try:
            ros_thread = ensure_ros_monitor_node()
            for topic in self.topics:
                if topic.endswith("/compressed"):
                    self.subscriptions.append(ros_thread.node.create_subscription(
                        RosCompressedImage,
                        topic,
                        lambda msg, t=topic: self._on_compressed_image(msg, t),
                        1,
                    ))
                else:
                    self.subscriptions.append(ros_thread.node.create_subscription(
                        RosImage,
                        topic,
                        lambda msg, t=topic: self._on_image(msg, t),
                        1,
                    ))
            self.sink.put(("log", f"{now_text()} Camera monitor subscribed to {', '.join(self.topics)}."))
            self.sink.put(("camera_status", "Subscribed"))
            while self.running:
                time.sleep(0.1)
        except Exception as exc:
            self.sink.put(("log", f"{now_text()} Camera monitor failed: {exc}"))
            self.sink.put(("camera_status", "Failed"))
        finally:
            self.sink.put(("camera_status", "Stopped"))

    def _prepare_frame(self, frame: np.ndarray, topic: str) -> None:
        max_w = 512
        max_h = 288
        scale = min(max_w / max(frame.shape[1], 1), max_h / max(frame.shape[0], 1), 1.0)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        h, w, _ = frame.shape
        ppm = b"P6\n%d %d\n255\n" % (w, h) + frame.tobytes()
        if not self.has_frame:
            self.has_frame = True
            self.sink.put(("log", f"{now_text()} Camera frames are now arriving from {topic}."))
            self.sink.put(("camera_status", "Streaming"))
        self.sink.put(("camera_frame", {"image": ppm, "topic": topic}))

    def _on_compressed_image(self, msg: RosCompressedImage, topic: str) -> None:
        if not self.running:
            return
        now = time.monotonic()
        if now - self.last_emit < self.min_emit_interval:
            return
        self.last_emit = now
        try:
            encoded = np.frombuffer(msg.data, dtype=np.uint8)
            frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError("cv2.imdecode returned None")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._prepare_frame(frame, topic)
        except Exception as exc:
            self.sink.put(("log", f"{now_text()} Camera frame decode failed: {exc}"))

    def _on_image(self, msg: RosImage, topic: str) -> None:
        if not self.running:
            return
        now = time.monotonic()
        if now - self.last_emit < self.min_emit_interval:
            return
        self.last_emit = now
        try:
            if msg.encoding in {"bgr8", "8UC3"}:
                frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif msg.encoding == "rgb8":
                frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            elif msg.encoding in {"mono8", "8UC1"}:
                mono = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
                frame = cv2.cvtColor(mono, cv2.COLOR_GRAY2RGB)
            else:
                self.sink.put(("log", f"{now_text()} Unsupported image encoding: {msg.encoding}"))
                return
            self._prepare_frame(frame, topic)
        except Exception as exc:
            self.sink.put(("log", f"{now_text()} Camera frame decode failed: {exc}"))


class RosSceneMonitor:
    def __init__(self, sink: queue.Queue[tuple[str, Any]]) -> None:
        self.sink = sink
        self.running = False
        self.subscriptions: list[Any] = []
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.points: list[tuple[float, float, float]] = []
        self.path_points: list[tuple[float, float, float]] = []
        self.last_emit = 0.0
        self.cloud_topic = "/odin1/cloud_slam"

    def start(self) -> None:
        self.running = True
        self.thread.start()

    def stop(self) -> None:
        self.running = False

    def reset(self) -> None:
        self.points = []
        self.path_points = []
        self.last_emit = 0.0

    def _run(self) -> None:
        try:
            ros_thread = ensure_ros_monitor_node()
            for topic in ("/odin1/cloud_render", "/odin1/cloud_slam"):
                self.subscriptions.append(ros_thread.node.create_subscription(
                    RosPointCloud2,
                    topic,
                    lambda msg, t=topic: self._on_cloud(msg, t),
                    1,
                ))
            self.subscriptions.append(ros_thread.node.create_subscription(MarkerArray, "/odin1/path", self._on_path, 1))
            self.sink.put(("log", f"{now_text()} Scene monitor subscribed to /odin1/cloud_render, /odin1/cloud_slam and /odin1/path."))
            self.sink.put(("scene_status", "Streaming"))
            while self.running:
                now = time.monotonic()
                if now - self.last_emit >= 1.4:
                    self.last_emit = now
                    self.sink.put(("scene_frame", {"points": self.points, "path": self.path_points}))
                time.sleep(0.1)
        except Exception as exc:
            self.sink.put(("log", f"{now_text()} Scene monitor failed: {exc}"))
            self.sink.put(("scene_status", "Failed"))
        finally:
            self.sink.put(("scene_status", "Stopped"))

    def _on_cloud(self, msg: RosPointCloud2, topic: str) -> None:
        if not self.running:
            return
        self.cloud_topic = topic
        pts: list[tuple[float, float, float]] = []
        try:
            for idx, p in enumerate(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)):
                if idx % 140 != 0:
                    continue
                x, y, z = float(p[0]), float(p[1]), float(p[2])
                pts.append((x, y, z))
                if len(pts) >= 260:
                    break
            self.points = pts
        except Exception:
            return

    def _on_path(self, msg: MarkerArray) -> None:
        if not self.running:
            return
        path: list[tuple[float, float, float]] = []
        try:
            for marker in msg.markers:
                for idx, pt in enumerate(marker.points):
                    if idx % 4 != 0:
                        continue
                    path.append((float(pt.x), float(pt.y), float(pt.z)))
            self.path_points = path[-220:]
        except Exception:
            return


class RosPoseDebugMonitor:
    def __init__(self, sink: queue.Queue[tuple[str, Any]]) -> None:
        self.sink = sink
        self.running = False
        self.subscriptions: list[Any] = []
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.last_emit = 0.0

    def start(self) -> None:
        self.running = True
        self.thread.start()

    def stop(self) -> None:
        self.running = False

    def _run(self) -> None:
        try:
            ros_thread = ensure_ros_monitor_node()
            self.subscriptions.append(
                ros_thread.node.create_subscription(RosOdometry, "/odin1/odometry_highfreq", self._on_odom, 1)
            )
            self.sink.put(("log", f"{now_text()} Pose debug monitor subscribed to /odin1/odometry_highfreq."))
            while self.running:
                time.sleep(0.1)
        except Exception as exc:
            self.sink.put(("log", f"{now_text()} Pose debug monitor failed: {exc}"))

    def _on_odom(self, msg: RosOdometry) -> None:
        if not self.running:
            return
        now = time.monotonic()
        if now - self.last_emit < 0.2:
            return
        self.last_emit = now
        pose = msg.pose.pose
        roll, pitch, yaw = quat_to_rpy(
            float(pose.orientation.x),
            float(pose.orientation.y),
            float(pose.orientation.z),
            float(pose.orientation.w),
        )
        self.sink.put(("pose_debug", {
            "x": float(pose.position.x),
            "y": float(pose.position.y),
            "z": float(pose.position.z),
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
        }))

class RoutePreview(ttk.Frame):
    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self.canvas = tk.Canvas(self, bg="#0b1220", height=220, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.points: list[tuple[float, float]] = []
        self.canvas.bind("<Configure>", lambda _event: self.redraw())

    def set_points(self, points: list[tuple[float, float]]) -> None:
        self.points = points
        self.redraw()

    def redraw(self) -> None:
        self.canvas.delete("all")
        width = max(self.canvas.winfo_width(), 1)
        height = max(self.canvas.winfo_height(), 1)
        if len(self.points) < 2:
            self.canvas.create_text(width / 2, height / 2, text="No route preview available", fill="#94a3b8", font=("TkDefaultFont", 14))
            return

        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x = max(max_x - min_x, 1e-6)
        span_y = max(max_y - min_y, 1e-6)
        pad = 28
        draw_w = max(width - pad * 2, 1)
        draw_h = max(height - pad * 2, 1)
        scale = min(draw_w / span_x, draw_h / span_y)

        for i in range(5):
            y = pad + i * max(draw_h // 4, 1)
            self.canvas.create_line(pad, y, width - pad, y, fill="#1e293b")

        mapped: list[tuple[float, float]] = []
        for x, y in self.points:
            px = pad + (x - min_x) * scale
            py = height - pad - (y - min_y) * scale
            mapped.append((px, py))

        for a, b in zip(mapped, mapped[1:]):
            self.canvas.create_line(a[0], a[1], b[0], b[1], fill="#38bdf8", width=3)

        sx, sy = mapped[0]
        ex, ey = mapped[-1]
        self.canvas.create_oval(sx - 5, sy - 5, sx + 5, sy + 5, fill="#22c55e", outline="")
        self.canvas.create_oval(ex - 5, ey - 5, ex + 5, ey + 5, fill="#ef4444", outline="")


class ScenePreview(ttk.Frame):
    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self.canvas = tk.Canvas(self, bg="#101418", height=380, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.points: list[tuple[float, float, float]] = []
        self.path_points: list[tuple[float, float, float]] = []
        self.canvas.bind("<Configure>", lambda _event: self.redraw())

    def set_scene(self, points: list[tuple[float, float, float]], path_points: list[tuple[float, float, float]]) -> None:
        self.points = points
        self.path_points = path_points
        self.redraw()

    def redraw(self) -> None:
        self.canvas.delete("all")
        width = max(self.canvas.winfo_width(), 1)
        height = max(self.canvas.winfo_height(), 1)
        if not self.points and not self.path_points:
            self.canvas.create_text(width / 2, height / 2, text="No 3D map data yet", fill="#94a3b8", font=("TkDefaultFont", 14))
            return
        pts = self.points + self.path_points
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        zs = [p[2] for p in pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        min_z, max_z = min(zs), max(zs)
        span_x = max(max_x - min_x, 1e-6)
        span_y = max(max_y - min_y, 1e-6)
        span_z = max(max_z - min_z, 1e-6)
        pad = 30
        scale = min((width - pad * 2) / max(span_x + span_y * 0.5, 1e-6), (height - pad * 2) / max(span_z + span_y * 0.35, 1e-6))

        def project(p: tuple[float, float, float]) -> tuple[float, float]:
            x, y, z = p
            px = pad + (x - min_x) * scale + (y - min_y) * scale * 0.45
            py = height - pad - (z - min_z) * scale - (y - min_y) * scale * 0.18
            return px, py

        for x, y, z in self.points:
            px, py = project((x, y, z))
            depth = (z - min_z) / span_z if span_z > 1e-6 else 0.5
            color = "#6ee7ff" if depth > 0.5 else "#3b82f6"
            self.canvas.create_rectangle(px, py, px + 1, py + 1, outline=color, fill=color)

        mapped_path = [project(p) for p in self.path_points]
        for a, b in zip(mapped_path, mapped_path[1:]):
            self.canvas.create_line(a[0], a[1], b[0], b[1], fill="#f59e0b", width=2)
        if mapped_path:
            sx, sy = mapped_path[0]
            ex, ey = mapped_path[-1]
            self.canvas.create_oval(sx - 4, sy - 4, sx + 4, sy + 4, fill="#22c55e", outline="")
            self.canvas.create_oval(ex - 4, ey - 4, ex + 4, ey + 4, fill="#ef4444", outline="")


class ScrollableFrame(ttk.Frame):
    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self.canvas = tk.Canvas(self, highlightthickness=0, bg="#f4f7fb")
        self.v_scroll = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self._window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.v_scroll.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)

    def _on_inner_configure(self, _event: object) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self._window_id, width=event.width)

    def _on_mousewheel(self, event: tk.Event) -> None:
        if event.delta:
            self.canvas.yview_scroll(int(-event.delta / 120), "units")
        elif getattr(event, "num", None) == 4:
            self.canvas.yview_scroll(-3, "units")
        elif getattr(event, "num", None) == 5:
            self.canvas.yview_scroll(3, "units")

    def _bind_mousewheel(self, _event: object) -> None:
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _unbind_mousewheel(self, _event: object) -> None:
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")


class MainWindow:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("autorun_final Control Console")
        self.root.geometry("1480x980")
        self.root.minsize(1080, 720)
        self.root.resizable(True, True)
        self.root.configure(bg="#f4f7fb")

        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.task_worker: ProcessWorker | None = None
        self.preview_worker: ProcessWorker | None = None
        self.record_localization_worker: ProcessWorker | None = None
        self.replay_localization_worker: ProcessWorker | None = None
        self.status_monitor: StatusMonitor | None = None
        self.camera_monitor: RosImageMonitor | None = None
        self.scene_monitor: RosSceneMonitor | None = None
        self.pose_debug_monitor: RosPoseDebugMonitor | None = None
        self.localization_map_path: Path | None = None
        self.pending_action: str | None = None
        self.closing = False
        self.pending_camera_frame: bytes | None = None
        self.pending_scene_frame: dict[str, Any] | None = None
        self.max_console_lines = 350
        self.last_camera_render_at = 0.0
        self.camera_restart_backoff_until = 0.0

        self.status_text = tk.StringVar(value="Idle")
        self.last_log_text = tk.StringVar(value="Ready.")
        self.record_localization_text = tk.StringVar(value="Not started")
        self.replay_localization_text = tk.StringVar(value="Not started")
        self.camera_status_text = tk.StringVar(value="Stopped")
        self.scene_status_text = tk.StringVar(value="Stopped")
        self.preview_source_text = tk.StringVar(value="Waiting for preview stream")
        self.console_autoscroll_var = tk.BooleanVar(value=True)
        self.map_combo_mapping: ttk.Combobox
        self.map_combo_record: ttk.Combobox
        self.map_combo_replay: ttk.Combobox
        self.mission_combo: ttk.Combobox
        self.camera_label: ttk.Label
        self.camera_photo: tk.PhotoImage | None = None
        self.scene_preview: ScenePreview
        self.map_paths: dict[str, Path] = {}
        self.mission_paths: dict[str, Path] = {}
        self.mapping_name_var = tk.StringVar(value=generated_name("map"))
        self.mapping_recorddata_var = tk.BooleanVar(value=False)
        self.record_name_var = tk.StringVar(value=generated_name("mission"))
        self.line_cruise_vx_var = tk.StringVar(value="0.20")
        self.line_target_center_offset_px_var = tk.StringVar(value="0")
        self.line_vehicle_direction_angle_deg_var = tk.StringVar(value="0.0")
        self.line_steer_sign_var = tk.StringVar(value="-1.0")
        self.line_kp_offset_var = tk.StringVar(value="7.0")
        self.line_kp_heading_var = tk.StringVar(value="0.08")
        self.line_max_wz_var = tk.StringVar(value="1.6")
        self.local_weight_in_row_var = tk.StringVar(value="0.5")
        self.global_weight_in_row_var = tk.StringVar(value="0.5")
        self.sensor_height_var = tk.StringVar(value=DEFAULT_SENSOR_HEIGHT_M)
        self.body_x_offset_var = tk.StringVar(value=DEFAULT_BODY_X_OFFSET_M)
        self.body_y_offset_var = tk.StringVar(value=DEFAULT_BODY_Y_OFFSET_M)
        self.roll_gain_var = tk.StringVar(value=DEFAULT_ROLL_GAIN)
        self.pitch_gain_var = tk.StringVar(value=DEFAULT_PITCH_GAIN)
        self.pose_vars = {key: tk.StringVar(value="--") for key in [
            "anchor", "raw_xy", "raw_rp", "proj_xy", "proj_delta", "proj_rpy"
        ]}
        self.anchor_pose_debug: dict[str, float] | None = None
        self.last_raw_pose_debug: dict[str, float] | None = None
        self.vehicle_vars = {key: tk.StringVar(value="--") for key in [
            "gear", "vx", "vy", "wz", "soc", "voltage", "current",
            "capacity", "charging", "estop", "remote", "unlock_ok",
            "charge_dock", "manual_charger", "light_mode",
            "error", "component", "code", "updated_at"
        ]}

        self._load_settings()
        self._build_ui()
        self._refresh_maps()
        self._refresh_missions()
        self._log("GUI is ready.")
        self._start_status_monitor(auto=True)
        self.root.after(150, self._pump_events)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _load_settings(self) -> None:
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(data, dict):
            return
        self.sensor_height_var.set(str(data.get("sensor_height_m", DEFAULT_SENSOR_HEIGHT_M)))
        self.body_x_offset_var.set(str(data.get("body_x_offset_m", DEFAULT_BODY_X_OFFSET_M)))
        self.body_y_offset_var.set(str(data.get("body_y_offset_m", DEFAULT_BODY_Y_OFFSET_M)))
        self.roll_gain_var.set(str(data.get("roll_gain", DEFAULT_ROLL_GAIN)))
        self.pitch_gain_var.set(str(data.get("pitch_gain", DEFAULT_PITCH_GAIN)))
        self.line_cruise_vx_var.set(str(data.get("line_cruise_vx", self.line_cruise_vx_var.get())))
        self.line_target_center_offset_px_var.set(str(data.get("line_target_center_offset_px", self.line_target_center_offset_px_var.get())))
        self.line_vehicle_direction_angle_deg_var.set(str(data.get("line_vehicle_direction_angle_deg", self.line_vehicle_direction_angle_deg_var.get())))
        self.line_steer_sign_var.set(str(data.get("line_steer_sign", self.line_steer_sign_var.get())))
        self.line_kp_offset_var.set(str(data.get("line_kp_offset", self.line_kp_offset_var.get())))
        self.line_kp_heading_var.set(str(data.get("line_kp_heading", self.line_kp_heading_var.get())))
        self.line_max_wz_var.set(str(data.get("line_max_wz", self.line_max_wz_var.get())))
        self.local_weight_in_row_var.set(str(data.get("local_weight_in_row", self.local_weight_in_row_var.get())))
        self.global_weight_in_row_var.set(str(data.get("global_weight_in_row", self.global_weight_in_row_var.get())))
        self.mapping_recorddata_var.set(bool(data.get("mapping_recorddata", self.mapping_recorddata_var.get())))

    def _save_settings(self) -> None:
        data = {
            "sensor_height_m": self.sensor_height_var.get().strip() or DEFAULT_SENSOR_HEIGHT_M,
            "body_x_offset_m": self.body_x_offset_var.get().strip() or DEFAULT_BODY_X_OFFSET_M,
            "body_y_offset_m": self.body_y_offset_var.get().strip() or DEFAULT_BODY_Y_OFFSET_M,
            "roll_gain": self.roll_gain_var.get().strip() or DEFAULT_ROLL_GAIN,
            "pitch_gain": self.pitch_gain_var.get().strip() or DEFAULT_PITCH_GAIN,
            "line_cruise_vx": self.line_cruise_vx_var.get().strip() or "0.20",
            "line_target_center_offset_px": self.line_target_center_offset_px_var.get().strip() or "0",
            "line_vehicle_direction_angle_deg": self.line_vehicle_direction_angle_deg_var.get().strip() or "0.0",
            "line_steer_sign": self.line_steer_sign_var.get().strip() or "-1.0",
            "line_kp_offset": self.line_kp_offset_var.get().strip() or "7.0",
            "line_kp_heading": self.line_kp_heading_var.get().strip() or "0.08",
            "line_max_wz": self.line_max_wz_var.get().strip() or "1.6",
            "local_weight_in_row": self.local_weight_in_row_var.get().strip() or "0.5",
            "global_weight_in_row": self.global_weight_in_row_var.get().strip() or "0.5",
            "mapping_recorddata": bool(self.mapping_recorddata_var.get()),
        }
        try:
            SETTINGS_PATH.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")
        except Exception as exc:
            self._log(f"Warning: failed to save GUI settings: {exc}")

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook.Tab", padding=(12, 8))
        style.configure("Card.TFrame", background="#ffffff")

        outer = ttk.Frame(self.root, padding=18)
        outer.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(outer, style="Card.TFrame", padding=16)
        top.pack(fill=tk.X)
        title_box = ttk.Frame(top, style="Card.TFrame")
        title_box.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(title_box, text="autorunlida Control Console", font=("TkDefaultFont", 18, "bold")).pack(anchor="w")
        ttk.Label(title_box, text="Odin global localization + lidar local row guidance + chassis monitoring").pack(anchor="w")
        ttk.Label(top, text="Current Task:").pack(side=tk.LEFT, padx=(12, 6))
        ttk.Label(top, textvariable=self.status_text).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Button(top, text="Stop Current Task", command=self._stop_current_task).pack(side=tk.RIGHT)

        ttk.Label(outer, textvariable=self.last_log_text).pack(fill=tk.X, pady=(10, 0))

        self.body = ttk.Panedwindow(outer, orient=tk.VERTICAL)
        self.body.pack(fill=tk.BOTH, expand=True, pady=(14, 0))

        notebook_wrap = ttk.Frame(self.body)
        self.notebook = ttk.Notebook(notebook_wrap)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.notebook.add(self._build_mapping_tab(), text="Mapping")
        self.notebook.add(self._build_record_tab(), text="Path Recording")
        self.notebook.add(self._build_replay_tab(), text="Hybrid Drive")
        self.notebook.add(self._build_preview_tab(), text="Video Preview")
        self.notebook.add(self._build_status_tab(), text="Vehicle Status")
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        console_group = ttk.LabelFrame(self.body, text="Console", padding=10)
        console_toolbar = ttk.Frame(console_group)
        console_toolbar.pack(fill=tk.X, pady=(0, 8))
        ttk.Checkbutton(
            console_toolbar,
            text="Auto Scroll",
            variable=self.console_autoscroll_var,
        ).pack(side=tk.LEFT)
        ttk.Button(console_toolbar, text="Jump to Latest", command=self._scroll_console_to_latest).pack(side=tk.LEFT, padx=(8, 0))
        console_inner = ttk.Frame(console_group)
        console_inner.pack(fill=tk.BOTH, expand=True)
        self.console = tk.Text(console_inner, height=10, wrap="word", bg="#ffffff", fg="#132238")
        self.console.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(console_inner, orient=tk.VERTICAL, command=self.console.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.console.configure(yscrollcommand=scrollbar.set)

        self.body.add(notebook_wrap, weight=4)
        self.body.add(console_group, weight=1)

    def _build_mapping_tab(self) -> ttk.Frame:
        tab = ttk.Frame(self.notebook, padding=16)
        group = ttk.LabelFrame(tab, text="Create a New Map", padding=14)
        group.pack(fill=tk.X)

        ttk.Label(group, text="Map Name").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=8)
        ttk.Entry(group, textvariable=self.mapping_name_var, width=40).grid(row=0, column=1, sticky="ew", pady=8)

        ttk.Label(group, text="Available Maps").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=8)
        self.map_combo_mapping = ttk.Combobox(group, state="readonly", width=60)
        self.map_combo_mapping.grid(row=1, column=1, sticky="ew", pady=8)

        ttk.Checkbutton(
            group,
            text="Enable MindCloud recorddata during mapping",
            variable=self.mapping_recorddata_var,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(2, 4))

        btn_row = ttk.Frame(group)
        btn_row.grid(row=3, column=0, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Button(btn_row, text="Refresh Maps", command=self._refresh_maps).pack(side=tk.LEFT)
        ttk.Button(btn_row, text="Delete Selected Map", command=self._delete_selected_map).pack(side=tk.LEFT, padx=8)
        ttk.Button(btn_row, text="Start Mapping", command=self._start_mapping).pack(side=tk.LEFT)

        group.columnconfigure(1, weight=1)
        ttk.Label(
            tab,
            text="Odin map files are saved as .bin files. When MindCloud recorddata is enabled, mapping stop will save the map and then turn recorddata off.",
        ).pack(anchor="w", pady=(12, 0))
        return tab

    def _build_record_tab(self) -> ttk.Frame:
        tab = ScrollableFrame(self.notebook)
        body = ttk.Frame(tab.inner, padding=16)
        body.pack(fill=tk.BOTH, expand=True)

        group = ttk.LabelFrame(body, text="Record a Taught Path", padding=14)
        group.pack(fill=tk.X)

        ttk.Label(group, text="Map File").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=8)
        self.map_combo_record = ttk.Combobox(group, state="readonly", width=60)
        self.map_combo_record.grid(row=0, column=1, sticky="ew", pady=8)

        ttk.Label(group, text="Mission Name").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=8)
        ttk.Entry(group, textvariable=self.record_name_var, width=40).grid(row=1, column=1, sticky="ew", pady=8)

        ttk.Label(group, text="The system relocalizes first. After localization becomes stable, drive manually and the path will be recorded automatically.").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=8
        )

        ttk.Label(group, text="Recording Localization Status").grid(row=3, column=0, sticky="w", padx=(0, 10), pady=8)
        ttk.Label(group, textvariable=self.record_localization_text).grid(row=3, column=1, sticky="w", pady=8)

        projection = ttk.LabelFrame(body, text="Ground Projection", padding=14)
        projection.pack(fill=tk.X, pady=(14, 0))
        ttk.Label(projection, text="Sensor Height (m)").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=8)
        ttk.Entry(projection, textvariable=self.sensor_height_var, width=12).grid(row=0, column=1, sticky="w", pady=8)
        ttk.Label(projection, text="Body X Offset (m)").grid(row=0, column=2, sticky="w", padx=(18, 10), pady=8)
        ttk.Entry(projection, textvariable=self.body_x_offset_var, width=12).grid(row=0, column=3, sticky="w", pady=8)
        ttk.Label(projection, text="Body Y Offset (m)").grid(row=0, column=4, sticky="w", padx=(18, 10), pady=8)
        ttk.Entry(projection, textvariable=self.body_y_offset_var, width=12).grid(row=0, column=5, sticky="w", pady=8)
        ttk.Label(projection, text="Roll Gain").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=8)
        ttk.Entry(projection, textvariable=self.roll_gain_var, width=12).grid(row=1, column=1, sticky="w", pady=8)
        ttk.Label(projection, text="Pitch Gain").grid(row=1, column=2, sticky="w", padx=(18, 10), pady=8)
        ttk.Entry(projection, textvariable=self.pitch_gain_var, width=12).grid(row=1, column=3, sticky="w", pady=8)
        ttk.Label(projection, text="Use these values to project the high-mounted sensor pose onto the ground reference point.").grid(
            row=2, column=0, columnspan=6, sticky="w", pady=(4, 0)
        )

        btn_row = ttk.Frame(group)
        btn_row.grid(row=4, column=0, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Button(btn_row, text="Refresh Maps", command=self._refresh_maps).pack(side=tk.LEFT)
        ttk.Button(btn_row, text="Start Localization", command=self._start_record_localization).pack(side=tk.LEFT, padx=8)
        ttk.Button(btn_row, text="Stop Localization", command=self._stop_record_localization).pack(side=tk.LEFT)
        ttk.Button(btn_row, text="Start Recording", command=self._start_recording).pack(side=tk.LEFT, padx=8)
        group.columnconfigure(1, weight=1)
        return tab

    def _build_replay_tab(self) -> ttk.Frame:
        tab = ScrollableFrame(self.notebook)
        body = ttk.Frame(tab.inner, padding=16)
        body.pack(fill=tk.BOTH, expand=True)

        group = ttk.LabelFrame(body, text="Hybrid Drive", padding=14)
        group.pack(fill=tk.X)

        ttk.Label(group, text="Map File").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=8)
        self.map_combo_replay = ttk.Combobox(group, state="readonly", width=60)
        self.map_combo_replay.grid(row=0, column=1, sticky="ew", pady=8)
        self.map_combo_replay.bind("<<ComboboxSelected>>", lambda _event: self._refresh_missions())

        ttk.Label(group, text="Mission File").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=8)
        self.mission_combo = ttk.Combobox(group, state="readonly", width=60)
        self.mission_combo.grid(row=1, column=1, sticky="ew", pady=8)
        self.mission_combo.bind("<<ComboboxSelected>>", lambda _event: self._update_route_preview())

        ttk.Label(group, text="The system relocalizes first, starts lidar local guidance, then blends global mission tracking with local row guidance.").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=8
        )

        ttk.Label(group, text="Hybrid Localization Status").grid(row=3, column=0, sticky="w", padx=(0, 10), pady=8)
        ttk.Label(group, textvariable=self.replay_localization_text).grid(row=3, column=1, sticky="w", pady=8)

        line_group = ttk.LabelFrame(body, text="Lidar Row Parameters", padding=14)
        line_group.pack(fill=tk.X, pady=(14, 0))
        ttk.Label(line_group, text="Row Cruise VX").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=8)
        ttk.Entry(line_group, textvariable=self.line_cruise_vx_var, width=12).grid(row=0, column=1, sticky="w", pady=8)
        ttk.Label(line_group, text="Local Weight").grid(row=0, column=2, sticky="w", padx=(18, 10), pady=8)
        ttk.Entry(line_group, textvariable=self.local_weight_in_row_var, width=12).grid(row=0, column=3, sticky="w", pady=8)
        ttk.Label(line_group, text="Global Weight").grid(row=0, column=4, sticky="w", padx=(18, 10), pady=8)
        ttk.Entry(line_group, textvariable=self.global_weight_in_row_var, width=12).grid(row=0, column=5, sticky="w", pady=8)
        ttk.Label(line_group, text="Line Offset px").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=8)
        ttk.Entry(line_group, textvariable=self.line_target_center_offset_px_var, width=12).grid(row=1, column=1, sticky="w", pady=8)
        ttk.Label(line_group, text="Direction Angle").grid(row=1, column=2, sticky="w", padx=(18, 10), pady=8)
        ttk.Entry(line_group, textvariable=self.line_vehicle_direction_angle_deg_var, width=12).grid(row=1, column=3, sticky="w", pady=8)
        ttk.Label(line_group, text="Use this page for row-following hybrid driving: lidar constrains in-row motion, global mission tracking takes over during row switch / end-of-row / lost centerline.").grid(
            row=2, column=0, columnspan=6, sticky="w", pady=(8, 0)
        )
        line_group.columnconfigure(1, weight=1)

        projection = ttk.LabelFrame(body, text="Ground Projection", padding=14)
        projection.pack(fill=tk.X, pady=(14, 0))
        ttk.Label(projection, text="Sensor Height (m)").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=8)
        ttk.Entry(projection, textvariable=self.sensor_height_var, width=12).grid(row=0, column=1, sticky="w", pady=8)
        ttk.Label(projection, text="Body X Offset (m)").grid(row=0, column=2, sticky="w", padx=(18, 10), pady=8)
        ttk.Entry(projection, textvariable=self.body_x_offset_var, width=12).grid(row=0, column=3, sticky="w", pady=8)
        ttk.Label(projection, text="Body Y Offset (m)").grid(row=0, column=4, sticky="w", padx=(18, 10), pady=8)
        ttk.Entry(projection, textvariable=self.body_y_offset_var, width=12).grid(row=0, column=5, sticky="w", pady=8)
        ttk.Label(projection, text="Roll Gain").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=8)
        ttk.Entry(projection, textvariable=self.roll_gain_var, width=12).grid(row=1, column=1, sticky="w", pady=8)
        ttk.Label(projection, text="Pitch Gain").grid(row=1, column=2, sticky="w", padx=(18, 10), pady=8)
        ttk.Entry(projection, textvariable=self.pitch_gain_var, width=12).grid(row=1, column=3, sticky="w", pady=8)

        btn_row = ttk.Frame(group)
        btn_row.grid(row=4, column=0, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Button(btn_row, text="Refresh Maps", command=self._refresh_maps).pack(side=tk.LEFT)
        ttk.Button(btn_row, text="Refresh Missions", command=self._refresh_missions).pack(side=tk.LEFT, padx=8)
        ttk.Button(btn_row, text="Delete Selected Mission", command=self._delete_selected_mission).pack(side=tk.LEFT)
        ttk.Button(btn_row, text="Start Localization", command=self._start_replay_localization).pack(side=tk.LEFT, padx=8)
        ttk.Button(btn_row, text="Stop Localization", command=self._stop_replay_localization).pack(side=tk.LEFT)
        ttk.Button(btn_row, text="Start Hybrid Drive", command=self._start_replay).pack(side=tk.LEFT, padx=8)
        group.columnconfigure(1, weight=1)

        preview_group = ttk.LabelFrame(body, text="Route Preview", padding=8)
        preview_group.pack(fill=tk.BOTH, expand=True, pady=(14, 0))
        self.route_preview = RoutePreview(preview_group)
        self.route_preview.pack(fill=tk.BOTH, expand=True)

        return tab

    def _build_preview_tab(self) -> ttk.Frame:
        tab = ttk.Frame(self.notebook, padding=16)
        preview_group = ttk.LabelFrame(tab, text="UVC Preview", padding=10)
        preview_group.pack(fill=tk.BOTH, expand=True)

        info_row = ttk.Frame(preview_group)
        info_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(info_row, text="Preview Source:").pack(side=tk.LEFT)
        ttk.Label(info_row, textvariable=self.preview_source_text).pack(side=tk.LEFT, padx=(6, 18))

        self.camera_label = ttk.Label(preview_group, text="No UVC preview yet.", anchor="center")
        self.camera_label.pack(fill=tk.BOTH, expand=True)
        return tab

    def _projection_args(self) -> list[str]:
        return [
            "--sensor-height-m", self.sensor_height_var.get().strip() or DEFAULT_SENSOR_HEIGHT_M,
            "--body-x-offset-m", self.body_x_offset_var.get().strip() or DEFAULT_BODY_X_OFFSET_M,
            "--body-y-offset-m", self.body_y_offset_var.get().strip() or DEFAULT_BODY_Y_OFFSET_M,
            "--roll-gain", self.roll_gain_var.get().strip() or DEFAULT_ROLL_GAIN,
            "--pitch-gain", self.pitch_gain_var.get().strip() or DEFAULT_PITCH_GAIN,
        ]

    def _hybrid_args(self) -> list[str]:
        args = [
            *self._projection_args(),
            "autorun",
            "--line-cruise-vx", self.line_cruise_vx_var.get().strip() or "0.20",
            "--line-target-center-offset-px", self.line_target_center_offset_px_var.get().strip() or "0",
            "--line-vehicle-direction-angle-deg", self.line_vehicle_direction_angle_deg_var.get().strip() or "0.0",
            "--line-steer-sign", self.line_steer_sign_var.get().strip() or "-1.0",
            "--line-kp-offset", self.line_kp_offset_var.get().strip() or "7.0",
            "--line-kp-heading", self.line_kp_heading_var.get().strip() or "0.08",
            "--line-max-wz", self.line_max_wz_var.get().strip() or "1.6",
            "--local-weight-in-row", self.local_weight_in_row_var.get().strip() or "0.5",
            "--global-weight-in-row", self.global_weight_in_row_var.get().strip() or "0.5",
            "--lidar-yaw-correction-deg", DEFAULT_LIDAR_YAW_CORRECTION_DEG,
            "--lidar-x-offset-m", DEFAULT_LIDAR_X_OFFSET_M,
            "--lidar-y-offset-m", DEFAULT_LIDAR_Y_OFFSET_M,
        ]
        return args

    def _hybrid_local_guidance_enabled(self) -> bool:
        try:
            return float(self.local_weight_in_row_var.get().strip() or "0.5") > 1e-6
        except Exception:
            return True

    def _build_status_tab(self) -> ttk.Frame:
        tab = ttk.Frame(self.notebook, padding=16)
        control_group = ttk.LabelFrame(tab, text="Status Monitor", padding=14)
        control_group.pack(fill=tk.X)
        ttk.Label(control_group, text="Read live feedback from can0.").pack(anchor="w")
        ttk.Button(control_group, text="Start Status Monitor", command=self._start_status_monitor).pack(anchor="w", pady=(10, 0))

        status_group = ttk.LabelFrame(tab, text="Live Vehicle Status", padding=14)
        status_group.pack(fill=tk.BOTH, expand=True, pady=(14, 0))

        grid = ttk.Frame(status_group)
        grid.pack(fill=tk.BOTH, expand=True)

        sections = [
            ("Motion", [("Gear", "gear"), ("Linear X", "vx"), ("Linear Y", "vy"), ("Yaw Rate", "wz")]),
            ("Battery", [("Battery SOC", "soc"), ("Voltage", "voltage"), ("Current", "current"), ("Capacity", "capacity"), ("Charging", "charging")]),
            ("Control", [("E-Stop", "estop"), ("Remote Priority", "remote"), ("Unlock OK", "unlock_ok"), ("Charge Dock", "charge_dock"), ("Manual Charger", "manual_charger"), ("Light Mode", "light_mode")]),
            ("Fault and Time", [("Error", "error"), ("Component", "component"), ("Error Code", "code"), ("Last Update", "updated_at")]),
        ]

        for col, (title, items) in enumerate(sections):
            card = ttk.LabelFrame(grid, text=title, padding=12)
            card.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 10, 0))
            grid.columnconfigure(col, weight=1, uniform="status")
            for row, (label, key) in enumerate(items):
                ttk.Label(card, text=label).grid(row=row, column=0, sticky="w", padx=(0, 14), pady=7)
                value = ttk.Label(card, textvariable=self.vehicle_vars[key], anchor="e")
                value.grid(row=row, column=1, sticky="ew", pady=7)
            card.columnconfigure(1, weight=1)
        return tab

    def _append_console(self, text: str) -> None:
        if self.closing:
            return
        self.last_log_text.set(text)
        self.console.insert(tk.END, f"{text}\n")
        line_count = int(self.console.index("end-1c").split(".")[0])
        overflow = line_count - self.max_console_lines
        if overflow > 0:
            self.console.delete("1.0", f"{overflow + 1}.0")
        if self.console_autoscroll_var.get():
            self.console.see(tk.END)

    def _scroll_console_to_latest(self) -> None:
        self.console.see(tk.END)

    def _render_pending_frames(self) -> None:
        if self.pending_camera_frame is not None:
            try:
                self.camera_photo = tk.PhotoImage(data=self.pending_camera_frame, format="PPM")
                self.camera_label.configure(image=self.camera_photo, text="")
                self.last_camera_render_at = time.monotonic()
            except Exception as exc:
                self._log(f"Camera frame render failed: {exc}")
            finally:
                self.pending_camera_frame = None
        else:
            self.pending_camera_frame = None
        self.pending_scene_frame = None

    def _clear_visuals(self) -> None:
        self.pending_camera_frame = None
        self.pending_scene_frame = None
        if self.camera_monitor is not None:
            self.camera_monitor.reset()
        self.camera_photo = None
        self.camera_label.configure(image="", text="No UVC preview yet.")
        self.last_camera_render_at = 0.0
        self.camera_status_text.set("Waiting")
        self.preview_source_text.set("Waiting for preview stream")
        self.scene_status_text.set("Waiting")

    def _clear_camera_preview_only(self, text: str = "Waiting for preview stream") -> None:
        self.pending_camera_frame = None
        if self.camera_monitor is not None:
            self.camera_monitor.reset()
        self.camera_photo = None
        self.camera_label.configure(image="", text=text)
        self.last_camera_render_at = 0.0
        self.preview_source_text.set(text)

    def _visual_stream_allowed(self) -> bool:
        return self.task_worker is not None or self._active_localization_worker() is not None

    def _log(self, text: str) -> None:
        self._append_console(f"{now_text()} {text}")

    def _pump_events(self) -> None:
        if self.closing:
            return
        processed = 0
        while processed < 30:
            try:
                event, payload = self.events.get_nowait()
            except queue.Empty:
                break
            processed += 1
            if event == "log":
                line = str(payload)
                self._append_console(line)
                if "Localization succeeded." in line and self._active_localization_worker() is not None:
                    self._set_localization_status("Ready")
                    if self.pending_action == "record":
                        self.pending_action = None
                        self.root.after(50, self._start_recording)
                    elif self.pending_action == "replay":
                        self.pending_action = None
                        self.root.after(50, self._start_replay)
            elif event == "task_finished":
                self._on_task_finished(payload)
            elif event == "snapshot":
                self._apply_snapshot(payload)
            elif event == "camera_frame":
                if isinstance(payload, dict):
                    image = payload.get("image")
                    topic = str(payload.get("topic", ""))
                    self.pending_camera_frame = image if isinstance(image, bytes) else None
                    if topic == UVC_PREVIEW_TOPIC:
                        self.preview_source_text.set(f"Raw UVC Preview  ({UVC_PREVIEW_TOPIC})")
                    elif topic:
                        self.preview_source_text.set(topic)
                else:
                    self.pending_camera_frame = payload if isinstance(payload, bytes) else None
            elif event == "camera_status":
                self.camera_status_text.set(str(payload))
            elif event == "scene_frame":
                self.pending_scene_frame = payload if isinstance(payload, dict) else None
            elif event == "scene_status":
                self.scene_status_text.set(str(payload))
            elif event == "pose_debug":
                if isinstance(payload, dict):
                    self._apply_pose_debug(payload)
            elif event == "status_finished":
                self.status_monitor = None
        self._render_pending_frames()
        self._watch_visual_streams()
        if not self.closing:
            self.root.after(80, self._pump_events)

    def _watch_visual_streams(self) -> None:
        if not self._visual_stream_allowed():
            return
        task_label = self.task_worker.label if self.task_worker is not None else ""
        if task_label != "Hybrid Drive":
            return
        if self._hybrid_local_guidance_enabled():
            return
        source_text = self.preview_source_text.get()
        using_raw_uvc = (UVC_PREVIEW_TOPIC in source_text) or ("Waiting for preview stream" in source_text)
        if not using_raw_uvc:
            return
        now = time.monotonic()
        if self.last_camera_render_at > 0.0 and (now - self.last_camera_render_at) < 1.5:
            return
        if now < self.camera_restart_backoff_until:
            return
        self.camera_restart_backoff_until = now + 3.0
        self._log("Hybrid Drive preview appears stale. Restarting raw UVC preview publisher...")
        self._stop_uvc_preview_publisher(log_message=False)
        if self.camera_monitor is not None:
            self.camera_monitor.reset()
        self._start_uvc_preview_publisher(auto=True)

    def _selected_map_path(self, combo: ttk.Combobox) -> Path | None:
        label = combo.get().strip()
        return self.map_paths.get(label)

    def _selected_mission_path(self) -> Path | None:
        label = self.mission_combo.get().strip()
        return self.mission_paths.get(label)

    def _default_map_path(self) -> Path | None:
        if not self.map_paths:
            return None
        first_key = next(iter(self.map_paths))
        return self.map_paths.get(first_key)

    def _set_localization_status(self, text: str) -> None:
        self.record_localization_text.set(text)
        self.replay_localization_text.set(text)

    def _active_localization_worker(self) -> ProcessWorker | None:
        return self.replay_localization_worker or self.record_localization_worker

    def _start_shared_localization(self, map_path: Path, *, auto: bool = False) -> None:
        active = self._active_localization_worker()
        if active is not None and self.localization_map_path == map_path:
            return
        if active is not None:
            self._log("Switching localization map. Restarting the shared localization session.")
            active.stop()
            self.record_localization_worker = None
            self.replay_localization_worker = None
        self.localization_map_path = map_path
        self._log(
            f"{'Auto-starting' if auto else 'Starting'} shared localization with map: {map_path.name}"
        )
        worker = ProcessWorker(
            [ROS_PYTHON, str(MAIN_SCRIPT), "localization", "--db", str(map_path), "--base-frame", "odin1_base_link", "--localization-wait-sec", "60"],
            PROJECT_ROOT,
            "Shared Localization",
            self.events,
        )
        self.record_localization_worker = worker
        self.replay_localization_worker = worker
        self._set_localization_status("Starting...")
        self._stop_uvc_preview_publisher(log_message=False)
        self._start_camera_monitor(auto=True)
        self._clear_visuals()
        self._start_uvc_preview_publisher(auto=True)
        worker.start()

    def _stop_shared_localization(self) -> None:
        active = self._active_localization_worker()
        if active is None:
            self._log("Shared localization is not running.")
            return
        self._log("Stop requested for shared localization.")
        active.stop()

    def _ensure_localization_for_tab(self, prefer_map: Path | None = None, *, auto: bool = False) -> bool:
        if auto and self._active_localization_worker() is not None:
            return True
        map_path = prefer_map or self._selected_map_path(self.map_combo_record) or self._selected_map_path(self.map_combo_replay) or self._default_map_path()
        if map_path is None:
            if not auto:
                messagebox.showwarning("No Map", "No map is available for localization.")
            return False
        self._start_shared_localization(map_path, auto=auto)
        return True

    def _refresh_maps(self) -> None:
        maps = sorted(MAPDATA_DIR.rglob("*.bin"), key=lambda p: p.stat().st_mtime, reverse=True)
        self.map_paths = {f"{p.name}  |  {p.parent.name}": p for p in maps}
        values = list(self.map_paths.keys())
        for combo in (self.map_combo_mapping, self.map_combo_record, self.map_combo_replay):
            current = combo.get()
            combo["values"] = values
            if current in self.map_paths:
                combo.set(current)
            elif values:
                combo.set(values[0])
            else:
                combo.set("")
        self._refresh_missions()

    def _refresh_missions(self) -> None:
        selected_map = self._selected_map_path(self.map_combo_replay)
        filtered: dict[str, Path] = {}
        for mission_path in missions_for_map(selected_map):
            try:
                payload = json.loads(mission_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            bound_map_db = str(payload.get("bound_map_db") or "").strip()
            if selected_map is not None and bound_map_db:
                try:
                    bound_tokens = {
                        Path(bound_map_db).name.lower(),
                        Path(bound_map_db).stem.lower(),
                        Path(bound_map_db).parent.name.lower(),
                    }
                    map_tokens = {
                        selected_map.name.lower(),
                        selected_map.stem.lower(),
                        selected_map.parent.name.lower(),
                    }
                    if not (bound_tokens & map_tokens):
                        continue
                except Exception:
                    continue
            filtered[mission_path.stem] = mission_path
        self.mission_paths = filtered
        values = list(self.mission_paths.keys())
        current = self.mission_combo.get()
        self.mission_combo["values"] = values
        if current in self.mission_paths:
            self.mission_combo.set(current)
        elif values:
            self.mission_combo.set(values[0])
        else:
            self.mission_combo.set("")
        self._update_route_preview()

    def _update_route_preview(self) -> None:
        mission_path = self._selected_mission_path()
        if mission_path is None or not mission_path.exists():
            self.route_preview.set_points([])
            return
        try:
            payload = json.loads(mission_path.read_text(encoding="utf-8"))
            points = [
                (float(sample["pose"]["x"]), float(sample["pose"]["y"]))
                for sample in payload.get("samples", [])
                if "pose" in sample
            ]
        except Exception:
            points = []
        self.route_preview.set_points(points)

    def _set_task_status(self, text: str) -> None:
        self.status_text.set(text)

    def _start_task(self, label: str, args: list[str], *, slot: str = "task") -> None:
        if slot == "task" and self.task_worker is not None:
            messagebox.showwarning("Task Running", "Stop the current task before starting a new one.")
            return
        self._log(f"Starting {label}: {' '.join(args)}")
        worker = ProcessWorker([ROS_PYTHON, str(MAIN_SCRIPT), *args], PROJECT_ROOT, label, self.events)
        if slot == "record_localization":
            self.record_localization_worker = worker
            self._set_localization_status("Starting...")
        elif slot == "replay_localization":
            self.replay_localization_worker = worker
            self._set_localization_status("Starting...")
        else:
            self.task_worker = worker
            self._set_task_status(label)
        worker.start()

    def _on_task_finished(self, payload: dict[str, Any]) -> None:
        code = int(payload["code"])
        stopped = bool(payload["stopped"])
        label = str(payload["label"])
        worker_id = int(payload.get("worker_id", -1))
        if label == "UVC Preview":
            if self.preview_worker is not None and self.preview_worker.worker_id == worker_id:
                self.preview_worker = None
            if stopped:
                self._log(f"{label} stopped with exit code {code}")
            else:
                self._log(f"{label} finished with exit code {code}")
            return
        if label in {"Record Localization", "Replay Localization", "Shared Localization"}:
            active = self._active_localization_worker()
            if active is not None and active.worker_id == worker_id:
                self.record_localization_worker = None
                self.replay_localization_worker = None
                if stopped:
                    self._set_localization_status("Stopped")
                elif code == 0:
                    self._set_localization_status("Ready")
                else:
                    self._set_localization_status("Failed")
        else:
            if self.task_worker is not None and self.task_worker.worker_id == worker_id:
                self.task_worker = None
        if label == "Hybrid Drive" and code != 0 and self._hybrid_local_guidance_enabled():
            self._clear_camera_preview_only("Preview unavailable")
        if stopped:
            self._log(f"{label} stopped with exit code {code}")
        else:
            self._log(f"{label} finished with exit code {code}")
            if code != 0:
                messagebox.showerror(
                    "Task Failed",
                    f"{label} exited with code {code}.\nCheck the Console panel for details.",
                )
        if label not in {"Record Localization", "Replay Localization", "Shared Localization"} and self.task_worker is None:
            self._set_task_status("Idle")
        if (
            label in {"Mapping", "Path Recording", "Shared Localization"}
            and self.task_worker is None
            and self._active_localization_worker() is None
        ):
            self._stop_uvc_preview_publisher(log_message=False)
        if label in {"Mapping", "Path Recording", "Hybrid Drive", "Record Localization", "Replay Localization", "Shared Localization"} and (
            (label in {"Record Localization", "Replay Localization", "Shared Localization"} and self._active_localization_worker() is None)
            or (label not in {"Record Localization", "Replay Localization", "Shared Localization"})
        ):
            self._clear_visuals()
        self._refresh_maps()
        self._refresh_missions()
        if label == "Mapping":
            self.mapping_name_var.set(generated_name("map"))
        if label == "Path Recording":
            self.record_name_var.set(generated_name("mission"))

    def _start_mapping(self) -> None:
        map_name = self.mapping_name_var.get().strip() or generated_name("map")
        self.pending_action = None
        active = self._active_localization_worker()
        if active is not None:
            self._log("Mapping will interrupt the active localization session.")
            self._set_localization_status("Interrupted by Mapping")
            active.stop()
        self._start_uvc_preview_publisher(auto=True)
        args = ["map", "--map-name", map_name, "--viz", "off"]
        if self.mapping_recorddata_var.get():
            self._log("Mapping will also enable MindCloud recorddata. When mapping stops, recorddata will be turned off explicitly.")
            args.append("--recorddata")
        self._start_task("Mapping", args)

    def _start_recording(self) -> None:
        map_path = self._selected_map_path(self.map_combo_record)
        if map_path is None:
            messagebox.showwarning("No Map", "Select a map file first.")
            return
        active = self._active_localization_worker()
        if active is None or self.localization_map_path != map_path:
            self.pending_action = "record"
            self._ensure_localization_for_tab(map_path)
            return
        if self.record_localization_text.get() != "Ready":
            self.pending_action = "record"
            self._log("Path recording will start automatically after localization becomes ready.")
            return
        mission_name = self.record_name_var.get().strip() or generated_name("mission")
        self._log("Path recording requested. The active localization session will be reused, then path recording will start.")
        self._start_uvc_preview_publisher(auto=True)
        self._start_task(
            "Path Recording",
            [*self._projection_args(), "record", "--db", str(map_path), "--mission-name", mission_name, "--base-frame", "odin1_base_link", "--localization-wait-sec", "60", "--reuse-localization"],
        )

    def _start_record_localization(self) -> None:
        map_path = self._selected_map_path(self.map_combo_record)
        if map_path is None:
            messagebox.showwarning("No Map", "Select a map file first.")
            return
        self._start_shared_localization(map_path)

    def _stop_record_localization(self) -> None:
        self._stop_shared_localization()

    def _start_replay(self) -> None:
        map_path = self._selected_map_path(self.map_combo_replay)
        mission_path = self._selected_mission_path()
        if map_path is None:
            messagebox.showwarning("No Map", "Select a map file first.")
            return
        if mission_path is None:
            messagebox.showwarning("No Mission", "Select a mission file first.")
            return
        try:
            mission_payload = json.loads(mission_path.read_text(encoding="utf-8"))
            bound_map_db = str(mission_payload.get("bound_map_db") or "").strip()
        except Exception as exc:
            messagebox.showwarning("Mission Error", f"Failed to read mission file.\n{exc}")
            return
        if bound_map_db:
            try:
                bound_tokens = {
                    Path(bound_map_db).name.lower(),
                    Path(bound_map_db).stem.lower(),
                    Path(bound_map_db).parent.name.lower(),
                }
                map_tokens = {
                    map_path.name.lower(),
                    map_path.stem.lower(),
                    map_path.parent.name.lower(),
                }
                if not (bound_tokens & map_tokens):
                    messagebox.showwarning(
                        "Map Mismatch",
                        "This mission belongs to a different map. Select the recorded map first.",
                    )
                    return
            except Exception:
                pass
        active = self._active_localization_worker()
        if active is None or self.localization_map_path != map_path:
            self.pending_action = "replay"
            self._ensure_localization_for_tab(map_path)
            return
        if self.replay_localization_text.get() != "Ready":
            self.pending_action = "replay"
            self._log("Hybrid drive will start automatically after localization becomes ready.")
            return
        use_local_guidance = self._hybrid_local_guidance_enabled()
        if use_local_guidance:
            self._log("Hybrid drive requested. The active localization session will be reused, then the system will start lidar local guidance and blend local row guidance with global mission tracking.")
            self._start_uvc_preview_publisher(auto=True)
        else:
            self._log("Hybrid drive requested. The active localization session will be reused, then the system will run pure global mission tracking and keep the raw UVC preview visible.")
            self._start_uvc_preview_publisher(auto=True)
        self._start_camera_monitor(auto=True)
        self._start_task(
            "Hybrid Drive",
            [*self._hybrid_args(), "--db", str(map_path), "--mission", str(mission_path), "--base-frame", "odin1_base_link", "--localization-wait-sec", "60", "--reuse-localization"],
        )

    def _start_replay_localization(self) -> None:
        map_path = self._selected_map_path(self.map_combo_replay)
        if map_path is None:
            messagebox.showwarning("No Map", "Select a map file first.")
            return
        self._start_shared_localization(map_path)

    def _stop_replay_localization(self) -> None:
        self._stop_shared_localization()

    def _on_tab_changed(self, _event: object) -> None:
        current = self.notebook.tab(self.notebook.select(), "text")
        if current in {"Path Recording", "Hybrid Drive"}:
            self._ensure_localization_for_tab(auto=True)
        if current in {"Hybrid Drive", "Video Preview"}:
            self._start_camera_monitor(auto=True)

    def _delete_selected_map(self) -> None:
        selected = self._selected_map_path(self.map_combo_mapping)
        if selected is None:
            return
        folder = selected.parent
        if not messagebox.askyesno("Delete Map", f"Delete map folder?\n{folder}"):
            return
        try:
            shutil.rmtree(folder)
        except Exception as exc:
            messagebox.showwarning("Delete Failed", str(exc))
            return
        self._log(f"Deleted map folder: {folder}")
        self._refresh_maps()

    def _delete_selected_mission(self) -> None:
        selected = self._selected_mission_path()
        if selected is None:
            return
        if not messagebox.askyesno("Delete Mission", f"Delete mission?\n{selected.name}"):
            return
        csv_path = selected.with_suffix(".csv")
        try:
            selected.unlink(missing_ok=True)
            csv_path.unlink(missing_ok=True)
        except Exception as exc:
            messagebox.showwarning("Delete Failed", str(exc))
            return
        self._log(f"Deleted mission file: {selected}")
        self._refresh_missions()

    def _start_status_monitor(self, auto: bool = False) -> None:
        if self.status_monitor is not None:
            if not auto:
                self._log("Status monitor is already running.")
            return
        self.status_monitor = StatusMonitor(self.events)
        self.status_monitor.start()
        self._log("Status monitor started." if not auto else "Status monitor auto-started.")

    def _start_camera_monitor(self, auto: bool = False) -> None:
        if self.camera_monitor is not None:
            if not auto:
                self._log("Camera monitor is already running.")
            return
        self.camera_monitor = RosImageMonitor(self.events, topics=(UVC_PREVIEW_TOPIC,))
        self.camera_monitor.start()
        self.camera_status_text.set("Starting...")
        self._log("UVC preview monitor started." if not auto else "UVC preview monitor auto-started.")

    def _uvc_preview_args(self) -> list[str]:
        return [
            str(UVC_PREVIEW_SCRIPT),
            "--source", "/dev/video0",
            "--camera-width", "1024",
            "--camera-height", "768",
            "--camera-fps", "10",
            "--camera-fourcc", "MJPG",
            "--image-topic", UVC_PREVIEW_TOPIC,
            "--preview-width", "640",
            "--preview-height", "360",
            "--jpeg-quality", "35",
            "--publish-fps", "6.0",
        ]

    def _start_uvc_preview_publisher(self, auto: bool = False) -> None:
        if self.preview_worker is not None:
            return
        worker = ProcessWorker([ROS_PYTHON, *self._uvc_preview_args()], PROJECT_ROOT, "UVC Preview", self.events)
        self.preview_worker = worker
        worker.start()
        self._start_camera_monitor(auto=True)
        self._log("UVC preview publisher started." if not auto else "UVC preview publisher auto-started.")

    def _stop_uvc_preview_publisher(self, *, log_message: bool = True) -> None:
        if self.preview_worker is None:
            return
        self.preview_worker.stop()
        if log_message:
            self._log("Stop requested for UVC preview publisher.")

    def _start_scene_monitor(self, auto: bool = False) -> None:
        if self.scene_monitor is not None:
            if not auto:
                self._log("Scene monitor is already running.")
            return
        self.scene_monitor = RosSceneMonitor(self.events)
        self.scene_monitor.start()
        self.scene_status_text.set("Starting...")
        self._log("Scene monitor started." if not auto else "Scene monitor auto-started.")

    def _stop_camera_monitor(self, *, log_message: bool = True) -> None:
        if self.camera_monitor is None:
            if log_message:
                self._log("Camera monitor is not running.")
            return
        self.camera_monitor.stop()
        self.camera_monitor = None
        self.camera_status_text.set("Stopped")
        self.camera_photo = None
        self.last_camera_render_at = 0.0
        if log_message:
            self._log("Camera monitor stopped.")
        if self.pose_debug_monitor is not None:
            self.pose_debug_monitor.stop()
            self.pose_debug_monitor = None

    def _stop_scene_monitor(self) -> None:
        if self.scene_monitor is None:
            return
        self.scene_monitor.stop()
        self.scene_monitor = None
        self.scene_status_text.set("Stopped")

    def _apply_snapshot(self, snapshot: dict[str, Any]) -> None:
        motion = snapshot.get("motion", {})
        battery = snapshot.get("battery", {})
        io_state = snapshot.get("io", {})
        error = snapshot.get("error", {})
        self.vehicle_vars["gear"].set(str(motion.get("gear", "--")))
        self.vehicle_vars["vx"].set(str(motion.get("vx_mps", "--")))
        self.vehicle_vars["vy"].set(str(motion.get("vy_mps", "--")))
        self.vehicle_vars["wz"].set(str(motion.get("wz_dps", "--")))
        self.vehicle_vars["soc"].set(str(battery.get("soc_pct", "--")))
        self.vehicle_vars["voltage"].set(str(battery.get("voltage_v", "--")))
        self.vehicle_vars["current"].set(str(battery.get("current_a", "--")))
        self.vehicle_vars["capacity"].set(str(battery.get("capacity_ah", "--")))
        self.vehicle_vars["charging"].set(str(battery.get("charging", "--")))
        self.vehicle_vars["estop"].set(str(io_state.get("estop", "--")))
        self.vehicle_vars["remote"].set(str(io_state.get("remote_control", "--")))
        self.vehicle_vars["unlock_ok"].set(str(io_state.get("unlock_ok", "--")))
        self.vehicle_vars["charge_dock"].set(str(io_state.get("charge_dock", "--")))
        self.vehicle_vars["manual_charger"].set(str(io_state.get("manual_charger", "--")))
        self.vehicle_vars["light_mode"].set(str(io_state.get("light_mode", "--")))
        self.vehicle_vars["error"].set(f"{error.get('level', '--')}/{error.get('type', '--')}")
        self.vehicle_vars["component"].set(str(error.get("component", "--")))
        self.vehicle_vars["code"].set(str(error.get("code", "--")))
        self.vehicle_vars["updated_at"].set(str(snapshot.get("time", "--")))

    def _capture_pose_anchor(self) -> None:
        if self.last_raw_pose_debug is None:
            self._log("No pose is available yet for anchor capture.")
            return
        self.anchor_pose_debug = dict(self.last_raw_pose_debug)
        self.pose_vars["anchor"].set(
            f"x={self.anchor_pose_debug['x']:.3f}, y={self.anchor_pose_debug['y']:.3f}, roll={math.degrees(self.anchor_pose_debug['roll']):.1f}deg, pitch={math.degrees(self.anchor_pose_debug['pitch']):.1f}deg"
        )
        self._apply_pose_debug(self.last_raw_pose_debug)
        self._log("Current pose captured as the ground-projection anchor.")

    def _apply_pose_debug(self, raw_pose: dict[str, float]) -> None:
        self.last_raw_pose_debug = dict(raw_pose)
        self.pose_vars["raw_xy"].set(f"x={raw_pose['x']:.3f}, y={raw_pose['y']:.3f}, z={raw_pose['z']:.3f}")
        self.pose_vars["raw_rp"].set(
            f"roll={math.degrees(raw_pose['roll']):.1f}deg, pitch={math.degrees(raw_pose['pitch']):.1f}deg"
        )
        if self.anchor_pose_debug is None:
            self.pose_vars["anchor"].set("Not set")
            self.pose_vars["proj_xy"].set("--")
            self.pose_vars["proj_delta"].set("--")
            self.pose_vars["proj_rpy"].set("--")
            return
        try:
            sensor_height = float(self.sensor_height_var.get().strip() or DEFAULT_SENSOR_HEIGHT_M)
            body_x = float(self.body_x_offset_var.get().strip() or DEFAULT_BODY_X_OFFSET_M)
            body_y = float(self.body_y_offset_var.get().strip() or DEFAULT_BODY_Y_OFFSET_M)
            roll_gain = float(self.roll_gain_var.get().strip() or DEFAULT_ROLL_GAIN)
            pitch_gain = float(self.pitch_gain_var.get().strip() or DEFAULT_PITCH_GAIN)
        except ValueError:
            sensor_height = float(DEFAULT_SENSOR_HEIGHT_M)
            body_x = float(DEFAULT_BODY_X_OFFSET_M)
            body_y = float(DEFAULT_BODY_Y_OFFSET_M)
            roll_gain = float(DEFAULT_ROLL_GAIN)
            pitch_gain = float(DEFAULT_PITCH_GAIN)
        projected = project_ground_pose(
            raw_pose,
            sensor_height_m=max(sensor_height, 0.0),
            body_x_offset_m=body_x,
            body_y_offset_m=body_y,
            roll_gain=roll_gain,
            pitch_gain=pitch_gain,
            anchor_roll_rad=float(self.anchor_pose_debug["roll"]),
            anchor_pitch_rad=float(self.anchor_pose_debug["pitch"]),
        )
        anchor_projected = project_ground_pose(
            self.anchor_pose_debug,
            sensor_height_m=max(sensor_height, 0.0),
            body_x_offset_m=body_x,
            body_y_offset_m=body_y,
            roll_gain=roll_gain,
            pitch_gain=pitch_gain,
            anchor_roll_rad=float(self.anchor_pose_debug["roll"]),
            anchor_pitch_rad=float(self.anchor_pose_debug["pitch"]),
        )
        dx = projected["x"] - anchor_projected["x"]
        dy = projected["y"] - anchor_projected["y"]
        self.pose_vars["proj_xy"].set(f"x={projected['x']:.3f}, y={projected['y']:.3f}, z={projected['z']:.3f}")
        self.pose_vars["proj_delta"].set(f"dx={dx:+.3f}, dy={dy:+.3f}")
        self.pose_vars["proj_rpy"].set(
            f"roll={math.degrees(projected['roll']):.1f}deg, pitch={math.degrees(projected['pitch']):.1f}deg, yaw={math.degrees(projected['yaw']):.1f}deg"
        )

    def _stop_current_task(self) -> None:
        if self.task_worker is not None:
            if self.task_worker.stop_requested:
                self._log("A stop is already in progress for the current task.")
            else:
                self._log("Stop requested for the current task.")
            self.task_worker.stop()
            if self.task_worker.label in {"Mapping", "Path Recording", "Hybrid Drive"}:
                self._stop_uvc_preview_publisher(log_message=False)
            return
        active = self._active_localization_worker()
        if active is not None:
            if active.stop_requested:
                self._log("A stop is already in progress for shared localization.")
            else:
                self._log("Stop requested for shared localization.")
            active.stop()
            self._stop_uvc_preview_publisher(log_message=False)
            return
        self._log("No task is currently running.")

    def _on_close(self) -> None:
        if self.closing:
            return
        self.closing = True
        self._save_settings()
        if self.task_worker is not None:
            self.task_worker.stop()
        active = self._active_localization_worker()
        if active is not None:
            active.stop()
        if self.preview_worker is not None:
            self.preview_worker.stop()
        if self.status_monitor is not None:
            self.status_monitor.stop()
        if self.camera_monitor is not None:
            self.camera_monitor.stop()
        if self.scene_monitor is not None:
            self.scene_monitor.stop()
        try:
            self.root.quit()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass
        # ROS/background threads can occasionally keep the process alive.
        # Force the Python process to exit shortly after the UI is torn down.
        os._exit(0)

    def run(self) -> int:
        self.root.mainloop()
        return 0


def main() -> int:
    window = MainWindow()
    return window.run()


if __name__ == "__main__":
    raise SystemExit(main())
