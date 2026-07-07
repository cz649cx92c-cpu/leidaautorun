#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import rclpy
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
import tf2_ros

PROJECT_ROOT = Path(__file__).resolve().parent
ROOT = PROJECT_ROOT.parent
CONTROL_ROOT = ROOT / 'control'
RTABMAP_ROOT = ROOT / 'rtabmap'
MAPDATA_DIR = RTABMAP_ROOT / 'mapdata'
MISSIONS_DIR = PROJECT_ROOT / 'missions'
LOG_DIR = PROJECT_ROOT / 'logs'
ROS_PYTHON = '/usr/bin/python3'
FACTOR_KEY = '2402F63F601DAB2A54B44AEBAC35C5A8'
DEFAULT_SENSOR_HEIGHT_M = 1.2
DEFAULT_BODY_X_OFFSET_M = 0.0
DEFAULT_BODY_Y_OFFSET_M = 0.0
DEFAULT_ROLL_GAIN = 0.65
DEFAULT_PITCH_GAIN = 1.0

for path in (MISSIONS_DIR, LOG_DIR):
    path.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(CONTROL_ROOT))
from fw_mini_controller import BodyCommand, FWMiniController, IOCommand, SteeringCommand, auto_park  # noqa: E402
from fw_mini_status_reader import build_snapshot, decode_msg, open_can_bus  # noqa: E402

STOP_REQUESTED = False
ACTIVE_CHILDREN: list['ManagedProcess'] = []
ROS2_READY = False
ROS2_REFCOUNT = 0
ROS2_LOCK = threading.Lock()


class LocalizationCancelled(RuntimeError):
    pass


class PoseTracker(Protocol):
    def lookup(self) -> 'Pose2D | None':
        ...


def now_text() -> str:
    return datetime.now().strftime('%H:%M:%S')


def log(message: str) -> None:
    print(f'{now_text()} {message}', flush=True)


@dataclass
class Pose2D:
    x: float
    y: float
    z: float
    yaw: float
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0

    def distance_to(self, other: 'Pose2D') -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass
class GroundProjection:
    enabled: bool
    sensor_height_m: float
    body_x_offset_m: float
    body_y_offset_m: float
    roll_gain: float = 0.65
    pitch_gain: float = 1.0
    anchor_roll_rad: float = 0.0
    anchor_pitch_rad: float = 0.0


@dataclass
class MotionSendState:
    unlock_sequence: list[bool]
    unlock_request_active: bool = False
    motion_unlock_armed: bool = True
    unlock_confirmed_until: float = 0.0
    motion_unlocked_session: bool = False
    unlock_force_started_at: float = 0.0
    last_unlock_pulse_ts: float = 0.0
    last_vx: float = 0.0
    last_vy: float = 0.0
    last_wz: float = 0.0
    last_cmd_at: float = 0.0
    last_crab_angle_deg: float = 0.0
    last_steer_cmd_ts: float = 0.0
    remote_paused: bool = False
    filtered_heading_err_deg: float = 0.0
    filtered_lateral_err_m: float = 0.0
    last_sent_gear: str | None = None
    crab_locked_until: int = -1
    crab_target_index: int = -1
    crab_best_dist: float = float('inf')
    crab_diverge_count: int = 0
    active_gear_sync_count: int = 0
    feedback_mismatch_started_at: float = 0.0

    @classmethod
    def create(cls) -> 'MotionSendState':
        return cls(unlock_sequence=[])

    def queue_unlock_sequence(self) -> None:
        sequence = [True, True, False, False]
        if not self.unlock_sequence:
            self.unlock_sequence = sequence.copy()
        else:
            self.unlock_sequence.extend(sequence)

    def reset_motion(self) -> None:
        self.unlock_sequence.clear()
        self.unlock_request_active = False
        self.motion_unlock_armed = True
        self.unlock_confirmed_until = 0.0
        self.motion_unlocked_session = False
        self.unlock_force_started_at = 0.0
        self.last_unlock_pulse_ts = 0.0
        self.last_vx = 0.0
        self.last_vy = 0.0
        self.last_wz = 0.0
        self.last_cmd_at = 0.0
        self.last_crab_angle_deg = 0.0
        self.last_steer_cmd_ts = 0.0
        self.filtered_heading_err_deg = 0.0
        self.filtered_lateral_err_m = 0.0
        self.last_sent_gear = None
        self.crab_locked_until = -1
        self.crab_target_index = -1
        self.crab_best_dist = float('inf')
        self.crab_diverge_count = 0
        self.active_gear_sync_count = 0
        self.feedback_mismatch_started_at = 0.0


class ManagedProcess:
    def __init__(self, cmd: list[str], cwd: Path, log_path: Path | None = None) -> None:
        self.cmd = cmd
        self.cwd = cwd
        self.log_path = log_path
        self.log_handle = None
        self.process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        if self.log_path is not None:
            self.log_handle = self.log_path.open('w', encoding='utf-8')
            stdout = self.log_handle
        else:
            stdout = None
        self.process = subprocess.Popen(
            self.cmd,
            cwd=str(self.cwd),
            stdout=stdout,
            stderr=subprocess.STDOUT if stdout is not None else None,
            text=True,
            preexec_fn=os.setsid,
        )
        ACTIVE_CHILDREN.append(self)

    def poll(self) -> int | None:
        return None if self.process is None else self.process.poll()

    def stop(self, timeout: float = 5.0) -> None:
        if self.process is None or self.process.poll() is not None:
            self._close_log()
            return
        try:
            os.killpg(self.process.pid, signal.SIGINT)
            self.process.wait(timeout=timeout)
        except Exception:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=2.0)
            except Exception:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except Exception:
                    pass
        self._close_log()

    def _close_log(self) -> None:
        if self in ACTIVE_CHILDREN:
            ACTIVE_CHILDREN.remove(self)
        if self.log_handle is not None:
            try:
                self.log_handle.close()
            except Exception:
                pass
            self.log_handle = None


class LocalizationSession:
    def __init__(self, db_path: Path, *, viz: str = 'off') -> None:
        self.db_path = db_path
        self.raw_log = LOG_DIR / f'localization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        self.proc = ManagedProcess(
            [
                ROS_PYTHON,
                str(RTABMAP_ROOT / 'start_rtabmap.py'),
                '--action', 'localization',
                '--db', str(db_path),
                '--rtabmap-viz', viz,
                '--factor-key', FACTOR_KEY,
                '--rtabmap-sync-queue-size', '30',
                '--rtabmap-topic-queue-size', '20',
                '--wait-for-transform', '0.3',
                '--localization-start-delay', '12.0',
            ],
            RTABMAP_ROOT,
            self.raw_log,
        )

    def start(self) -> None:
        self.proc.start()
        log(f'Localization process started. Raw log: {self.raw_log}')

    def poll(self) -> int | None:
        return self.proc.poll()

    def stop(self) -> None:
        self.proc.stop()


def localize_with_retry(
    db_path: Path,
    map_frame: str,
    base_frame: str,
    node_name: str,
    timeout_sec: float,
    stage_name: str,
    attempts: int = 2,
    settle_pause_sec: float = 2.0,
) -> tuple[LocalizationSession, PoseTracker, Pose2D]:
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        if STOP_REQUESTED:
            raise LocalizationCancelled(f'{stage_name} was cancelled.')
        session = LocalizationSession(db_path, viz='off')
        session.start()
        try:
            tracker: PoseTracker | None = None
            tracker_deadline = time.monotonic() + max(1.0, timeout_sec)
            while time.monotonic() < tracker_deadline and not STOP_REQUESTED:
                try:
                    tracker = HybridPoseTracker(map_frame, base_frame, node_name)
                    break
                except Exception as exc:
                    last_exc = exc
                    time.sleep(0.5)
            if STOP_REQUESTED:
                raise LocalizationCancelled(f'{stage_name} was cancelled.')
            if tracker is None:
                raise RuntimeError(f'{stage_name} could not initialize the ROS TF tracker before timeout.')
            if attempt > 1:
                log(f'{stage_name} retry {attempt}/{attempts} started after the previous localization attempt did not stabilize.')
            pose = wait_for_stable_pose(tracker, session, timeout_sec, stage_name)
            return session, tracker, pose
        except Exception as exc:
            last_exc = exc
            session.stop()
            if isinstance(exc, LocalizationCancelled):
                break
            if attempt >= attempts or STOP_REQUESTED:
                break
            log(f'{stage_name} did not stabilize on attempt {attempt}. Waiting {settle_pause_sec:.0f}s before retrying localization...')
            time.sleep(settle_pause_sec)
    assert last_exc is not None
    raise last_exc


class CANFeedbackReader:
    def __init__(self, interface: str, channel: str, bitrate: int) -> None:
        _, self.bus = open_can_bus(interface, channel, bitrate, passive=True)
        self.latest: dict[str, Any] = {}

    def poll(self, timeout: float = 0.0, limit: int = 50) -> dict[str, Any]:
        count = 0
        while count < limit:
            msg = self.bus.recv(timeout=timeout if count == 0 else 0.0)
            if msg is None:
                break
            decoded = decode_msg(msg.arbitration_id, bytes(msg.data))
            if decoded:
                self.latest[decoded['name']] = decoded['data']
            count += 1
        return self.latest

    def snapshot(self) -> dict[str, Any]:
        return build_snapshot(self.latest) if self.latest else {}

    def close(self) -> None:
        self.bus.shutdown()


def install_signal_handlers() -> None:
    def _handler(signum, frame):
        global STOP_REQUESTED
        STOP_REQUESTED = True
        for child in ACTIVE_CHILDREN[:]:
            child.stop()

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    return (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
    )


def _quat_to_rpy(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float, float]:
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


def project_pose_to_ground(pose: Pose2D, projection: GroundProjection) -> Pose2D:
    if not projection.enabled:
        return pose
    roll, pitch, yaw = _quat_to_rpy(pose.qx, pose.qy, pose.qz, pose.qw)
    rel_roll = (roll - projection.anchor_roll_rad) * projection.roll_gain
    rel_pitch = (pitch - projection.anchor_pitch_rad) * projection.pitch_gain
    ground_forward = projection.body_x_offset_m - projection.sensor_height_m * math.sin(rel_pitch)
    ground_left = projection.body_y_offset_m + projection.sensor_height_m * math.sin(rel_roll)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    dx = cy * ground_forward - sy * ground_left
    dy = sy * ground_forward + cy * ground_left
    dz = -projection.sensor_height_m * math.cos(rel_roll) * math.cos(rel_pitch)
    return Pose2D(
        x=pose.x + dx,
        y=pose.y + dy,
        z=pose.z + dz,
        yaw=pose.yaw,
        qx=pose.qx,
        qy=pose.qy,
        qz=pose.qz,
        qw=pose.qw,
    )


def projection_from_args(args: argparse.Namespace) -> GroundProjection:
    sensor_height = float(getattr(args, 'sensor_height_m', DEFAULT_SENSOR_HEIGHT_M) or DEFAULT_SENSOR_HEIGHT_M)
    body_x = float(getattr(args, 'body_x_offset_m', DEFAULT_BODY_X_OFFSET_M) or DEFAULT_BODY_X_OFFSET_M)
    body_y = float(getattr(args, 'body_y_offset_m', DEFAULT_BODY_Y_OFFSET_M) or DEFAULT_BODY_Y_OFFSET_M)
    roll_gain = float(getattr(args, 'roll_gain', DEFAULT_ROLL_GAIN) or DEFAULT_ROLL_GAIN)
    pitch_gain = float(getattr(args, 'pitch_gain', DEFAULT_PITCH_GAIN) or DEFAULT_PITCH_GAIN)
    return GroundProjection(
        enabled=sensor_height > 1e-6 or abs(body_x) > 1e-6 or abs(body_y) > 1e-6,
        sensor_height_m=max(sensor_height, 0.0),
        body_x_offset_m=body_x,
        body_y_offset_m=body_y,
        roll_gain=roll_gain,
        pitch_gain=pitch_gain,
    )


def anchor_projection_to_pose(projection: GroundProjection, pose: Pose2D) -> GroundProjection:
    if not projection.enabled:
        return projection
    roll, pitch, _yaw = _quat_to_rpy(pose.qx, pose.qy, pose.qz, pose.qw)
    return GroundProjection(
        enabled=projection.enabled,
        sensor_height_m=projection.sensor_height_m,
        body_x_offset_m=projection.body_x_offset_m,
        body_y_offset_m=projection.body_y_offset_m,
        roll_gain=projection.roll_gain,
        pitch_gain=projection.pitch_gain,
        anchor_roll_rad=roll,
        anchor_pitch_rad=pitch,
    )


def ensure_ros2_runtime() -> None:
    global ROS2_READY, ROS2_REFCOUNT
    with ROS2_LOCK:
        if not rclpy.ok():
            rclpy.init(args=None)
        ROS2_READY = True
        ROS2_REFCOUNT += 1


def release_ros2_runtime() -> None:
    global ROS2_REFCOUNT, ROS2_READY
    with ROS2_LOCK:
        ROS2_REFCOUNT = max(0, ROS2_REFCOUNT - 1)
        if ROS2_REFCOUNT == 0 and rclpy.ok():
            rclpy.shutdown()
            ROS2_READY = False


class TFPoseTracker:
    def __init__(self, fixed_frame: str, base_frame: str, node_name: str) -> None:
        self.fixed_frame = fixed_frame
        self.base_frame = base_frame
        ensure_ros2_runtime()
        self.node = Node(node_name)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.spin_thread = threading.Thread(target=self.executor.spin, name=f"{node_name}-spin", daemon=True)
        self.spin_thread.start()
        self.buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.listener = tf2_ros.TransformListener(self.buffer, self.node, spin_thread=False)
        self._lock = threading.Lock()
        self._latest_odom_stamp = None
        self._latest_odom_recv = 0.0
        self._odom_sub = self.node.create_subscription(Odometry, '/odin1/odometry_highfreq', self._on_odom, 10)

    def _on_odom(self, msg) -> None:
        stamp = getattr(msg.header, 'stamp', None)
        if stamp is None:
            return
        with self._lock:
            self._latest_odom_stamp = stamp
            self._latest_odom_recv = time.monotonic()

    def _lookup_sync_stamp(self):
        now = time.monotonic()
        with self._lock:
            if self._latest_odom_stamp is not None and (now - self._latest_odom_recv) < 0.30:
                return self._latest_odom_stamp
        return Time()

    def lookup(self) -> Pose2D | None:
        lookup_stamp = self._lookup_sync_stamp()
        try:
            tf = self.buffer.lookup_transform(self.fixed_frame, self.base_frame, lookup_stamp, timeout=Duration(seconds=0.08))
        except Exception:
            try:
                tf = self.buffer.lookup_transform(self.fixed_frame, self.base_frame, Time(), timeout=Duration(seconds=0.2))
            except Exception:
                return None
        trans = tf.transform.translation
        rot = tf.transform.rotation
        yaw = math.atan2(2.0 * (rot.w * rot.z + rot.x * rot.y), 1.0 - 2.0 * (rot.y * rot.y + rot.z * rot.z))
        return Pose2D(trans.x, trans.y, trans.z, yaw, rot.x, rot.y, rot.z, rot.w)

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
            self.node.destroy_subscription(self._odom_sub)
        except Exception:
            pass
        try:
            self.node.destroy_node()
        except Exception:
            pass
        if self.spin_thread.is_alive():
            self.spin_thread.join(timeout=1.0)
        release_ros2_runtime()


class OdomPoseTracker:
    def __init__(self, node_name: str) -> None:
        ensure_ros2_runtime()
        self.node = Node(node_name)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.spin_thread = threading.Thread(target=self.executor.spin, name=f"{node_name}-spin", daemon=True)
        self.spin_thread.start()
        self._lock = threading.Lock()
        self._latest_pose: Pose2D | None = None
        self._latest_recv = 0.0
        self._odom_sub = self.node.create_subscription(Odometry, '/odin1/odometry_highfreq', self._on_odom, 10)

    def _on_odom(self, msg) -> None:
        pose = msg.pose.pose
        pos = pose.position
        rot = pose.orientation
        yaw = math.atan2(2.0 * (rot.w * rot.z + rot.x * rot.y), 1.0 - 2.0 * (rot.y * rot.y + rot.z * rot.z))
        latest = Pose2D(pos.x, pos.y, pos.z, yaw, rot.x, rot.y, rot.z, rot.w)
        with self._lock:
            self._latest_pose = latest
            self._latest_recv = time.monotonic()

    def lookup(self) -> Pose2D | None:
        with self._lock:
            if self._latest_pose is None:
                return None
            if (time.monotonic() - self._latest_recv) > 0.6:
                return None
            return self._latest_pose

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
            self.node.destroy_subscription(self._odom_sub)
        except Exception:
            pass
        try:
            self.node.destroy_node()
        except Exception:
            pass
        if self.spin_thread.is_alive():
            self.spin_thread.join(timeout=1.0)
        release_ros2_runtime()


class HybridPoseTracker:
    def __init__(self, fixed_frame: str, base_frame: str, node_name: str) -> None:
        self.fixed_frame = fixed_frame
        self.base_frame = base_frame
        self.odom_tracker = OdomPoseTracker(f'{node_name}_odom')
        self.tf_tracker: TFPoseTracker | None = None
        self.tf_error: Exception | None = None
        try:
            self.tf_tracker = TFPoseTracker(fixed_frame, base_frame, node_name)
        except Exception as exc:
            self.tf_error = exc
            log(
                f'Falling back to /odin1/odometry_highfreq because TF tracker '
                f'for {fixed_frame}->{base_frame} is not ready yet: {exc}'
            )

    def lookup(self) -> Pose2D | None:
        if self.tf_tracker is not None:
            pose = self.tf_tracker.lookup()
            if pose is not None:
                return pose
        return self.odom_tracker.lookup()

    def close(self) -> None:
        if self.tf_tracker is not None:
            self.tf_tracker.close()
        self.odom_tracker.close()


def _close_tracker(tracker: PoseTracker | None) -> None:
    if tracker is None:
        return
    close_fn = getattr(tracker, 'close', None)
    if callable(close_fn):
        try:
            close_fn()
        except Exception:
            pass


def wait_for_stable_pose(tracker: PoseTracker, session: LocalizationSession | None, timeout_sec: float, stage_name: str) -> Pose2D:
    deadline = time.monotonic() + timeout_sec
    stable_hits = 0
    last_pose: Pose2D | None = None
    last_log = 0.0
    while time.monotonic() < deadline and not STOP_REQUESTED:
        if session is not None and session.poll() is not None:
            raise RuntimeError(f'{stage_name} stopped unexpectedly. Check log: {session.raw_log}')
        pose = tracker.lookup()
        now = time.monotonic()
        if pose is None:
            if now - last_log >= 2.0:
                log('Waiting for a valid localization pose...')
                last_log = now
            time.sleep(0.2)
            continue
        if last_pose is not None:
            dist = pose.distance_to(last_pose)
            dyaw = abs(normalize_angle(pose.yaw - last_pose.yaw))
            stable_hits = stable_hits + 1 if dist < 0.05 and dyaw < math.radians(5.0) else 0
        last_pose = pose
        if stable_hits >= 3:
            log('Localization is stable.')
            return pose
        time.sleep(0.2)
    if STOP_REQUESTED:
        raise LocalizationCancelled(f'{stage_name} was cancelled.')
    if last_pose is not None:
        log(
            f'{stage_name} did not reach the strict stability threshold before timeout, '
            'but a valid localization pose is available. Keeping the localization session running.'
        )
        return last_pose
    raise RuntimeError(f'{stage_name} timed out while waiting for a valid localization pose. Check log: {session.raw_log if session else "none"}')


def create_tracker_with_retry(map_frame: str, base_frame: str, node_name: str, timeout_sec: float, stage_name: str) -> PoseTracker:
    last_exc: Exception | None = None
    deadline = time.monotonic() + min(8.0, timeout_sec)
    while time.monotonic() < deadline and not STOP_REQUESTED:
        try:
            return HybridPoseTracker(map_frame, base_frame, node_name)
        except Exception as exc:
            last_exc = exc
            time.sleep(0.5)
    if STOP_REQUESTED:
        raise LocalizationCancelled(f'{stage_name} was cancelled.')
    if last_exc is not None:
        raise RuntimeError(f'{stage_name} could not initialize the ROS TF tracker before timeout: {last_exc}')
    raise RuntimeError(f'{stage_name} could not initialize the ROS TF tracker before timeout.')


def ensure_can_ready(channel: str, bitrate: int) -> None:
    state_path = Path('/sys/class/net') / channel / 'operstate'
    if not state_path.exists():
        raise RuntimeError(f'CAN channel not found: {channel}')
    state = state_path.read_text(encoding='utf-8').strip().lower()
    if state in {'up', 'unknown'}:
        return

    def _run_ip(cmd: list[str], *, allow_failure: bool = False) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0 and not allow_failure:
            detail = (result.stderr or '').strip()
            if detail:
                raise RuntimeError(f'Failed to configure CAN channel {channel}: {" ".join(cmd)}: {detail}')
            raise RuntimeError(f'Failed to configure CAN channel {channel}: {" ".join(cmd)} returned {result.returncode}')
        return result

    # Some kernels/drivers return an error when the interface is already down.
    # Treat that as non-fatal and continue with the CAN type/bitrate setup.
    _run_ip(['ip', 'link', 'set', channel, 'down'], allow_failure=True)
    _run_ip(['ip', 'link', 'set', channel, 'type', 'can', 'bitrate', str(bitrate)])
    _run_ip(['ip', 'link', 'set', channel, 'up'])


def generated_name(prefix: str) -> str:
    return f'{prefix}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'


def map_identity_tokens(map_db: str | Path | None) -> set[str]:
    if not map_db:
        return set()
    try:
        path = Path(map_db).resolve()
    except Exception:
        path = Path(map_db)
    tokens = {
        str(path),
        path.name,
        path.stem,
        path.parent.name,
    }
    return {token.strip().lower() for token in tokens if str(token).strip()}


def same_map_identity(left_map_db: str | Path | None, right_map_db: str | Path | None) -> bool:
    left = map_identity_tokens(left_map_db)
    right = map_identity_tokens(right_map_db)
    if not left or not right:
        return False
    if str(Path(str(left_map_db)).name).lower() == str(Path(str(right_map_db)).name).lower():
        return True
    if str(Path(str(left_map_db)).stem).lower() == str(Path(str(right_map_db)).stem).lower():
        return True
    return bool(left & right)


def mission_storage_dir_for_map_db(map_db: str | Path) -> Path:
    return Path(map_db).resolve().parent


def write_mission_files(
    mission_name: str,
    payload: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    map_db: str | Path,
) -> tuple[Path, Path]:
    storage_dir = mission_storage_dir_for_map_db(map_db)
    storage_dir.mkdir(parents=True, exist_ok=True)
    json_path = storage_dir / f'{mission_name}.json'
    csv_path = storage_dir / f'{mission_name}.csv'
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    with csv_path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['index', 't', 'x', 'y', 'z', 'yaw_deg', 'gear', 'vx', 'vy', 'wz'])
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def _sample_pose(sample: dict[str, Any]) -> Pose2D:
    pose = sample.get('pose', {})
    return Pose2D(
        float(pose.get('x', 0.0)),
        float(pose.get('y', 0.0)),
        float(pose.get('z', 0.0)),
        math.radians(float(pose.get('yaw_deg', 0.0))),
    )


def _sample_motion(sample: dict[str, Any]) -> dict[str, Any]:
    snapshot = sample.get('snapshot', {})
    motion = snapshot.get('motion', {}) if isinstance(snapshot, dict) else {}
    return {
        'gear': str(motion.get('gear') or '4t4d'),
        'vx': float(motion.get('vx_mps') or 0.0),
        'vy': float(motion.get('vy_mps') or 0.0),
        'wz': float(motion.get('wz_dps') or 0.0),
    }


def _estimate_motion_from_path(points: list[Pose2D], sample_period: float, index: int) -> dict[str, Any]:
    cur = points[index]
    nxt = points[min(index + 1, len(points) - 1)]
    dt = max(sample_period, 1e-3)
    dx = nxt.x - cur.x
    dy = nxt.y - cur.y
    dyaw = normalize_angle(nxt.yaw - cur.yaw)
    c = math.cos(cur.yaw)
    s = math.sin(cur.yaw)
    forward = (c * dx + s * dy) / dt
    lateral = (-s * dx + c * dy) / dt
    wz = math.degrees(dyaw) / dt
    gear = 'crab' if abs(lateral) > abs(forward) * 1.2 and abs(lateral) > 0.03 else '4t4d'
    return {'gear': gear, 'vx': forward, 'vy': lateral, 'wz': wz}


def _repair_missing_motion(samples: list[dict[str, Any]], sample_period: float) -> tuple[list[dict[str, Any]], bool]:
    points = [_sample_pose(sample) for sample in samples]
    motions: list[dict[str, Any]] = []
    used_fallback = False
    for index, sample in enumerate(samples):
        snapshot = sample.get('snapshot', {})
        motion_data = snapshot.get('motion', {}) if isinstance(snapshot, dict) else {}
        has_motion = any(motion_data.get(key) is not None for key in ('gear', 'vx_mps', 'vy_mps', 'wz_dps'))
        if has_motion:
            motions.append(_sample_motion(sample))
        else:
            motions.append(_estimate_motion_from_path(points, sample_period, index))
            used_fallback = True
    return motions, used_fallback


def _stabilize_crab_segments(motions: list[dict[str, Any]]) -> int:
    fix_count = 0
    index = 0
    while index < len(motions):
        if _resolve_replay_gear(motions[index], None) != 'crab':
            index += 1
            continue
        start = index
        end = _find_gear_segment_end(motions, start, 'crab')
        signs: list[float] = []
        for j in range(start, end + 1):
            vy = float(motions[j].get('vy', 0.0) or 0.0)
            if abs(vy) >= 0.05:
                signs.append(math.copysign(1.0, vy))
        dominant = 0.0
        if signs:
            dominant = 1.0 if signs.count(1.0) >= signs.count(-1.0) else -1.0
        if dominant != 0.0:
            for j in range(start, end + 1):
                vy = float(motions[j].get('vy', 0.0) or 0.0)
                if abs(vy) < 0.05:
                    continue
                if math.copysign(1.0, vy) != dominant:
                    motions[j] = {
                        **motions[j],
                        'vx': 0.0 if abs(float(motions[j].get('vx', 0.0) or 0.0)) < 0.08 else float(motions[j].get('vx', 0.0) or 0.0),
                        'vy': abs(vy) * dominant,
                    }
                    fix_count += 1
        index = end + 1
    return fix_count


def _soften_low_speed_turns(motions: list[dict[str, Any]]) -> int:
    softened = 0
    for motion in motions:
        gear = str(motion.get('gear') or '4t4d')
        if gear == 'crab':
            continue
        vx = abs(float(motion.get('vx', 0.0) or 0.0))
        vy = abs(float(motion.get('vy', 0.0) or 0.0))
        wz = float(motion.get('wz', 0.0) or 0.0)
        if vx < 0.12 and vy < 0.02 and abs(wz) > 2.0:
            motion['wz'] = 0.0
            softened += 1
    return softened


def _regularize_recorded_motion(
    samples: list[dict[str, Any]],
    sample_period: float,
) -> tuple[list[dict[str, Any]], bool, int]:
    motions, used_fallback = _repair_missing_motion(samples, sample_period)
    crab_fix_count = _stabilize_crab_segments(motions)
    soften_count = _soften_low_speed_turns(motions)
    soften_count = _soften_low_speed_turns(motions)
    points = [_sample_pose(sample) for sample in samples]
    estimated = [_estimate_motion_from_path(points, sample_period, i) for i in range(len(samples))]
    adjusted: list[dict[str, Any]] = []
    fix_count = 0

    def near_crab_segment(index: int) -> bool:
        start = max(0, index - 3)
        end = min(len(motions), index + 4)
        for j in range(start, end):
            if _resolve_replay_gear(motions[j], None) == 'crab':
                return True
        return False

    for index, motion in enumerate(motions):
        current = dict(motion)
        est = estimated[index]
        resolved = _resolve_replay_gear(current, None)
        lateral_est = abs(est['vy']) > max(0.05, abs(est['vx']) * 1.3)

        if resolved == 'crab':
            if lateral_est:
                current = {
                    'gear': 'crab',
                    'vx': 0.0 if abs(est['vx']) < abs(est['vy']) * 0.6 else est['vx'],
                    'vy': est['vy'],
                    'wz': est['wz'],
                }
                fix_count += 1
        else:
            raw_vx = abs(float(current.get('vx', 0.0) or 0.0))
            if lateral_est and raw_vx < 0.06 and near_crab_segment(index):
                current = {
                    'gear': 'crab',
                    'vx': est['vx'],
                    'vy': est['vy'],
                    'wz': est['wz'],
                }
                fix_count += 1

        adjusted.append(current)

    fix_count += _stabilize_crab_segments(adjusted)
    return adjusted, used_fallback, fix_count


def _target_heading(points: list[Pose2D], index: int) -> float:
    if index < len(points) - 1:
        cur = points[index]
        nxt = points[index + 1]
        if cur.distance_to(nxt) > 1e-6:
            return math.atan2(nxt.y - cur.y, nxt.x - cur.x)
    return points[index].yaw


def _resolve_replay_gear(motion: dict[str, Any], current_gear: str | None) -> str:
    gear = str(motion.get('gear') or '')
    if gear == 'crab':
        return 'crab'
    if gear == '4t4d':
        return '4t4d'
    if gear in {'park', 'neutral', '', '--'}:
        return current_gear or '4t4d'
    vx = abs(float(motion.get('vx', 0.0) or 0.0))
    vy = abs(float(motion.get('vy', 0.0) or 0.0))
    wz = abs(float(motion.get('wz', 0.0) or 0.0))
    if vy > max(0.03, vx * 1.2):
        return 'crab'
    if vx > 0.03 or wz > 3.0:
        return '4t4d'
    return current_gear or '4t4d'


def _tracking_heading(points: list[Pose2D], motions: list[dict[str, Any]], index: int, gear: str) -> float:
    heading = _target_heading(points, index)
    if gear == '4t4d' and float(motions[index].get('vx', 0.0) or 0.0) < -0.03:
        return normalize_angle(heading + math.pi)
    return heading


def _body_frame_error(pose: Pose2D, target: Pose2D) -> tuple[float, float]:
    dx = target.x - pose.x
    dy = target.y - pose.y
    c = math.cos(pose.yaw)
    s = math.sin(pose.yaw)
    return c * dx + s * dy, -s * dx + c * dy


def _axis_progress_reached(current: float, target: float, direction: float, tol: float) -> bool:
    if abs(direction) < 1e-9:
        return abs(target - current) <= tol
    if direction > 0.0:
        return current >= (target - tol)
    return current <= (target + tol)


def _find_future_gear_start(motions: list[dict[str, Any]], start_index: int, gear: str, limit: int = 24) -> int | None:
    upper = min(len(motions), start_index + limit + 1)
    for idx in range(start_index, upper):
        if _resolve_replay_gear(motions[idx], None) == gear:
            return idx
    return None


def _find_gear_segment_end(motions: list[dict[str, Any]], start_index: int, gear: str) -> int:
    end = start_index
    while end + 1 < len(motions) and _resolve_replay_gear(motions[end + 1], None) == gear:
        end += 1
    return end


def _find_first_active_crab_index(motions: list[dict[str, Any]], start_index: int, end_index: int) -> int:
    for idx in range(start_index, end_index + 1):
        vy = abs(float(motions[idx].get('vy', 0.0) or 0.0))
        if vy >= 0.05:
            return idx
    return start_index


def _find_last_active_crab_index(motions: list[dict[str, Any]], start_index: int, end_index: int) -> int:
    for idx in range(end_index, start_index - 1, -1):
        motion = motions[idx]
        vy = abs(float(motion.get('vy', 0.0) or 0.0))
        vx = abs(float(motion.get('vx', 0.0) or 0.0))
        wz = abs(float(motion.get('wz', 0.0) or 0.0))
        if vy >= 0.05 or vx >= 0.05 or wz >= 3.0:
            return idx
    return end_index


def _select_crab_progress_target(start_index: int, locked_until: int, minimum_target: int) -> int:
    if locked_until < 0:
        return minimum_target
    return max(minimum_target, locked_until)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _signed_crawl(forward_err: float, magnitude: float = 0.025) -> float:
    if forward_err > 0.0:
        return magnitude
    if forward_err < 0.0:
        return -magnitude
    return 0.0


def _normalize_autodrive_feedback_gear(value: Any) -> str:
    gear = str(value or '').strip().lower()
    if gear == '4t4d':
        return '4t4d'
    if gear == 'crab':
        return 'crab'
    if gear and gear not in {'-', '--'}:
        return gear
    return ''


def _limit_4t4d_turn_rate(vx: float, wz: float) -> float:
    speed = abs(vx)
    if vx > 0.0:
        if speed < 0.035:
            return 0.0 if abs(wz) < 2.5 else _clamp(wz, -2.0, 2.0)
        if speed < 0.060:
            return 0.0 if abs(wz) < 2.0 else _clamp(wz, -3.0, 3.0)
        if speed < 0.090:
            return 0.0 if abs(wz) < 1.6 else _clamp(wz, -4.5, 4.5)
    if speed < 0.035:
        return _clamp(wz, -3.0, 3.0)
    if speed < 0.060:
        return _clamp(wz, -4.5, 4.5)
    if speed < 0.090:
        return _clamp(wz, -6.0, 6.0)
    return wz


def _normalize_crab_target(vx: float, vy: float) -> tuple[float, float]:
    speed = math.hypot(vx, vy)
    if speed < 0.02:
        return 0.0, 0.0
    if abs(vx) < 0.02:
        return (90.0 if vy >= 0.0 else -90.0), abs(vy)
    if abs(vy) < 0.02:
        return 0.0, vx
    angle = math.degrees(math.atan2(vy, vx))
    wheel_speed = speed
    if angle > 90.0:
        angle -= 180.0
        wheel_speed = -wheel_speed
    elif angle < -90.0:
        angle += 180.0
        wheel_speed = -wheel_speed
    return angle, wheel_speed


def _slew_steer_angle(state: MotionSendState, target_angle: float, rate_dps: float = 120.0) -> float:
    now = time.monotonic()
    dt = max(0.02, now - state.last_steer_cmd_ts) if state.last_steer_cmd_ts > 0.0 else 0.05
    max_delta = rate_dps * dt
    angle = max(state.last_crab_angle_deg - max_delta, min(state.last_crab_angle_deg + max_delta, target_angle))
    state.last_crab_angle_deg = angle
    state.last_steer_cmd_ts = now
    return angle


def _slew_axis(prev: float, target: float, step: float) -> float:
    if target > prev + step:
        return prev + step
    if target < prev - step:
        return prev - step
    return target


def _smooth_drive_command(
    state: MotionSendState,
    gear: str,
    target_vx: float,
    target_vy: float,
    target_wz: float,
) -> tuple[float, float, float]:
    now = time.monotonic()
    dt = max(0.03, now - state.last_cmd_at) if state.last_cmd_at > 0.0 else 0.05
    if gear in {'park', 'neutral'}:
        state.last_vx = 0.0
        state.last_vy = 0.0
        state.last_wz = 0.0
        state.last_cmd_at = now
        return 0.0, 0.0, 0.0
    vx_step = 0.24 * dt
    vy_step = 0.22 * dt
    wz_step = 48.0 * dt
    state.last_vx = _slew_axis(state.last_vx, target_vx, vx_step)
    state.last_vy = _slew_axis(state.last_vy, target_vy, vy_step)
    state.last_wz = _slew_axis(state.last_wz, target_wz, wz_step)
    state.last_cmd_at = now
    return state.last_vx, state.last_vy, state.last_wz


def _drivable_indexes(points: list[Pose2D], motions: list[dict[str, Any]]) -> list[int]:
    out: list[int] = []
    for i, motion in enumerate(motions):
        moving_hint = abs(float(motion.get('vx', 0.0))) > 1e-3 or abs(float(motion.get('vy', 0.0))) > 1e-3 or abs(float(motion.get('wz', 0.0))) > 1e-3
        pose_change = i < len(points) - 1 and points[i].distance_to(points[i + 1]) > 0.03
        if moving_hint or pose_change:
            out.append(i)
    return out


def _choose_start_index(points: list[Pose2D], motions: list[dict[str, Any]]) -> int:
    candidates = _drivable_indexes(points, motions)
    if not candidates:
        return 0
    anchor = points[candidates[0]]
    for idx in candidates:
        motion = motions[idx]
        dist_from_anchor = points[idx].distance_to(anchor)
        moving = abs(float(motion.get('vx', 0.0) or 0.0)) > 0.05 or abs(float(motion.get('vy', 0.0) or 0.0)) > 0.05
        turning = abs(float(motion.get('wz', 0.0) or 0.0)) > 8.0
        if dist_from_anchor >= 0.25 and (moving or turning):
            return idx
    return candidates[0]


def _find_tracking_index(points: list[Pose2D], current_index: int, pose: Pose2D, window: int = 12) -> int:
    if not points:
        return 0
    start = max(0, current_index)
    end = min(len(points), current_index + max(2, window))
    best_index = current_index
    best_dist = float('inf')
    for idx in range(start, end):
        dist = pose.distance_to(points[idx])
        if dist < best_dist:
            best_dist = dist
            best_index = idx
    return best_index


def _smooth_tracking_errors(
    state: MotionSendState,
    heading_err_deg: float,
    lateral_err_m: float,
    alpha: float = 0.22,
) -> tuple[float, float]:
    state.filtered_heading_err_deg += alpha * (heading_err_deg - state.filtered_heading_err_deg)
    state.filtered_lateral_err_m += alpha * (lateral_err_m - state.filtered_lateral_err_m)
    return state.filtered_heading_err_deg, state.filtered_lateral_err_m


def _send_drive(
    controller: FWMiniController,
    state: MotionSendState,
    snapshot: dict[str, Any],
    gear: str,
    vx: float,
    vy: float,
    wz: float,
) -> tuple[bool, bool, float, float, float]:
    now = time.monotonic()
    body_feedback_gear = _normalize_autodrive_feedback_gear(snapshot.get('motion', {}).get('gear'))
    steer_feedback_gear = _normalize_autodrive_feedback_gear(snapshot.get('steering', {}).get('gear'))
    raw_feedback_mismatch = False
    if gear in {'4t4d', 'crab'}:
        if body_feedback_gear and body_feedback_gear != gear:
            raw_feedback_mismatch = True
        if steer_feedback_gear and steer_feedback_gear not in {gear, body_feedback_gear or gear}:
            raw_feedback_mismatch = True
    urgent_feedback_mismatch = (
        gear in {'4t4d', 'crab'}
        and (
            body_feedback_gear in {'0', '1', '2', 'neutral', 'park'}
            or steer_feedback_gear in {'0', '1', '2', 'neutral', 'park'}
        )
    )
    if raw_feedback_mismatch:
        if state.feedback_mismatch_started_at <= 0.0:
            state.feedback_mismatch_started_at = now
    else:
        state.feedback_mismatch_started_at = 0.0
    mismatch_grace_s = 0.22
    feedback_mismatch = urgent_feedback_mismatch or (
        raw_feedback_mismatch
        and state.feedback_mismatch_started_at > 0.0
        and (now - state.feedback_mismatch_started_at) >= mismatch_grace_s
    )
    if gear != state.last_sent_gear:
        # Match the reference control app: when the mode changes, synchronize
        # both body and steering controllers to the same gear with zero motion
        # before sending actual movement commands.
        controller.send_body(BodyCommand(gear=gear, vx=0.0, vy=0.0, wz=0.0))
        if gear != 'crab':
            controller.send_steering(SteeringCommand(gear=gear, speed=0.0, angle=0.0))
        state.last_vx = 0.0
        state.last_vy = 0.0
        state.last_wz = 0.0
        state.last_cmd_at = 0.0
        state.last_sent_gear = gear
    elif feedback_mismatch:
        # If the chassis feedback drops one control channel back to park/disable
        # while we still expect an active driving mode, resynchronize the gear
        # before sending the next motion command.
        controller.send_body(BodyCommand(gear=gear, vx=0.0, vy=0.0, wz=0.0))
        if gear not in {'4t4d', 'crab'}:
            controller.send_steering(SteeringCommand(gear=gear, speed=0.0, angle=0.0))
        state.last_cmd_at = 0.0
    motion_active = gear not in {'park', 'neutral'} and any(abs(v) > 1e-6 for v in (vx, vy, wz))
    io_state = snapshot.get('io', {})
    reported_unlock_ok = bool(io_state.get('unlock_ok', False))
    # Match the working ROS2 line-runner path: some FW-mini units keep reporting
    # neutral or delayed gear feedback for a short while even though they accept
    # motion once unlock is asserted. Do not let delayed feedback permanently
    # block motion dispatch here.
    state.active_gear_sync_count = 1000 if gear in {'4t4d', 'crab'} else 0
    if reported_unlock_ok:
        state.unlock_confirmed_until = now + 2.0
        state.motion_unlocked_session = True
        state.unlock_request_active = False
        state.unlock_force_started_at = 0.0
    effective_unlock_ok = state.motion_unlocked_session or reported_unlock_ok or now < state.unlock_confirmed_until
    waiting_unlock = motion_active and not effective_unlock_ok
    if motion_active and state.motion_unlock_armed:
        state.unlock_request_active = True
        state.queue_unlock_sequence()
        state.motion_unlock_armed = False
        state.unlock_force_started_at = now
        state.last_unlock_pulse_ts = now
    elif state.unlock_request_active and not effective_unlock_ok and not state.unlock_sequence:
        state.queue_unlock_sequence()
        state.last_unlock_pulse_ts = now
    elif not motion_active:
        state.motion_unlock_armed = True
        state.unlock_force_started_at = 0.0
        state.last_unlock_pulse_ts = 0.0

    # Some chassis do not report unlock_ok reliably, but they still accept motion
    # after unlock has been asserted for a short while. Mirror the practical
    # behavior of the reference control app and stop blocking forever.
    if motion_active and not effective_unlock_ok:
        if state.unlock_force_started_at <= 0.0:
            state.unlock_force_started_at = now
        if now - state.unlock_force_started_at >= 0.6:
            effective_unlock_ok = True
            waiting_unlock = False
            state.motion_unlocked_session = True

    if motion_active and not state.unlock_sequence and now - state.last_unlock_pulse_ts >= 0.8:
        # FW-mini can drift back to a non-driving state unless unlock is
        # reasserted periodically while motion commands are active.
        state.queue_unlock_sequence()
        state.last_unlock_pulse_ts = now

    body_vx, body_vy, body_wz = vx, vy, wz
    if gear == 'crab':
        # Match the stable follow/web control path: pure lateral motion is
        # driven by a single body CtrlCmd in gear 8. Sending a parallel
        # steering command here makes this chassis drop out of crab and hunt.
        controller.send_body(
            BodyCommand(
                gear=gear,
                vx=body_vx,
                vy=body_vy,
                wz=math.radians(body_wz),
            )
        )
    else:
        controller.send_body(BodyCommand(gear=gear, vx=body_vx, vy=body_vy, wz=math.radians(body_wz)))
        if gear != '4t4d':
            controller.send_steering(SteeringCommand(gear=gear, speed=0.0, angle=0.0))
    unlock_now = state.unlock_sequence.pop(0) if state.unlock_sequence else False
    if motion_active and gear in {'4t4d', 'crab'}:
        # Match the stable line-runner path: keep unlock asserted while
        # actively commanding motion so the chassis does not hold vx at zero.
        unlock_now = True
    force_io_send = unlock_now or bool(state.unlock_sequence)
    io_cmd = IOCommand(light_mode='auto', unlock=unlock_now)
    if io_cmd.active() or force_io_send:
        controller.send_io(io_cmd)
    if motion_active and now - state.last_cmd_at >= 1.0:
        log(
            f"send gear={gear} vx={body_vx:+.3f} vy={body_vy:+.3f} "
            f"wz={body_wz:+.2f} unlock={unlock_now} "
            f"fb_unlock={reported_unlock_ok} steer_sync={str(snapshot.get('steering', {}).get('gear', '-'))}"
        )
        state.last_cmd_at = now
    return waiting_unlock, unlock_now, body_vx, body_vy, body_wz


def _log_feedback(prefix: str, snapshot: dict[str, Any], pose: Pose2D) -> None:
    motion = snapshot.get('motion', {})
    steering = snapshot.get('steering', {})
    io_state = snapshot.get('io', {})
    err = snapshot.get('error', {})
    log(
        f"{prefix} feedback: body_gear={motion.get('gear', '--')} "
        f"steer_gear={steering.get('gear', '--')} "
        f"steer_speed={steering.get('wheel_speed_mps', '--')} "
        f"steer_angle={steering.get('wheel_angle_deg', '--')} "
        f"unlock_ok={io_state.get('unlock_ok', '--')} remote_control={io_state.get('remote_control', '--')} "
        f"estop={io_state.get('estop', '--')} error={err.get('level', '--')}/{err.get('type', '--')}"
    )
    log(f"{prefix} pose: x={pose.x:.3f} y={pose.y:.3f} yaw_deg={math.degrees(pose.yaw):.1f}")


def _hold_current_gear_stop(
    controller: FWMiniController,
    state: MotionSendState,
    gear: str | None,
) -> None:
    hold_gear = gear if gear in {'4t4d', 'crab'} else '4t4d'
    controller.send_body(BodyCommand(gear=hold_gear, vx=0.0, vy=0.0, wz=0.0))
    controller.send_steering(SteeringCommand(gear=hold_gear, speed=0.0, angle=0.0))
    state.last_sent_gear = hold_gear
    state.last_vx = 0.0
    state.last_vy = 0.0
    state.last_wz = 0.0
    state.last_cmd_at = 0.0


def _approach_start_point(
    tracker: PoseTracker,
    controller: FWMiniController,
    can_reader: CANFeedbackReader,
    points: list[Pose2D],
    index: int,
    send_state: MotionSendState,
) -> None:
    log(f'Navigating to mission start sample #{index}.')
    last_log = 0.0
    warmup_until = time.monotonic() + 0.5
    target_index = index
    target = points[target_index]
    best_dist = float('inf')
    stuck_count = 0
    last_vx_sign = 0
    flip_count = 0
    while not STOP_REQUESTED:
        pose = tracker.lookup()
        if pose is None:
            time.sleep(0.05)
            continue
        can_reader.poll(timeout=0.0, limit=20)
        snapshot = can_reader.snapshot()
        dist = pose.distance_to(target)
        yaw_err = normalize_angle(target.yaw - pose.yaw)
        if dist < 0.26 or (dist < 0.18 and abs(math.degrees(yaw_err)) < 30.0):
            log('Reached mission start point.')
            return
        if dist + 0.02 < best_dist:
            best_dist = dist
            stuck_count = 0
        else:
            stuck_count += 1
        forward_err, lateral_err = _body_frame_error(pose, target)
        heading_ref = math.atan2(target.y - pose.y, target.x - pose.x) if dist > 1e-6 else target.yaw
        heading_err = normalize_angle(heading_ref - pose.yaw)
        heading_err_deg, lateral_err = _smooth_tracking_errors(
            send_state,
            math.degrees(heading_err),
            lateral_err,
            alpha=0.25,
        )
        if abs(heading_err_deg) < 3.0:
            heading_err_deg = 0.0
        if abs(lateral_err) < 0.025:
            lateral_err = 0.0
        gear = '4t4d'
        vx = _clamp(forward_err * 0.72, -0.15, 0.15)
        vy = 0.0
        wz = _clamp(heading_err_deg * 0.72 + lateral_err * 13.0, -14.0, 14.0)
        if dist < 0.30:
            wz = _clamp(wz, -8.0, 8.0)
        if abs(heading_err_deg) > 25.0:
            vx = _clamp(vx, -0.05, 0.05)
        if abs(heading_err_deg) > 55.0:
            vx = _signed_crawl(forward_err, 0.025)
        vx_sign = 1 if vx > 0.01 else (-1 if vx < -0.01 else 0)
        if vx_sign and last_vx_sign and vx_sign != last_vx_sign:
            flip_count += 1
        last_vx_sign = vx_sign or last_vx_sign
        if stuck_count >= 8 and flip_count >= 2 and target_index + 3 < len(points):
            target_index = min(len(points) - 1, target_index + 3)
            target = points[target_index]
            best_dist = float('inf')
            stuck_count = 0
            flip_count = 0
            last_vx_sign = 0
            send_state.filtered_heading_err_deg = 0.0
            send_state.filtered_lateral_err_m = 0.0
            log(f'Approach fallback: skipping ahead to mission sample #{target_index} after local oscillation.')
            continue
        wz = _limit_4t4d_turn_rate(vx, wz)
        if time.monotonic() < warmup_until:
            vx = 0.0
            vy = 0.0
            wz = 0.0
        cmd_vx, cmd_vy, cmd_wz = _smooth_drive_command(send_state, gear, vx, vy, wz)
        waiting_unlock, unlock_now, sent_vx, sent_vy, sent_wz = _send_drive(
            controller, send_state, snapshot, gear, cmd_vx, cmd_vy, cmd_wz
        )
        now = time.monotonic()
        if now - last_log >= 1.0:
            log(
                f'Approach command: gear={gear} target_vx={cmd_vx:.2f} target_vy={cmd_vy:.2f} '
                f'target_wz={cmd_wz:.1f} sent_vx={sent_vx:.2f} sent_vy={sent_vy:.2f} '
                f'sent_wz={sent_wz:.1f} target_index={target_index} dist={dist:.2f}'
            )
            if waiting_unlock or unlock_now:
                log(f'Approach unlock: waiting_unlock={waiting_unlock} unlock_pulse={unlock_now}')
            _log_feedback('Approach', snapshot, pose)
            last_log = now
        time.sleep(0.05)


def cmd_map(args: argparse.Namespace) -> int:
    map_name = args.map_name or generated_name('map')
    log_path = LOG_DIR / f'mapping_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    proc = ManagedProcess(
        [
            ROS_PYTHON,
            str(RTABMAP_ROOT / 'headless_rtabmap.py'),
            'slam',
            '--map-name', map_name,
            '--rtabmap-viz', args.rtabmap_viz,
            '--record-mode', 'none',
        ],
        RTABMAP_ROOT,
        log_path,
    )
    log(f'Starting mapping session: {map_name}')
    proc.start()
    log(f'Mapping raw log: {log_path}')
    try:
        while not STOP_REQUESTED:
            code = proc.poll()
            if code is not None:
                return code
            time.sleep(0.2)
        return 0
    finally:
        proc.stop()


def cmd_localization(args: argparse.Namespace) -> int:
    session: LocalizationSession | None = None
    try:
        log('Waiting for localization to succeed.')
        session, _tracker, _pose = localize_with_retry(
            Path(args.db),
            args.map_frame,
            args.base_frame,
            'autorun_localization',
            args.localization_wait_sec,
            'Localization',
        )
        log('Localization succeeded. Keep observing the chassis state. Press Stop Current Task to end this test.')
        while not STOP_REQUESTED:
            time.sleep(0.2)
        return 0
    except LocalizationCancelled:
        log('Localization was cancelled.')
        return 0
    finally:
        if session is not None:
            session.stop()


def cmd_record(args: argparse.Namespace) -> int:
    ensure_can_ready(args.channel, args.bitrate)
    mission_name = args.mission_name or generated_name('mission')
    projection = projection_from_args(args)
    can_reader = CANFeedbackReader(args.interface, args.channel, args.bitrate)
    samples: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    start_pose: Pose2D | None = None
    last_sample_at = 0.0
    session: LocalizationSession | None = None
    tracker: PoseTracker | None = None
    try:
        if getattr(args, 'reuse_localization', False):
            log('Path recording requested. Reusing the active localization session.')
            tracker = create_tracker_with_retry(
                args.map_frame,
                args.base_frame,
                'autorun_record_reuse',
                args.localization_wait_sec,
                'Path recording',
            )
            start_pose_raw = wait_for_stable_pose(tracker, None, args.localization_wait_sec, 'Path recording')
        else:
            log('Path recording requested. Waiting for localization to succeed first.')
            session, tracker, start_pose_raw = localize_with_retry(
                Path(args.db),
                args.map_frame,
                args.base_frame,
                'autorun_record',
                args.localization_wait_sec,
                'Path recording',
            )
        projection = anchor_projection_to_pose(projection, start_pose_raw)
        start_pose = project_pose_to_ground(start_pose_raw, projection)
        log('Localization succeeded. Path recording is active now. Drive the vehicle with the Xbox controller.')
        while not STOP_REQUESTED:
            assert tracker is not None
            pose_raw = tracker.lookup()
            can_reader.poll(timeout=0.02, limit=20)
            snapshot = can_reader.snapshot()
            now = time.monotonic()
            if pose_raw is not None and now - last_sample_at >= args.sample_period:
                pose = project_pose_to_ground(pose_raw, projection)
                motion = snapshot.get('motion', {})
                index = len(samples)
                sample = {
                    'index': index,
                    't': round(now, 3),
                    'pose': {'x': pose.x, 'y': pose.y, 'z': pose.z, 'yaw_deg': math.degrees(pose.yaw)},
                    'snapshot': snapshot,
                }
                samples.append(sample)
                rows.append({
                    'index': index,
                    't': round(now, 3),
                    'x': round(pose.x, 4),
                    'y': round(pose.y, 4),
                    'z': round(pose.z, 4),
                    'yaw_deg': round(math.degrees(pose.yaw), 3),
                    'gear': motion.get('gear', ''),
                    'vx': motion.get('vx_mps', ''),
                    'vy': motion.get('vy_mps', ''),
                    'wz': motion.get('wz_dps', ''),
                })
                last_sample_at = now
            time.sleep(0.02)
    finally:
        if session is not None:
            session.stop()
        can_reader.close()
    if len(samples) < 2:
        raise RuntimeError('No usable path was recorded. Move the vehicle after localization succeeds.')

    payload = {
        'mission_name': mission_name,
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'bound_map_db': str(Path(args.db).resolve()),
        'map_frame': args.map_frame,
        'base_frame': args.base_frame,
        'sample_period': args.sample_period,
        'ground_projection': asdict(projection),
        'start_pose': asdict(start_pose) if start_pose else None,
        'samples': samples,
    }
    motions, used_fallback, fix_count = _regularize_recorded_motion(samples, args.sample_period)
    for sample, motion in zip(payload['samples'], motions):
        sample.setdefault('snapshot', {})
        sample['snapshot']['motion'] = {
            'gear': motion['gear'],
            'vx_mps': motion['vx'],
            'vy_mps': motion['vy'],
            'wz_dps': motion['wz'],
        }
    for row, motion in zip(rows, motions):
        row['gear'] = motion['gear']
        row['vx'] = round(motion['vx'], 4)
        row['vy'] = round(motion['vy'], 4)
        row['wz'] = round(motion['wz'], 3)
    if used_fallback:
        log('Recording fallback: inferred missing motion data from the recorded path geometry.')
    if fix_count:
        log(f'Recording cleanup: regularized {fix_count} transition samples around gear changes.')
    json_path, csv_path = write_mission_files(
        mission_name,
        payload,
        rows,
        map_db=str(Path(args.db).resolve()),
    )
    log(f'Mission saved: {json_path}')
    log(f'Mission CSV saved: {csv_path}')
    return 0


def cmd_autorun(args: argparse.Namespace) -> int:
    ensure_can_ready(args.channel, args.bitrate)
    mission = json.loads(Path(args.mission).read_text(encoding='utf-8'))
    mission_projection = mission.get('ground_projection', {})
    projection = GroundProjection(
        enabled=bool(mission_projection.get('enabled', False)),
        sensor_height_m=float(mission_projection.get('sensor_height_m', getattr(args, 'sensor_height_m', 0.0)) or 0.0),
        body_x_offset_m=float(mission_projection.get('body_x_offset_m', getattr(args, 'body_x_offset_m', 0.0)) or 0.0),
        body_y_offset_m=float(mission_projection.get('body_y_offset_m', getattr(args, 'body_y_offset_m', 0.0)) or 0.0),
        anchor_roll_rad=float(mission_projection.get('anchor_roll_rad', 0.0) or 0.0),
        anchor_pitch_rad=float(mission_projection.get('anchor_pitch_rad', 0.0) or 0.0),
    )
    if not projection.enabled:
        projection = projection_from_args(args)
    bound_map_db = str(mission.get('bound_map_db') or '').strip()
    selected_map_db = str(Path(args.db).resolve())
    if bound_map_db and not same_map_identity(bound_map_db, selected_map_db):
        raise RuntimeError(
            'Mission map mismatch. '
            f'This mission was recorded on: {Path(bound_map_db).name}, '
            f'but the selected replay map is: {Path(selected_map_db).name}.'
        )
    samples = mission.get('samples', [])
    if len(samples) < 2:
        raise RuntimeError('Mission has too few samples.')

    controller = FWMiniController(args.interface, args.channel, args.bitrate)
    can_reader = CANFeedbackReader(args.interface, args.channel, args.bitrate)
    points = [_sample_pose(sample) for sample in samples]
    sample_period = float(mission.get('sample_period', 0.2) or 0.2)
    motions, used_fallback = _repair_missing_motion(samples, sample_period)
    crab_fix_count = _stabilize_crab_segments(motions)
    start_index = _choose_start_index(points, motions)
    current_gear: str | None = None
    send_state = MotionSendState.create()
    session: LocalizationSession | None = None
    tracker: PoseTracker | None = None
    try:
        if getattr(args, 'reuse_localization', False):
            log('Auto run requested. Reusing the active localization session.')
            tracker = create_tracker_with_retry(
                args.map_frame,
                args.base_frame,
                'autorun_follow_reuse',
                args.localization_wait_sec,
                'Auto run',
            )
            _pose_raw = wait_for_stable_pose(tracker, None, args.localization_wait_sec, 'Auto run')
        else:
            log('Auto run requested. Waiting for localization to succeed first.')
            session, tracker, _pose_raw = localize_with_retry(
                Path(args.db),
                args.map_frame,
                args.base_frame,
                'autorun_follow',
                args.localization_wait_sec,
                'Auto run',
            )
        if projection.enabled and abs(projection.anchor_roll_rad) < 1e-9 and abs(projection.anchor_pitch_rad) < 1e-9:
            projection = anchor_projection_to_pose(projection, _pose_raw)
        _pose = project_pose_to_ground(_pose_raw, projection)
        log('Localization succeeded. Auto run started.')
        if used_fallback:
            log('Auto run fallback: mission motion data was missing, using path-derived motion estimates.')
        if crab_fix_count:
            log(f'Auto run cleanup: stabilized {crab_fix_count} crab transition samples from the mission.')
        if soften_count:
            log(f'Auto run cleanup: softened {soften_count} low-speed turns from the mission.')
        assert tracker is not None
        _approach_start_point(tracker, controller, can_reader, points, start_index, send_state)
        log(f'Auto run starts replay from mission sample #{start_index}.')
        last_cmd_log = 0.0
        last_feedback_log = 0.0
        current_gear = None
        while not STOP_REQUESTED and start_index < len(points):
            pose_raw = tracker.lookup()
            if pose_raw is None:
                time.sleep(0.05)
                continue
            pose = project_pose_to_ground(pose_raw, projection)
            can_reader.poll(timeout=0.0, limit=20)
            snapshot = can_reader.snapshot()
            io_state = snapshot.get('io', {})
            if bool(io_state.get('remote_control', False)):
                if not send_state.remote_paused:
                    send_state.remote_paused = True
                    send_state.reset_motion()
                    log('Remote controller took over. Auto run is paused and progress is preserved.')
                time.sleep(0.10)
                continue
            if send_state.remote_paused:
                send_state.remote_paused = False
                send_state.reset_motion()
                nearest_resume = _find_tracking_index(points, start_index, pose, window=24)
                start_index = max(start_index, nearest_resume)
                current_gear = None
                log(f'Remote controller released. Auto run resumed from nearest mission sample #{start_index}.')
            nearest_index = _find_tracking_index(points, start_index, pose, window=14)
            start_index = max(start_index, nearest_index)
            motion_here = motions[min(start_index, len(motions) - 1)]
            reversing_here = float(motion_here.get('vx', 0.0) or 0.0) < -0.03
            if send_state.crab_locked_until >= start_index and send_state.crab_target_index >= 0:
                gear = 'crab'
                target_index = _select_crab_progress_target(
                    start_index,
                    send_state.crab_locked_until,
                    max(start_index + 1, send_state.crab_target_index),
                )
                target = points[target_index]
                motion = motions[target_index]
                if _resolve_replay_gear(motion, None) != 'crab':
                    send_state.crab_locked_until = -1
                    send_state.crab_target_index = -1
                    gear = _resolve_replay_gear(motion_here, current_gear)
                    base_lookahead = 2 if gear == 'crab' else (
                        3 if reversing_here else (4 if abs(float(motion_here.get('wz', 0.0))) < 10.0 else 3)
                    )
                    target_index = min(len(points) - 1, start_index + base_lookahead)
                    target = points[target_index]
                    motion = motions[target_index]
            else:
                base_lookahead = 2 if _resolve_replay_gear(motion_here, current_gear) == 'crab' else (
                    3 if reversing_here else (4 if abs(float(motion_here.get('wz', 0.0))) < 10.0 else 3)
                )
                target_index = min(len(points) - 1, start_index + base_lookahead)
                target = points[target_index]
                motion = motions[target_index]
                gear = _resolve_replay_gear(motion, current_gear)
            if gear != 'crab':
                upcoming_crab = _find_future_gear_start(motions, start_index, 'crab', limit=20)
                if upcoming_crab is not None:
                    crab_entry = points[upcoming_crab]
                    if pose.distance_to(crab_entry) <= 0.35:
                        crab_end = _find_gear_segment_end(motions, upcoming_crab, 'crab')
                        crab_active = _find_first_active_crab_index(motions, upcoming_crab, crab_end)
                        crab_active_end = _find_last_active_crab_index(motions, crab_active, crab_end)
                        send_state.crab_locked_until = crab_active_end
                        send_state.crab_target_index = crab_active
                        target_index = _select_crab_progress_target(
                            start_index,
                            crab_active_end,
                            crab_active,
                        )
                        target = points[target_index]
                        motion = motions[target_index]
                        gear = 'crab'
            if gear != current_gear:
                log(f'Auto run mode switched to {gear}.')
                current_gear = gear

            tracking_heading = _tracking_heading(points, motions, target_index, gear)
            yaw_err = normalize_angle(tracking_heading - pose.yaw)
            dist = pose.distance_to(target)
            if gear == 'crab':
                if dist + 0.03 < send_state.crab_best_dist:
                    send_state.crab_best_dist = dist
                    send_state.crab_diverge_count = 0
                elif dist > send_state.crab_best_dist + 0.20:
                    send_state.crab_diverge_count += 1
                else:
                    send_state.crab_diverge_count = max(0, send_state.crab_diverge_count - 1)
                if send_state.crab_diverge_count >= 6:
                    log(
                        f'Auto run crab fallback: target distance kept increasing '
                        f'(best={send_state.crab_best_dist:.2f}, now={dist:.2f}). Returning to 4t4d tracking.'
                    )
                    send_state.crab_locked_until = -1
                    send_state.crab_target_index = -1
                    send_state.crab_best_dist = float("inf")
                    send_state.crab_diverge_count = 0
                    current_gear = None
                    send_state.last_sent_gear = None
                    continue
                ref_point = points[start_index]
                axis_dx = target.x - ref_point.x
                axis_dy = target.y - ref_point.y
                if abs(axis_dy) >= abs(axis_dx):
                    if _axis_progress_reached(pose.y, target.y, axis_dy, 0.10):
                        start_index = min(len(points) - 1, target_index + 1)
                        if start_index > send_state.crab_locked_until:
                            send_state.crab_locked_until = -1
                            send_state.crab_target_index = -1
                            send_state.crab_best_dist = float("inf")
                            send_state.crab_diverge_count = 0
                        continue
                else:
                    if _axis_progress_reached(pose.x, target.x, axis_dx, 0.10):
                        start_index = min(len(points) - 1, target_index + 1)
                        if start_index > send_state.crab_locked_until:
                            send_state.crab_locked_until = -1
                            send_state.crab_target_index = -1
                            send_state.crab_best_dist = float("inf")
                            send_state.crab_diverge_count = 0
                        continue
            elif dist < 0.12 and abs(math.degrees(yaw_err)) < 12.0:
                send_state.crab_best_dist = float("inf")
                send_state.crab_diverge_count = 0
                start_index = min(len(points) - 1, target_index + 1)
                continue

            forward_err, lateral_err = _body_frame_error(pose, target)
            heading_ref = tracking_heading
            heading_err = normalize_angle(heading_ref - pose.yaw)
            near_finish = target_index >= len(points) - 4
            if near_finish and dist < 0.10:
                _hold_current_gear_stop(controller, send_state, current_gear)
                log(f'Auto run reached final area near mission end (dist={dist:.2f}). Stopping without final alignment.')
                break
            if gear != 'crab' and abs(lateral_err) > 0.07:
                target_index = min(len(points) - 1, start_index + 2)
                target = points[target_index]
                motion = motions[target_index]
                tracking_heading = _tracking_heading(points, motions, target_index, gear)
                forward_err, lateral_err = _body_frame_error(pose, target)
                heading_ref = tracking_heading
                heading_err = normalize_angle(heading_ref - pose.yaw)

            if gear in {'park', 'neutral'}:
                hold_gear = current_gear if current_gear in {'4t4d', 'crab'} else '4t4d'
                cmd_vx, cmd_vy, cmd_wz = _smooth_drive_command(send_state, hold_gear, 0.0, 0.0, 0.0)
                _send_drive(controller, send_state, snapshot, hold_gear, cmd_vx, cmd_vy, cmd_wz)
                time.sleep(max(sample_period, 0.05))
                start_index += 1
                continue

            reversing = float(motion.get('vx', 0.0) or 0.0) < -0.03
            vx_cap = max(abs(motion['vx']), 0.22)
            wz_cap = max(abs(motion['wz']), 20.0)
            heading_err_deg, lateral_err = _smooth_tracking_errors(
                send_state,
                math.degrees(heading_err),
                lateral_err,
                alpha=0.24,
            )
            if abs(heading_err_deg) < 2.0:
                heading_err_deg = 0.0
            if abs(lateral_err) < 0.02:
                lateral_err = 0.0
            cmd_vx = _clamp(forward_err * 0.75, -vx_cap, vx_cap)
            cmd_vy = 0.0
            if gear == 'crab':
                vy_cap = max(abs(float(motion.get('vy', 0.0) or 0.0)), 0.25)
                cmd_vx = 0.0
                # During row switching, prioritize a direct lateral translation
                # to the next row. Do not keep rotating the chassis while crabbing,
                # otherwise the wheel angle hunts and the vehicle stalls.
                cmd_vy = _clamp(lateral_err * 1.15, -vy_cap, vy_cap)
                if abs(lateral_err) < 0.04:
                    cmd_vy = 0.0
                cmd_wz = 0.0
            else:
                cmd_vx = _clamp(cmd_vx, -0.16, 0.16)
                if reversing:
                    cmd_wz = _clamp(
                        heading_err_deg * 0.72 + lateral_err * 18.0,
                        -min(wz_cap, 12.0),
                        min(wz_cap, 12.0),
                    )
                else:
                    if abs(lateral_err) > 0.05:
                        cmd_vx = _clamp(cmd_vx, -0.10, 0.10)
                    if abs(lateral_err) > 0.08:
                        cmd_vx = _clamp(cmd_vx, -0.07, 0.07)
                    if abs(lateral_err) > 0.11:
                        cmd_vx = _clamp(cmd_vx, -0.05, 0.05)
                    cmd_wz = _clamp(
                        heading_err_deg * 0.72 + lateral_err * 18.0,
                        -min(wz_cap, 12.0),
                        min(wz_cap, 12.0),
                    )
            if abs(heading_err_deg) > 45.0:
                cmd_vx = _clamp(cmd_vx, -0.04, 0.04)
            if abs(heading_err_deg) > 65.0:
                cmd_vx = _signed_crawl(forward_err, 0.025)
            if gear == '4t4d':
                cmd_wz = _limit_4t4d_turn_rate(cmd_vx, cmd_wz)

            if (
                gear == 'crab'
                and dist < 0.35
                and abs(lateral_err) < 0.25
                and target_index >= max(start_index + 1, send_state.crab_locked_until - 1)
            ):
                start_index = min(len(points) - 1, target_index + 1)
                continue

            cmd_vx, cmd_vy, cmd_wz = _smooth_drive_command(send_state, gear, cmd_vx, cmd_vy, cmd_wz)
            if gear == 'crab':
                send_state.last_vx = 0.0
                cmd_vx = 0.0
            waiting_unlock, unlock_now, sent_vx, sent_vy, sent_wz = _send_drive(
                controller, send_state, snapshot, gear, cmd_vx, cmd_vy, cmd_wz
            )
            now = time.monotonic()
            if now - last_cmd_log >= 1.0:
                log(
                    f'Auto run command: gear={gear} target_vx={cmd_vx:.2f} target_vy={cmd_vy:.2f} '
                    f'target_wz={cmd_wz:.1f} sent_vx={sent_vx:.2f} sent_vy={sent_vy:.2f} '
                    f'sent_wz={sent_wz:.1f} nearest_index={nearest_index} target_index={target_index} dist={dist:.2f}'
                )
                if waiting_unlock or unlock_now:
                    log(f'Auto run unlock: waiting_unlock={waiting_unlock} unlock_pulse={unlock_now}')
                if gear in {'4t4d', 'crab'} and send_state.active_gear_sync_count < 2:
                    log(
                        f'Auto run anti-park hold: waiting for stable {gear} feedback '
                        f'(stable_count={send_state.active_gear_sync_count}).'
                    )
                last_cmd_log = now
            if now - last_feedback_log >= 1.0:
                _log_feedback('Auto run', snapshot, pose)
                last_feedback_log = now
            time.sleep(max(0.03, min(0.10, sample_period)))

        _hold_current_gear_stop(controller, send_state, current_gear)
        log('Auto run finished.')
        return 0
    finally:
        try:
            _hold_current_gear_stop(controller, send_state, current_gear)
        except Exception:
            pass
        can_reader.close()
        controller.close()
        if session is not None:
            session.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Simple UGV drive backend rebuilt from control + rtabmap')
    parser.add_argument('--interface', default='socketcan')
    parser.add_argument('--channel', default='can0')
    parser.add_argument('--bitrate', type=int, default=500000)
    parser.add_argument('--sensor-height-m', type=float, default=DEFAULT_SENSOR_HEIGHT_M)
    parser.add_argument('--body-x-offset-m', type=float, default=DEFAULT_BODY_X_OFFSET_M)
    parser.add_argument('--body-y-offset-m', type=float, default=DEFAULT_BODY_Y_OFFSET_M)
    parser.add_argument('--roll-gain', type=float, default=DEFAULT_ROLL_GAIN)
    parser.add_argument('--pitch-gain', type=float, default=DEFAULT_PITCH_GAIN)
    sub = parser.add_subparsers(dest='command', required=True)

    map_p = sub.add_parser('map', help='Start mapping')
    map_p.add_argument('--map-name', default='')
    map_p.add_argument('--rtabmap-viz', choices=['on', 'off'], default='off')
    map_p.set_defaults(func=cmd_map)

    loc_p = sub.add_parser('localization', help='Run localization only')
    loc_p.add_argument('--db', required=True)
    loc_p.add_argument('--map-frame', default='map')
    loc_p.add_argument('--base-frame', default='base_link')
    loc_p.add_argument('--localization-wait-sec', type=float, default=30.0)
    loc_p.set_defaults(func=cmd_localization)

    rec_p = sub.add_parser('record', help='Localize first, then record a mission path')
    rec_p.add_argument('--db', required=True)
    rec_p.add_argument('--mission-name', default='')
    rec_p.add_argument('--map-frame', default='map')
    rec_p.add_argument('--base-frame', default='base_link')
    rec_p.add_argument('--sample-period', type=float, default=0.2)
    rec_p.add_argument('--localization-wait-sec', type=float, default=30.0)
    rec_p.add_argument('--reuse-localization', action='store_true')
    rec_p.set_defaults(func=cmd_record)

    run_p = sub.add_parser('drive', help='Localize, navigate to the recorded path start, then drive the taught mission')
    run_p.add_argument('--db', required=True)
    run_p.add_argument('--mission', required=True)
    run_p.add_argument('--map-frame', default='map')
    run_p.add_argument('--base-frame', default='base_link')
    run_p.add_argument('--localization-wait-sec', type=float, default=30.0)
    run_p.add_argument('--reuse-localization', action='store_true')
    run_p.set_defaults(func=cmd_autorun)

    return parser


def main() -> int:
    install_signal_handlers()
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == '__main__':
    raise SystemExit(main())
