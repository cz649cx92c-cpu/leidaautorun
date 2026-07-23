#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CONTROL_ROOT = Path(__file__).resolve().parent.parent / "control"
if str(CONTROL_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTROL_ROOT))

from fw_mini_controller import BodyCommand, FWMiniController, IOCommand, SteeringCommand, auto_park  # noqa: E402
from fw_mini_status_reader import build_snapshot, decode_msg, open_can_bus  # noqa: E402
from row_geometry import (  # noqa: E402
    ROBOT_FRAME_BACK,
    RowEstimate,
    RowFollowerConfig,
    blend_line,
    estimate_row,
)

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from sensor_msgs.msg import LaserScan  # noqa: E402
from std_msgs.msg import String  # noqa: E402


@dataclass
class MotionSendState:
    unlock_sequence: list[bool]
    unlock_request_active: bool = False
    motion_unlock_armed: bool = True
    unlock_confirmed_until: float = 0.0
    motion_unlocked_session: bool = False
    unlock_force_started_at: float = 0.0
    last_unlock_pulse_ts: float = 0.0
    last_cmd_log_ts: float = 0.0
    last_sent_gear: str | None = None
    startup_unlock_pending: int = 8

    @classmethod
    def create(cls) -> "MotionSendState":
        return cls(unlock_sequence=[])

    def queue_unlock_sequence(self) -> None:
        sequence = [True, True, False, False]
        if not self.unlock_sequence:
            self.unlock_sequence = sequence.copy()
        else:
            self.unlock_sequence.extend(sequence)


def _normalize_feedback_gear(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    mapping = {
        "0": "neutral",
        "1": "park",
        "2": "neutral",
        "6": "4t4d",
        "8": "crab",
    }
    return mapping.get(text, text)


def _gear_code_for_log(gear: str) -> int:
    normalized = str(gear or "").strip().lower()
    if normalized == "crab":
        return 8
    if normalized in {"4t4d", "4wd", "6"}:
        return 6
    return 2


@dataclass
class ControlState:
    body: BodyCommand
    steering: SteeringCommand | None
    io: IOCommand
    sequence: int = 0


def _history_median(history: list[tuple[float, float]]) -> float:
    if not history:
        return 0.0
    return float(np.median(np.asarray([value for _ts, value in history], dtype=np.float64)))


class CommandSender:
    def __init__(
        self,
        interface: str,
        channel: str,
        bitrate: int,
        period_s: float,
        initial_state: ControlState,
        startup_unlock_cycles: int = 0,
        command_timeout_s: float = 0.0,
    ) -> None:
        self.interface = interface
        self.channel = channel
        self.bitrate = int(bitrate)
        self.period_s = float(period_s)
        self.command_timeout_s = max(0.0, float(command_timeout_s))
        self.startup_unlock_cycles = max(0, int(startup_unlock_cycles))
        self._unlock_pending = self.startup_unlock_cycles
        self._state = initial_state
        self._last_update_ts = time.monotonic()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._controller: FWMiniController | None = None
        self._error: BaseException | None = None
        self._feedback: dict[str, Any] = {}
        self._unlock_sequence: list[bool] = []
        self._unlock_confirmed_until = 0.0
        self._motion_unlocked_session = False
        self._unlock_request_active = False
        self._motion_unlock_armed = True
        self._unlock_force_started_at = 0.0
        self._last_unlock_pulse_ts = 0.0
        self._last_sent_gear: str | None = None
        self._last_motion_log_ts = 0.0
        self._active_gear_sync_count = 0
        self._runtime_state: dict[str, Any] = {
            "waiting_unlock": False,
            "motion_active": False,
            "effective_unlock_ok": False,
            "gear_sync_wait": False,
            "gear_sync_count": 0,
            "sent_body_vx": 0.0,
            "sent_body_vy": 0.0,
            "sent_body_wz": 0.0,
            "steering_cmd_present": False,
            "send_steering_speed": 0.0,
            "unlock_now": False,
            "unlock_fallback": False,
            "unlock_wait_s": 0.0,
            "command_stale": False,
            "command_age_s": 0.0,
        }
        self._stop_requested = False

    def _log_runtime(self, message: str) -> None:
        print(f"[can] {message}", flush=True)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="fwmini-can-sender", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=3.0)
        if self._error is not None:
            raise RuntimeError(f"failed to start CAN sender: {self._error}") from self._error

    def update(self, body: BodyCommand, steering: SteeringCommand | None, io: IOCommand) -> None:
        with self._lock:
            self._state = ControlState(body=body, steering=steering, io=io, sequence=self._state.sequence + 1)
            self._last_update_ts = time.monotonic()

    def request_unlock(self, cycles: int = 8) -> None:
        with self._lock:
            self._unlock_pending = max(self._unlock_pending, int(cycles))

    def feedback_snapshot(self) -> dict[str, Any]:
        with self._lock:
            snapshot = dict(self._feedback)
            snapshot["_runtime"] = dict(self._runtime_state)
            return snapshot

    def _queue_unlock_sequence(self, repeat: int = 1) -> None:
        sequence = [True, True, False, False] * max(1, int(repeat))
        if not self._unlock_sequence:
            self._unlock_sequence = sequence.copy()
        else:
            self._unlock_sequence.extend(sequence)

    def _run(self) -> None:
        try:
            self._log_runtime("CommandSender thread starting")
            controller = FWMiniController(self.interface, self.channel, self.bitrate)
            self._controller = controller
            self._ready.set()
            next_tick = time.monotonic()
            while not self._stop.is_set():
                with self._lock:
                    state = self._state
                    command_age_s = time.monotonic() - self._last_update_ts
                    if self._unlock_pending > 0:
                        self._queue_unlock_sequence(max(1, self._unlock_pending // 4))
                        self._unlock_pending = 0

                command_stale = self.command_timeout_s > 0.0 and command_age_s > self.command_timeout_s
                if command_stale and state.body.gear not in {"park", "neutral"}:
                    state = ControlState(
                        body=BodyCommand(gear=state.body.gear, vx=0.0, vy=0.0, wz=0.0),
                        steering=state.steering,
                        io=state.io,
                        sequence=state.sequence,
                    )

                io_feedback = self._feedback.get("io_fb", {})
                gear = state.body.gear
                steering_feedback = self._feedback.get("steering_ctrl_fb", {})
                if gear != self._last_sent_gear:
                    self._log_runtime(f"gear request {self._last_sent_gear}->{gear}")
                    controller.send_body(BodyCommand(gear=gear, vx=0.0, vy=0.0, wz=0.0))
                    if state.steering is not None:
                        controller.send_steering(SteeringCommand(gear=gear, speed=0.0, angle=state.steering.angle))
                    self._last_sent_gear = gear
                    self._active_gear_sync_count = 0

                reported_unlock_ok = bool(io_feedback.get("unlock_ok", False))
                now = time.monotonic()
                self._active_gear_sync_count = 1000
                if reported_unlock_ok:
                    self._unlock_confirmed_until = now + 2.0
                    self._motion_unlocked_session = True
                    self._unlock_request_active = False
                    self._unlock_force_started_at = 0.0
                effective_unlock_ok = (
                    self._motion_unlocked_session
                    or reported_unlock_ok
                    or now < self._unlock_confirmed_until
                )
                motion_active = state.body.gear not in {"park", "neutral"} and any(
                    abs(value) > 1e-6 for value in (state.body.vx, state.body.vy, state.body.wz)
                )
                waiting_unlock = motion_active and not effective_unlock_ok
                if motion_active and self._motion_unlock_armed:
                    self._unlock_request_active = True
                    self._queue_unlock_sequence()
                    self._motion_unlock_armed = False
                    self._unlock_force_started_at = now
                    self._last_unlock_pulse_ts = now
                elif self._unlock_request_active and not effective_unlock_ok and not self._unlock_sequence:
                    self._queue_unlock_sequence()
                    self._last_unlock_pulse_ts = now
                elif not motion_active:
                    self._motion_unlock_armed = True
                    self._unlock_force_started_at = 0.0
                    self._last_unlock_pulse_ts = 0.0

                unlock_wait_s = 0.0
                unlock_fallback = False
                if motion_active and not effective_unlock_ok:
                    if self._unlock_force_started_at <= 0.0:
                        self._unlock_force_started_at = now
                    unlock_wait_s = now - self._unlock_force_started_at
                    if unlock_wait_s >= 0.6:
                        effective_unlock_ok = True
                        waiting_unlock = False
                        unlock_fallback = True
                        self._motion_unlocked_session = True

                gear_sync_wait = False
                if motion_active and not self._unlock_sequence and now - self._last_unlock_pulse_ts >= 0.8:
                    self._queue_unlock_sequence()
                    self._last_unlock_pulse_ts = now

                body_cmd = state.body
                steering_cmd = state.steering
                controller.send_body(body_cmd)
                if steering_cmd is not None and steering_cmd.gear not in {"4t4d", "crab"}:
                    controller.send_steering(steering_cmd)

                io_cmd = state.io
                unlock_now = self._unlock_sequence.pop(0) if self._unlock_sequence else False
                if motion_active and state.body.gear in {"4t4d", "crab"}:
                    unlock_now = True
                if unlock_now:
                    io_cmd = IOCommand(
                        light_mode=state.io.light_mode,
                        unlock=True,
                        low_beam=state.io.low_beam,
                        turn=state.io.turn,
                        brake=state.io.brake,
                    )
                if io_cmd.active() or unlock_now or bool(self._unlock_sequence):
                    controller.send_io(io_cmd)
                if motion_active and now - self._last_motion_log_ts >= 1.0:
                    self._log_runtime(
                        "send "
                        f"gear={body_cmd.gear} vx={body_cmd.vx:+.3f} "
                        f"wz={body_cmd.wz:+.2f} unlock={unlock_now} "
                        f"fb_unlock={reported_unlock_ok} steer_sync={str(steering_feedback.get('gear', '-'))}"
                    )
                    self._last_motion_log_ts = now

                with self._lock:
                    self._runtime_state = {
                        "waiting_unlock": waiting_unlock,
                        "motion_active": motion_active,
                        "effective_unlock_ok": effective_unlock_ok,
                        "gear_sync_wait": gear_sync_wait,
                        "gear_sync_count": self._active_gear_sync_count,
                        "sent_body_vx": body_cmd.vx,
                        "sent_body_vy": body_cmd.vy,
                        "sent_body_wz": body_cmd.wz,
                        "steering_cmd_present": steering_cmd is not None,
                        "send_steering_angle": None if steering_cmd is None else float(steering_cmd.angle),
                        "send_steering_speed": 0.0 if steering_cmd is None else float(steering_cmd.speed),
                        "send_steering_zeroed": bool(steering_cmd is not None and abs(float(steering_cmd.angle)) < 1e-6),
                        "unlock_now": unlock_now,
                        "unlock_fallback": unlock_fallback,
                        "unlock_wait_s": unlock_wait_s,
                        "command_stale": command_stale,
                        "command_age_s": command_age_s,
                    }

                msgs = controller.poll(limit=10)
                if msgs:
                    with self._lock:
                        for msg in msgs:
                            self._feedback[msg["name"]] = msg["data"]
                next_tick += self.period_s
                sleep_s = next_tick - time.monotonic()
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
                else:
                    next_tick = time.monotonic()
        except BaseException as exc:
            self._log_runtime(f"CommandSender error: {exc!r}")
            self._error = exc
            self._ready.set()
        finally:
            self._log_runtime(f"CommandSender thread exiting stop_requested={self._stop_requested}")
            controller = self._controller
            if controller is not None:
                try:
                    if self._stop_requested:
                        auto_park(controller, self.period_s)
                finally:
                    controller.close()

    def stop(self) -> None:
        self._stop_requested = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._error is not None:
            raise RuntimeError(f"CAN sender stopped with error: {self._error}") from self._error


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
                self.latest[decoded["name"]] = decoded["data"]
            count += 1
        return self.latest

    def snapshot(self) -> dict[str, Any]:
        return build_snapshot(self.latest) if self.latest else {}

    def close(self) -> None:
        self.bus.shutdown()


class PlantRowFollower(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("plant_lidar_centerline_follower")
        self.args = args
        self.scan_lock = threading.Lock()
        self.last_scan: LaserScan | None = None
        self.last_scan_time = 0.0
        self.last_estimate = RowEstimate(found=False)
        self.last_good_row_width = float(args.row_width)
        self.last_found_time = 0.0
        self.last_good_time = 0.0
        self.last_filtered_center_y = 0.0
        self.last_good_center_y = 0.0
        self.last_error_y = 0.0
        self.last_direct_error_y = 0.0
        self.last_good_heading_deg = 0.0
        self.direct_error_history: list[tuple[float, float]] = []
        self.last_cmd_vx = 0.0
        self.last_cmd_wz = 0.0
        self.last_steering_angle = 0.0
        self.last_steering_feedback_angle = 0.0
        self.wz_not_following_count = 0
        self.last_debug: dict[str, Any] = {}
        self.last_good_cmd = BodyCommand(gear=args.gear, vx=0.0, vy=0.0, wz=0.0)
        self.last_good_estimate = RowEstimate(found=False)
        self.last_good_center_line: tuple[float, float] | None = None
        self.last_good_left_line: tuple[float, float] | None = None
        self.last_good_right_line: tuple[float, float] | None = None
        self.last_good_mode = ""
        self.last_following_wz_deg = 0.0
        self.send_state = MotionSendState.create()
        self.row_cfg = RowFollowerConfig(
            row_width=float(args.row_width),
            min_row_width=float(args.min_row_width),
            max_row_width=float(args.max_row_width),
            lookahead_x=float(args.lookahead_x),
            forward_min=float(args.forward_min),
            forward_max=float(args.forward_max),
            lateral_limit=float(args.lateral_limit),
            range_min=float(args.range_min),
            range_max=float(args.range_max),
            bin_size=float(args.bin_size),
            min_points=int(args.min_points),
            min_bins=int(args.min_bins),
            min_line_bins=int(args.min_line_bins),
            min_side_points_per_bin=int(args.min_side_points_per_bin),
            center_deadband=float(args.center_deadband),
            left_percentile=float(args.left_percentile),
            right_percentile=float(args.right_percentile),
            sensor_yaw_deg=float(args.sensor_yaw_deg),
            lidar_yaw_correction_deg=float(args.lidar_yaw_correction_deg),
            lidar_x_offset_m=float(args.lidar_x_offset_m),
            lidar_y_offset_m=float(args.lidar_y_offset_m),
            boundary_max_gap_x=0.45,
            boundary_width_tolerance_m=float(args.boundary_width_tolerance_m),
            vehicle_half_width=0.5 * float(args.vehicle_width),
            safety_margin=float(args.safety_margin),
            center_jump_reject=float(args.center_jump_reject),
            one_side_center_jump_reject=float(args.one_side_center_jump_reject),
        )
        initial_gear = self._normalized_gear()
        self.sender = CommandSender(
            args.interface,
            args.channel,
            args.bitrate,
            period_s=max(0.02, float(args.control_period)),
            initial_state=ControlState(
                body=BodyCommand(gear=initial_gear, vx=0.0, vy=0.0, wz=0.0),
                steering=SteeringCommand(gear=initial_gear, speed=0.0, angle=self.last_steering_angle),
                io=IOCommand(light_mode="free" if args.low_beam else "auto", low_beam=bool(args.low_beam)),
            ),
            startup_unlock_cycles=8,
            command_timeout_s=0.6,
        )
        self.sender.start()
        self.can_reader = CANFeedbackReader(args.interface, args.channel, args.bitrate)
        self.status_pub = self.create_publisher(String, args.status_topic, 10)
        self.create_subscription(LaserScan, args.scan_topic, self._on_scan, 10)
        self.control_timer = self.create_timer(max(0.02, float(args.control_period)), self._on_control)
        self.create_timer(max(0.1, float(args.status_period)), self._publish_status)
        self._stopping = False
        self.get_logger().info(
            f"started scan_topic={args.scan_topic} speed={args.speed:.2f} row_width={args.row_width:.2f}"
        )

    def clear_motion_history(self) -> None:
        self.last_found_time = 0.0
        self.last_good_time = 0.0
        self.last_filtered_center_y = 0.0
        self.last_good_center_y = 0.0
        self.last_error_y = 0.0
        self.last_direct_error_y = 0.0
        self.last_good_heading_deg = 0.0
        self.direct_error_history.clear()
        self.last_cmd_vx = 0.0
        self.last_cmd_wz = 0.0
        self.last_good_cmd = BodyCommand(gear=self.args.gear, vx=0.0, vy=0.0, wz=0.0)
        self.last_good_estimate = RowEstimate(found=False)
        self.last_good_center_line = None
        self.last_good_left_line = None
        self.last_good_right_line = None
        self.last_good_mode = ""
        self.last_following_wz_deg = 0.0
        self.wz_not_following_count = 0
        self.last_debug = {}

    def _on_scan(self, msg: LaserScan) -> None:
        with self.scan_lock:
            self.last_scan = msg
            self.last_scan_time = time.monotonic()

    def _estimate_row(self, scan: LaserScan) -> RowEstimate:
        estimate, _debug = estimate_row(scan, self.row_cfg, self.last_good_row_width)
        if not estimate.found:
            return estimate

        self.last_good_row_width = float(self.args.row_width)
        valid_mode = str(getattr(estimate, "effective_mode", estimate.mode) or "")
        raw_center_y = float(estimate.raw_center_y)
        raw_center_limit = (
            float(self.args.one_side_raw_center_out_of_range)
            if valid_mode in {"left_only", "right_only"}
            else float(self.args.raw_center_out_of_range)
        )
        if valid_mode in {"left_only", "right_only"} and abs(raw_center_y) > float(self.args.raw_center_out_of_range):
            estimate.warning = "large_one_side_center"
        if abs(raw_center_y) > raw_center_limit:
            estimate.found = False
            estimate.mode = "reject"
            estimate.reject_reason = "raw_center_out_of_range"
            return estimate
        warning = str(getattr(estimate, "warning", "") or "")
        if abs(raw_center_y) > 0.20:
            estimate.found = False
            estimate.mode = "reject"
            estimate.reject_reason = "raw_center_out_of_range"
            return estimate

        same_mode = valid_mode == self.last_good_mode
        keep_ratio = 0.0
        estimate.left_line = blend_line(self.last_good_left_line, getattr(estimate, "left_line", None), keep_ratio=keep_ratio)
        estimate.right_line = blend_line(self.last_good_right_line, getattr(estimate, "right_line", None), keep_ratio=keep_ratio)
        estimate.center_line = blend_line(self.last_good_center_line, getattr(estimate, "center_line", None), keep_ratio=keep_ratio)

        if estimate.center_line is not None:
            reference_x = max(
                float(self.row_cfg.forward_min),
                min(float(self.row_cfg.forward_max), float(self.row_cfg.lookahead_x)),
            )
            center_y_ref = float(estimate.center_line[0] * reference_x + estimate.center_line[1])
            estimate.center_y = center_y_ref
            estimate.raw_center_y = center_y_ref
            estimate.heading_rad = float(math.atan(estimate.center_line[0]))

        center_jump = abs(float(estimate.center_y) - float(self.last_good_center_y)) if self.last_good_time > 0.0 else 0.0
        heading_deg = math.degrees(float(estimate.heading_rad))
        heading_jump = abs(heading_deg - float(self.last_good_heading_deg)) if self.last_good_time > 0.0 else 0.0
        estimate.center_jump = center_jump
        estimate.heading_jump_deg = heading_jump

        jump_reject_enabled = same_mode and self.last_good_center_line is not None
        if jump_reject_enabled and center_jump > 0.06:
            estimate.center_jump_rejected = True
            estimate.warning = "center_jump_rejected"
            estimate.center_line = self.last_good_center_line
            estimate.left_line = self.last_good_left_line
            estimate.right_line = self.last_good_right_line
            estimate.center_y = float(self.last_good_center_y)
            estimate.raw_center_y = float(self.last_good_center_y)
            estimate.heading_rad = math.radians(float(self.last_good_heading_deg))
        elif jump_reject_enabled and heading_jump > 8.0:
            estimate.heading_jump_rejected = True
            estimate.warning = "heading_jump_rejected"
            estimate.center_line = self.last_good_center_line
            estimate.left_line = self.last_good_left_line
            estimate.right_line = self.last_good_right_line
            estimate.center_y = float(self.last_good_center_y)
            estimate.raw_center_y = float(self.last_good_center_y)
            estimate.heading_rad = math.radians(float(self.last_good_heading_deg))

        filtered_center_limit = float(self.args.center_y_reject_abs)
        if valid_mode in {"left_only", "right_only"}:
            filtered_center_limit = max(filtered_center_limit, raw_center_limit)
        if abs(float(estimate.center_y)) > filtered_center_limit:
            estimate.found = False
            estimate.mode = "reject"
            estimate.reject_reason = "filtered_center_out_of_range"
            return estimate

        estimate.warning = warning
        self.last_filtered_center_y = float(estimate.center_y)
        self.last_good_center_y = float(estimate.center_y)
        return estimate

    def _set_debug_snapshot(
        self,
        *,
        estimate: RowEstimate,
        found: bool,
        control_phase: str,
        stop_reason: str,
        final_vx: float,
        final_wz: float,
        center_y_target: float | None = None,
        error_y: float = 0.0,
        heading_error: float = 0.0,
        warning: str = "",
        **extra: Any,
    ) -> None:
        mode = str(estimate.mode or "")
        effective_mode = str(getattr(estimate, "effective_mode", estimate.mode) or "")
        payload: dict[str, Any] = {
            "found": found,
            "control_phase": control_phase,
            "reverse": bool(self.args.reverse),
            "calibrated": True,
            "sensor_yaw_deg": float(self.args.sensor_yaw_deg),
            "lidar_yaw_correction_deg": float(self.args.lidar_yaw_correction_deg),
            "lidar_x_offset_m": float(self.args.lidar_x_offset_m),
            "lidar_y_offset_m": float(self.args.lidar_y_offset_m),
            "raw_center_y": float(getattr(estimate, "raw_center_y", 0.0) or 0.0),
            "filtered_center_y": float(getattr(estimate, "center_y", 0.0) or 0.0),
            "center_y_target": float(center_y_target if center_y_target is not None else float(self.args.center_y_target)),
            "error_y": float(error_y),
            "heading_deg": math.degrees(float(heading_error)),
            "mode": mode,
            "effective_mode": effective_mode,
            "left_bins": int(getattr(estimate, "left_bins", 0) or 0),
            "right_bins": int(getattr(estimate, "right_bins", 0) or 0),
            "candidate_bins": int(getattr(estimate, "candidate_bins", 0) or 0),
            "final_vx": float(final_vx),
            "final_wz_rad": float(final_wz),
            "final_wz_deg": math.degrees(float(final_wz)),
            "stop_reason": stop_reason,
            "warning": warning or str(getattr(estimate, "warning", "") or ""),
            "reject_reason": str(getattr(estimate, "reject_reason", "") or ""),
            "row_width": float(getattr(estimate, "row_width", 0.0) or 0.0),
            "raw_points_count": int(getattr(estimate, "raw_points_count", 0) or 0),
            "filtered_points_count": int(getattr(estimate, "filtered_points_count", 0) or 0),
            "left_points_count": int(getattr(estimate, "left_points_count", 0) or 0),
            "right_points_count": int(getattr(estimate, "right_points_count", 0) or 0),
            "center_y_jump": float(getattr(estimate, "center_jump", 0.0) or 0.0),
            "heading_jump": float(getattr(estimate, "heading_jump_deg", 0.0) or 0.0),
            "center_jump_rejected": bool(getattr(estimate, "center_jump_rejected", False)),
            "heading_jump_rejected": bool(getattr(estimate, "heading_jump_rejected", False)),
            "left_valid": bool(getattr(estimate, "left_valid", False)),
            "right_valid": bool(getattr(estimate, "right_valid", False)),
            "left_reject_reason": str(getattr(estimate, "left_reject_reason", "") or ""),
            "right_reject_reason": str(getattr(estimate, "right_reject_reason", "") or ""),
            "parallel_angle_diff_deg": float(getattr(estimate, "parallel_angle_diff_deg", 0.0) or 0.0),
            "width_error_m": float(getattr(estimate, "width_error_m", 0.0) or 0.0),
            "boundary_width_tolerance_m": float(getattr(self.row_cfg, "boundary_width_tolerance_m", 0.0) or 0.0),
            "left_residual_median": float(getattr(estimate, "left_residual_median", 0.0) or 0.0),
            "right_residual_median": float(getattr(estimate, "right_residual_median", 0.0) or 0.0),
            "left_consecutive_bins": int(getattr(estimate, "left_consecutive_bins", 0) or 0),
            "right_consecutive_bins": int(getattr(estimate, "right_consecutive_bins", 0) or 0),
            "boundary_source": str(getattr(estimate, "boundary_source", "") or ""),
        }
        payload.update(extra)
        self.last_debug = payload

    def _make_command(self, estimate: RowEstimate) -> BodyCommand:
        mode = str(getattr(estimate, "effective_mode", estimate.mode) or "")
        one_side_mode = mode in {"left_only", "right_only"}
        reverse = bool(self.args.reverse)
        max_wz_deg = min(abs(float(self.args.max_wz_deg)), 5.0)
        max_wz_rad = math.radians(max_wz_deg)
        max_heading_wz_deg = max(0.0, float(self.args.max_heading_wz_deg))
        max_heading_wz_rad = math.radians(max_heading_wz_deg)
        requested_max_wz_deg = max_wz_deg
        target_speed = float(self.args.reverse_speed if reverse else self.args.speed)
        center_y_target = float(self.args.center_y_target)
        line_fit = getattr(estimate, "center_line", None)
        # Reuse the same fitted lines in both directions.  In reverse, extend
        # the front-observed plant boundaries back to the leading rear edge of
        # the vehicle (rear body length plus the configured safety margin).
        # This makes the control reference match the part of the body that can
        # contact a pot first instead of an arbitrary point 0.60 m behind it.
        forward_lookahead_x = abs(float(self.args.forward_lookahead_x))
        reverse_rear_x = -(
            abs(float(ROBOT_FRAME_BACK))
            + max(0.0, float(self.row_cfg.safety_margin))
        )
        effective_lookahead_x = reverse_rear_x if reverse else forward_lookahead_x
        target_y = float(estimate.center_y)
        center_y_direct_error = float(estimate.center_y)
        direct_error_y = float(center_y_direct_error - center_y_target)
        track_x = effective_lookahead_x
        line_y_at_track = float(estimate.center_y)
        track_error_y = float(direct_error_y)
        error_y = float(track_error_y)
        heading_error = float(math.atan(line_fit[0])) if line_fit is not None else float(estimate.heading_rad)
        heading_error_deg = math.degrees(heading_error)
        stop_reason = ""
        control_phase = "normal_reverse_tracking" if reverse else "forward_tracking"
        warning = str(getattr(estimate, "warning", "") or "")
        wz_zeroed_reason = ""
        active_reverse_steer_sign = float(self.args.reverse_steer_sign)
        target_wz_deg = 0.0
        filtered_wz_deg = 0.0
        lat_term_deg = 0.0
        heading_term_deg = 0.0
        base_wz_deg = 0.0
        heading_term_raw_deg = 0.0
        reverse_rear_left_clearance_m = 0.0
        reverse_rear_right_clearance_m = 0.0
        reverse_rear_min_clearance_m = 0.0
        reverse_rear_guard_active = False

        if not reverse and abs(error_y) <= float(self.args.control_deadband_y):
            error_y = 0.0

        k_lat_eff = float(self.args.k_lat)
        k_heading_eff = float(self.args.k_heading)
        heading_term_raw = 0.0
        heading_term_limited = 0.0
        base_wz = 0.0
        wz_raw = 0.0
        vx = -abs(target_speed) if reverse else max(float(self.args.min_speed), float(target_speed))

        if estimate.reject_reason and not reverse:
            stop_reason = f"reject:{estimate.reject_reason}"
            control_phase = "emergency_stop"
            vx = 0.0
            wz_raw = 0.0
        elif reverse:
            if line_fit is not None:
                line_y_at_track = float(line_fit[0] * track_x + line_fit[1])
                track_error_y = float(line_y_at_track - center_y_target)
                target_y = line_y_at_track
                error_y = track_error_y
                heading_error = float(math.atan(line_fit[0]))
                heading_error_deg = math.degrees(heading_error)

            # Check the two leading rear corners against the left/right plant
            # boundaries after extending those straight fitted lines rearward.
            # One-side detection already supplies a virtual opposite boundary,
            # so the same footprint calculation works in all valid line modes.
            left_line = getattr(estimate, "left_line", None)
            right_line = getattr(estimate, "right_line", None)
            half_row_width = 0.5 * float(getattr(estimate, "row_width", 0.0) or self.args.row_width)
            if line_fit is not None:
                if left_line is None:
                    left_line = (float(line_fit[0]), float(line_fit[1]) + half_row_width)
                if right_line is None:
                    right_line = (float(line_fit[0]), float(line_fit[1]) - half_row_width)
            rear_half_width = float(self.row_cfg.vehicle_half_width)
            if left_line is not None:
                left_norm = math.hypot(1.0, float(left_line[0]))
                left_y_at_rear = float(left_line[0]) * track_x + float(left_line[1])
                reverse_rear_left_clearance_m = (
                    left_y_at_rear - rear_half_width
                ) / left_norm
            if right_line is not None:
                right_norm = math.hypot(1.0, float(right_line[0]))
                right_y_at_rear = float(right_line[0]) * track_x + float(right_line[1])
                reverse_rear_right_clearance_m = (
                    -rear_half_width - right_y_at_rear
                ) / right_norm
            reverse_rear_min_clearance_m = min(
                reverse_rear_left_clearance_m,
                reverse_rear_right_clearance_m,
            )
            reverse_rear_guard_active = (
                math.isfinite(reverse_rear_min_clearance_m)
                and reverse_rear_min_clearance_m < float(self.row_cfg.safety_margin)
            )

            # One centerline, one proportional controller, one kinematic sign.
            # No reverse deadband, minimum turn, recenter scaling, sign-flip
            # clearing, slew limiter, start ramp, or speed reduction is applied.
            vx = -abs(target_speed)
            k_lat_eff = float(self.args.reverse_k_lat)
            k_heading_eff = float(self.args.reverse_k_heading)
            max_wz_deg = abs(float(self.args.reverse_max_wz_deg))
            requested_max_wz_deg = max_wz_deg
            max_wz_rad = math.radians(max_wz_deg)
            lat_term_deg = k_lat_eff * track_error_y
            heading_term_raw_deg = k_heading_eff * heading_error_deg
            # When a rear corner is already inside the safety margin, heading
            # correction must not cancel the turn that moves the rear back
            # toward the corridor center.  Speed remains the requested fixed
            # reverse speed; only the priority of the two P terms changes.
            heading_term_deg = 0.0 if reverse_rear_guard_active else heading_term_raw_deg
            target_wz_deg = active_reverse_steer_sign * (lat_term_deg + heading_term_deg)
            last_wz_deg = math.degrees(float(self.last_cmd_wz))
            smoothing_alpha = max(0.0, min(0.40, float(self.args.reverse_wz_filter_alpha)))
            filtered_wz_deg = (
                smoothing_alpha * last_wz_deg
                + (1.0 - smoothing_alpha) * target_wz_deg
            )
            filtered_wz_deg = max(-max_wz_deg, min(max_wz_deg, filtered_wz_deg))
            base_wz_deg = target_wz_deg
            wz_raw = math.radians(target_wz_deg)
            wz = math.radians(filtered_wz_deg)
        else:
            if line_fit is not None:
                track_x = float(self.args.forward_lookahead_x)
                line_y_at_track = float(line_fit[0] * track_x + line_fit[1])
                track_error_y = float(line_y_at_track - center_y_target)
                target_y = line_y_at_track
                error_y = track_error_y
                heading_error = float(math.atan(line_fit[0]))
                heading_error_deg = math.degrees(heading_error)
                heading_deg = heading_error_deg
            if one_side_mode:
                max_wz_deg = min(max_wz_deg, 0.8)
                max_wz_rad = math.radians(max_wz_deg)
                k_lat_eff *= 0.5
                k_heading_eff *= 0.3
                max_heading_wz_deg = min(max_heading_wz_deg, max_wz_deg)
                max_heading_wz_rad = math.radians(max_heading_wz_deg)
            heading_term_raw = k_heading_eff * heading_error
            heading_term_limited = max(-max_heading_wz_rad, min(max_heading_wz_rad, heading_term_raw))
            if (
                abs(error_y) > float(self.args.heading_conflict_error_y)
                and (k_lat_eff * error_y) * heading_term_limited < 0.0
            ):
                heading_term_limited *= float(self.args.heading_conflict_scale)
            base_wz = k_lat_eff * error_y + heading_term_limited
            lat_term_deg = math.degrees(k_lat_eff * error_y)
            heading_term_deg = math.degrees(heading_term_limited)
            wz_raw = base_wz
            vx = max(float(self.args.min_speed), float(target_speed))

        if not reverse:
            base_wz_deg = math.degrees(base_wz)
        if not reverse and abs(error_y) < 0.012:
            base_wz = 0.0
            base_wz_deg = 0.0
            wz_raw = 0.0
            wz_zeroed_reason = "forward_deadband"
        if not reverse:
            wz_raw = max(-max_wz_rad, min(max_wz_rad, wz_raw))
            wz = wz_raw
            # Never let the minimum useful forward correction exceed the
            # configured maximum.  The old ordering produced 2 deg/s even when
            # row entry or one-side tracking was explicitly capped at 0.8-1.2.
            min_forward_wz_rad = min(math.radians(2.0), max_wz_rad)
            if abs(wz) > 1e-9 and abs(wz) < min_forward_wz_rad:
                wz = math.copysign(min_forward_wz_rad, wz)
            target_wz_deg = math.degrees(wz_raw)
            filtered_wz_deg = math.degrees(wz)

        self.last_direct_error_y = direct_error_y
        self.last_error_y = error_y
        now = time.monotonic()
        self.direct_error_history = [
            (ts, value) for ts, value in self.direct_error_history if now - ts <= 3.0
        ]
        self.direct_error_history.append((now, direct_error_y))
        rolling_center_error_median = _history_median(self.direct_error_history)
        suggested_center_y_target = float(center_y_target + rolling_center_error_median)
        self._set_debug_snapshot(
            estimate=estimate,
            found=True,
            control_phase=control_phase,
            stop_reason=stop_reason,
            final_vx=vx,
            final_wz=wz,
            center_y_target=center_y_target,
            error_y=error_y,
            heading_error=heading_error,
            warning=warning,
            center_y_direct_error=center_y_direct_error,
            error_y_used_for_control=error_y,
            rolling_center_error_median=rolling_center_error_median,
            suggested_center_y_target=suggested_center_y_target,
            target_y=target_y,
            effective_lookahead_x=effective_lookahead_x,
            track_x=track_x,
            line_y_at_track=line_y_at_track,
            track_error_y=track_error_y,
            lateral_error=track_error_y,
            heading_error_deg=heading_error_deg,
            line_mode=mode,
            line_slope=0.0 if line_fit is None else float(line_fit[0]),
            k_lat_eff=k_lat_eff,
            k_heading_eff=k_heading_eff,
            lat_term_deg=lat_term_deg,
            lateral_term_deg=lat_term_deg,
            heading_term_deg=heading_term_deg,
            heading_term_raw_deg=heading_term_raw_deg,
            base_wz_deg=base_wz_deg,
            target_wz_deg=target_wz_deg,
            wz_target_deg=target_wz_deg,
            wz_filtered_deg=filtered_wz_deg,
            wz_cmd_deg=math.degrees(wz),
            wz_raw_deg=math.degrees(wz_raw),
            reverse_max_wz_deg=max_wz_deg,
            wz_zeroed_reason=wz_zeroed_reason,
            requested_max_wz_deg=requested_max_wz_deg,
            effective_max_wz_deg=max_wz_deg,
            max_wz_deg=max_wz_deg,
            reverse_steer_sign=active_reverse_steer_sign,
            reverse_wz_filter_alpha=max(
                0.0, min(0.40, float(self.args.reverse_wz_filter_alpha))
            ),
            reverse_rear_x=reverse_rear_x,
            reverse_rear_left_clearance_m=reverse_rear_left_clearance_m,
            reverse_rear_right_clearance_m=reverse_rear_right_clearance_m,
            reverse_rear_min_clearance_m=reverse_rear_min_clearance_m,
            reverse_rear_guard_active=reverse_rear_guard_active,
            raw_center_warning=warning,
        )

        return BodyCommand(gear=self.args.gear, vx=vx, vy=0.0, wz=wz)

    def _send_stop(self, *, force_brake: bool = False) -> None:
        self.last_cmd_vx = 0.0
        self.last_cmd_wz = 0.0
        self._send_drive(self.args.gear, 0.0, 0.0, force_brake=force_brake)

    def _hold_last_good_command(
        self,
        *,
        estimate: RowEstimate,
        control_phase: str,
        stop_reason: str,
        warning: str,
        wz_limit_deg: float,
        wz_scale: float = 1.0,
    ) -> bool:
        if self.last_good_time <= 0.0:
            return False
        hold_vx = float(self.last_good_cmd.vx)
        hold_wz_deg = math.degrees(float(self.last_good_cmd.wz)) * float(wz_scale)
        hold_wz_deg = max(-abs(float(wz_limit_deg)), min(abs(float(wz_limit_deg)), hold_wz_deg))
        hold_cmd = BodyCommand(
            gear=self._normalized_gear(),
            vx=hold_vx,
            vy=0.0,
            wz=math.radians(hold_wz_deg),
        )
        self.last_cmd_vx = float(hold_cmd.vx)
        self.last_cmd_wz = float(hold_cmd.wz)
        self._set_debug_snapshot(
            estimate=estimate,
            found=False,
            control_phase=control_phase,
            stop_reason=stop_reason,
            final_vx=float(hold_cmd.vx),
            final_wz=float(hold_cmd.wz),
            warning=warning,
            lost_hold_forward=(control_phase == "lost_hold_forward"),
            last_good_age_sec=time.monotonic() - self.last_good_time if self.last_good_time > 0.0 else -1.0,
            last_good_error_y=float(self.last_error_y),
            last_good_heading_deg=float(self.last_good_heading_deg),
            last_good_cmd_wz_deg=math.degrees(float(self.last_good_cmd.wz)),
            hold_vx=float(hold_cmd.vx),
            hold_wz_deg=hold_wz_deg,
            control_using_last_good_line=True,
        )
        self._send_drive(hold_cmd.gear, float(hold_cmd.vx), float(hold_cmd.wz))
        return True

    def _continue_reverse_command(self, estimate: RowEstimate) -> None:
        """Keep moving on fresh scans when centerline fitting misses briefly."""
        previous_debug = dict(self.last_debug)
        hold_vx = -abs(float(self.args.reverse_speed))
        hold_wz_deg = (
            math.degrees(float(self.last_good_cmd.wz))
            if self.last_good_time > 0.0
            else 0.0
        )
        max_wz_deg = abs(float(self.args.reverse_max_wz_deg))
        hold_wz_deg = max(-max_wz_deg, min(max_wz_deg, hold_wz_deg))
        hold_wz = math.radians(hold_wz_deg)
        self.last_cmd_vx = hold_vx
        self.last_cmd_wz = hold_wz
        self._set_debug_snapshot(
            estimate=estimate,
            found=False,
            control_phase="reverse_transient_line_hold",
            stop_reason="",
            final_vx=hold_vx,
            final_wz=hold_wz,
            warning=str(getattr(estimate, "warning", "") or "transient_centerline_loss"),
            lateral_error=float(previous_debug.get("lateral_error", self.last_error_y)),
            heading_error_deg=float(previous_debug.get("heading_error_deg", self.last_good_heading_deg)),
            lateral_term_deg=float(previous_debug.get("lateral_term_deg", 0.0)),
            heading_term_deg=float(previous_debug.get("heading_term_deg", 0.0)),
            wz_target_deg=float(previous_debug.get("wz_target_deg", hold_wz_deg)),
            wz_filtered_deg=hold_wz_deg,
            reverse_steer_sign=float(self.args.reverse_steer_sign),
            line_mode=str(getattr(estimate, "effective_mode", estimate.mode) or ""),
            control_using_last_good_line=self.last_good_time > 0.0,
        )
        self._send_drive(self._normalized_gear(), hold_vx, hold_wz)

    def _normalized_gear(self) -> str:
        gear = str(self.args.gear).strip().lower()
        if gear == "8":
            return "crab"
        if gear == "6":
            return "4t4d"
        if gear in {"4t4d", "crab", "neutral"}:
            return gear
        return "4t4d"

    def _send_drive(self, gear: str, vx: float, wz: float, force_brake: bool = False) -> None:
        now = time.monotonic()
        gear = self._normalized_gear() if gear in {"6", "8"} else gear
        steering_angle = self.last_steering_angle
        steering_speed = 0.0
        steering_zeroed = True
        steering_assist_active = False
        cmd_wz_deg_s = math.degrees(wz)
        reverse_motion = float(vx) < -1e-6
        auto_reverse_steering_assist = False
        assist_enabled = bool(self.args.enable_4t4d_steering_assist)
        if gear == "4t4d" and assist_enabled:
            effective_vx = float(vx)
            if abs(effective_vx) < float(self.args.steering_assist_min_speed_mps):
                fallback_sign = 1.0 if effective_vx == 0.0 else math.copysign(1.0, effective_vx)
                effective_vx = fallback_sign * float(self.args.steering_assist_min_speed_mps)
            steering_angle = math.degrees(
                math.atan(
                    float(self.args.steering_assist_wheelbase_m) * float(wz) / effective_vx
                )
            ) * float(self.args.steering_assist_gain)
            steering_angle = max(
                -abs(float(self.args.steering_assist_max_angle_deg)),
                min(abs(float(self.args.steering_assist_max_angle_deg)), steering_angle),
            )
            steering_zeroed = abs(steering_angle) < 1e-6
            steering_assist_active = True
            steering_speed = abs(float(self.args.steering_assist_speed_mps))
            steering_cmd = SteeringCommand(gear=gear, speed=steering_speed, angle=steering_angle)
        else:
            # Keep a 4t4d steering command in the state so CommandSender sends
            # one explicit steering-mode sync when the body changes from crab
            # back to 4t4d.  Continuous 4t4d steering frames remain filtered
            # by CommandSender; this is only the gear-transition handshake.
            steering_cmd = SteeringCommand(
                gear=gear,
                speed=0.0,
                angle=0.0 if gear == "4t4d" else steering_angle,
            )

        motion_active = gear not in {"park", "neutral"} and (abs(vx) > 1e-6 or abs(wz) > 1e-6)
        io_cmd = IOCommand(
            light_mode="free" if self.args.low_beam else "auto",
            low_beam=bool(self.args.low_beam),
            brake=force_brake and not motion_active,
        )
        self.sender.update(
            BodyCommand(gear=gear, vx=vx, vy=0.0, wz=wz),
            steering_cmd,
            io_cmd,
        )
        if motion_active:
            self.sender.request_unlock()

        self.can_reader.poll(timeout=0.0, limit=20)
        snapshot = self.can_reader.snapshot()
        motion = snapshot.get("motion", {})
        steering = snapshot.get("steering", {})
        io_state = snapshot.get("io", {})
        runtime = self.sender.feedback_snapshot().get("_runtime", {})
        reported_unlock_ok = bool(io_state.get("unlock_ok", False))
        unlock_now = bool(runtime.get("unlock_now", False))
        body_feedback_gear = _normalize_feedback_gear(motion.get("gear"))
        steering_feedback_gear = _normalize_feedback_gear(steering.get("gear"))
        steering_feedback_angle = float(steering.get("wheel_angle_deg", 0.0) or 0.0)
        steering_feedback_speed = float(steering.get("wheel_speed_mps", 0.0) or 0.0)
        self.last_steering_feedback_angle = steering_feedback_angle
        fb_vx = float(motion.get("vx_mps", motion.get("vx", 0.0)) or 0.0)
        fb_wz = float(motion.get("wz_dps", motion.get("wz", 0.0)) or 0.0)
        wz_following = not (abs(cmd_wz_deg_s) > 0.5 and abs(fb_wz) < 0.05)
        if wz_following:
            self.wz_not_following_count = 0
            if abs(cmd_wz_deg_s) >= 0.8:
                self.last_following_wz_deg = cmd_wz_deg_s
        else:
            self.wz_not_following_count += 1
            if self.wz_not_following_count == 5:
                self.get_logger().warning("wz_not_following")

        if self.last_debug:
            self.last_debug.update(
                {
                    "cmd_wz_deg_s": cmd_wz_deg_s,
                    "fb_wz": fb_wz,
                    "wz_following": wz_following,
                    "wz_not_following_count": int(self.wz_not_following_count),
                    "send_body_wz": float(wz),
                    "send_body_wz_deg": cmd_wz_deg_s,
                    "send_steering_angle": steering_angle if steering_cmd is not None else runtime.get("send_steering_angle", None),
                    "send_steering_speed": steering_speed if steering_cmd is not None else float(runtime.get("send_steering_speed", 0.0) or 0.0),
                    "steering_cmd_present": steering_cmd is not None or bool(runtime.get("steering_cmd_present", False)),
                    "send_steering_zeroed": steering_zeroed if steering_cmd is not None else bool(runtime.get("send_steering_zeroed", False)),
                    "steering_assist_active": steering_assist_active,
                    "auto_reverse_steering_assist": auto_reverse_steering_assist,
                    "fb_steering_angle_deg": steering_feedback_angle,
                    "fb_steering_speed": steering_feedback_speed,
                    "command_stale": bool(runtime.get("command_stale", False)),
                    "command_age_s": float(runtime.get("command_age_s", 0.0) or 0.0),
                }
            )

        if now - self.send_state.last_cmd_log_ts >= 0.5:
            effective_log_max_wz_deg = float(
                self.last_debug.get("effective_max_wz_deg", min(abs(float(self.args.max_wz_deg)), 5.0))
                if self.last_debug
                else min(abs(float(self.args.max_wz_deg)), 5.0)
            )
            self.get_logger().info(
                "drive "
                f"gear={gear} gear_code={_gear_code_for_log(gear)} vx={vx:+.3f} "
                f"wz_rad={wz:+.3f} wz_deg={math.degrees(wz):+.3f} max_wz_deg={effective_log_max_wz_deg:.3f} "
                f"unlock={unlock_now} fb_unlock={reported_unlock_ok} "
                f"fb_body_gear={body_feedback_gear or '-'} fb_steer_gear={steering_feedback_gear or '-'} "
                f"fb_vx={fb_vx:+.3f} fb_wz={fb_wz:+.3f} "
                f"steer_angle={steering_feedback_angle:+.3f} steer_speed={steering_feedback_speed:+.3f}"
            )
            self.send_state.last_cmd_log_ts = now

    def _on_control(self) -> None:
        now = time.monotonic()
        with self.scan_lock:
            scan = self.last_scan
            scan_age = now - self.last_scan_time if self.last_scan is not None else float("inf")
        if not bool(self.drive_enable):
            if scan is None or scan_age > float(self.args.scan_timeout):
                self.last_estimate = RowEstimate(found=False, mode="scan_timeout")
            else:
                self.last_estimate = self._estimate_row(scan)
            return
        if scan is None or scan_age > float(self.args.scan_timeout):
            self.last_estimate = RowEstimate(found=False, mode="scan_timeout")
            if not bool(self.args.reverse) and self._hold_last_good_command(
                estimate=self.last_estimate,
                control_phase="scan_timeout_hold_forward",
                stop_reason="scan_timeout_hold_forward",
                warning="scan_timeout_hold_forward",
                wz_limit_deg=float(self.args.forward_lost_hold_max_wz_deg),
                wz_scale=float(self.args.forward_lost_hold_wz_scale),
            ):
                return
            self._set_debug_snapshot(
                estimate=self.last_estimate,
                found=False,
                control_phase="emergency_stop",
                stop_reason="scan_timeout",
                final_vx=0.0,
                final_wz=0.0,
            )
            self._send_stop(force_brake=bool(self.args.reverse))
            return

        estimate = self._estimate_row(scan)
        self.last_estimate = estimate
        if estimate.found:
            self.last_found_time = now
            self.last_good_time = now
            cmd = self._make_command(estimate)
            cmd.gear = self._normalized_gear()
            self.last_good_cmd = cmd
            self.last_good_estimate = estimate
            self.last_good_center_line = getattr(estimate, "center_line", None)
            self.last_good_left_line = getattr(estimate, "left_line", None)
            self.last_good_right_line = getattr(estimate, "right_line", None)
            self.last_good_mode = str(getattr(estimate, "effective_mode", estimate.mode) or "")
            self.last_good_heading_deg = math.degrees(float(getattr(estimate, "heading_rad", 0.0) or 0.0))
            self.last_cmd_vx = float(cmd.vx)
            self.last_cmd_wz = float(cmd.wz)
            self._send_drive(cmd.gear, float(cmd.vx), float(cmd.wz))
        else:
            if bool(self.args.reverse):
                self._continue_reverse_command(estimate)
                return
            last_good_age = now - self.last_good_time if self.last_good_time > 0.0 else float("inf")
            if not bool(self.args.reverse) and self._hold_last_good_command(
                estimate=estimate,
                control_phase="lost_hold_forward",
                stop_reason="lost_hold_forward",
                warning="lost_hold_forward",
                wz_limit_deg=float(self.args.forward_lost_hold_max_wz_deg),
                wz_scale=float(self.args.forward_lost_hold_wz_scale),
            ):
                return
            self._set_debug_snapshot(
                estimate=estimate,
                found=False,
                control_phase="emergency_stop",
                stop_reason=estimate.reject_reason or estimate.mode or "lost",
                final_vx=0.0,
                final_wz=0.0,
                warning=str(getattr(estimate, "warning", "") or ""),
                lost_hold_forward=False,
                last_good_age_sec=last_good_age if self.last_good_time > 0.0 else -1.0,
                last_good_error_y=float(self.last_error_y),
                last_good_heading_deg=float(self.last_good_heading_deg),
                last_good_cmd_wz_deg=math.degrees(float(self.last_good_cmd.wz)),
                hold_vx=0.0,
                hold_wz_deg=0.0,
                control_using_last_good_line=False,
            )
            self._send_stop(force_brake=bool(self.args.reverse))

    def _publish_status(self) -> None:
        estimate = self.last_estimate
        payload = {
            "found": estimate.found,
            "mode": estimate.mode,
            "raw_center_y_m": round(float(estimate.raw_center_y), 4),
            "filtered_center_y_m": round(float(estimate.center_y), 4),
            "center_y_m": round(float(estimate.center_y), 4),
            "heading_deg": round(math.degrees(float(estimate.heading_rad)), 3),
            "row_width_m": round(float(estimate.row_width), 4),
            "left_bins": int(estimate.left_bins),
            "right_bins": int(estimate.right_bins),
            "candidate_bins": int(estimate.candidate_bins),
            "reject_reason": estimate.reject_reason,
            "last_good_row_width_m": round(float(self.last_good_row_width), 4),
            "cmd_vx_mps": round(float(self.last_cmd_vx), 4),
            "cmd_wz_rad_s": round(float(self.last_cmd_wz), 4),
            "cmd_wz_deg_s": round(math.degrees(float(self.last_cmd_wz)), 4),
            "max_wz_deg": round(
                abs(float(self.args.reverse_max_wz_deg))
                if bool(self.args.reverse)
                else min(abs(float(self.args.max_wz_deg)), 5.0),
                4,
            ),
            "control_phase": self.last_debug.get("control_phase", ""),
            "lateral_error": round(float(self.last_debug.get("lateral_error", 0.0)), 4),
            "heading_error_deg": round(float(self.last_debug.get("heading_error_deg", 0.0)), 4),
            "lateral_term_deg": round(float(self.last_debug.get("lateral_term_deg", 0.0)), 4),
            "heading_term_deg": round(float(self.last_debug.get("heading_term_deg", 0.0)), 4),
            "wz_target_deg": round(float(self.last_debug.get("wz_target_deg", 0.0)), 4),
            "wz_filtered_deg": round(float(self.last_debug.get("wz_filtered_deg", 0.0)), 4),
            "final_vx": round(float(self.last_debug.get("final_vx", 0.0)), 4),
            "final_wz_deg": round(float(self.last_debug.get("final_wz_deg", 0.0)), 4),
            "reverse_steer_sign": round(float(self.last_debug.get("reverse_steer_sign", 0.0)), 4),
            "line_mode": self.last_debug.get("line_mode", ""),
        }
        text = json.dumps(payload, ensure_ascii=True)
        self.status_pub.publish(String(data=text))
        print(text, flush=True)
        if self.last_debug:
            debug_text = json.dumps(
                {
                        "found": bool(self.last_debug.get("found", False)),
                        "calibrated": bool(self.last_debug.get("calibrated", False)),
                        "sensor_yaw_deg": round(float(self.last_debug.get("sensor_yaw_deg", 0.0)), 4),
                        "lidar_yaw_correction_deg": round(float(self.last_debug.get("lidar_yaw_correction_deg", 0.0)), 4),
                        "lidar_x_offset_m": round(float(self.last_debug.get("lidar_x_offset_m", 0.0)), 4),
                        "lidar_y_offset_m": round(float(self.last_debug.get("lidar_y_offset_m", 0.0)), 4),
                        "reverse": bool(self.last_debug.get("reverse", False)),
                        "effective_mode": self.last_debug.get("effective_mode", ""),
                        "raw_center_y": round(float(self.last_debug.get("raw_center_y", 0.0)), 4),
                        "filtered_center_y": round(float(self.last_debug.get("filtered_center_y", 0.0)), 4),
                        "center_y_target": round(float(self.last_debug.get("center_y_target", 0.0)), 4),
                        "center_y_direct_error": round(float(self.last_debug.get("center_y_direct_error", 0.0)), 4),
                        "track_x": round(float(self.last_debug.get("track_x", 0.0)), 4),
                        "line_y_at_track": round(float(self.last_debug.get("line_y_at_track", 0.0)), 4),
                        "track_error_y": round(float(self.last_debug.get("track_error_y", 0.0)), 4),
                        "error_y_used_for_control": round(float(self.last_debug.get("error_y_used_for_control", 0.0)), 4),
                        "error_y": round(float(self.last_debug.get("error_y", 0.0)), 4),
                        "heading_deg": round(float(self.last_debug.get("heading_deg", 0.0)), 4),
                        "heading_error_deg": round(float(self.last_debug.get("heading_error_deg", 0.0)), 4),
                        "k_lat_eff": round(float(self.last_debug.get("k_lat_eff", 0.0)), 4),
                        "k_heading_eff": round(float(self.last_debug.get("k_heading_eff", 0.0)), 4),
                        "lat_term_deg": round(float(self.last_debug.get("lat_term_deg", 0.0)), 4),
                        "heading_term_deg": round(float(self.last_debug.get("heading_term_deg", 0.0)), 4),
                        "base_wz_deg": round(float(self.last_debug.get("base_wz_deg", 0.0)), 4),
                        "lateral_error": round(float(self.last_debug.get("lateral_error", 0.0)), 4),
                        "lateral_term_deg": round(float(self.last_debug.get("lateral_term_deg", 0.0)), 4),
                        "wz_target_deg": round(float(self.last_debug.get("wz_target_deg", 0.0)), 4),
                        "wz_filtered_deg": round(float(self.last_debug.get("wz_filtered_deg", 0.0)), 4),
                        "line_mode": self.last_debug.get("line_mode", ""),
                        "wz_cmd_deg": round(float(self.last_debug.get("wz_cmd_deg", 0.0)), 4),
                        "wz_raw_deg": round(float(self.last_debug.get("wz_raw_deg", 0.0)), 4),
                        "reverse_max_wz_deg": round(float(self.last_debug.get("reverse_max_wz_deg", 0.0)), 4),
                        "reverse_wz_filter_alpha": round(
                            float(self.last_debug.get("reverse_wz_filter_alpha", 0.0)), 4
                        ),
                        "wz_zeroed_reason": self.last_debug.get("wz_zeroed_reason", ""),
                        "lost_hold_forward": bool(self.last_debug.get("lost_hold_forward", False)),
                        "last_good_age_sec": round(float(self.last_debug.get("last_good_age_sec", 0.0)), 4),
                        "last_good_error_y": round(float(self.last_debug.get("last_good_error_y", 0.0)), 4),
                        "last_good_heading_deg": round(float(self.last_debug.get("last_good_heading_deg", 0.0)), 4),
                        "last_good_cmd_wz_deg": round(float(self.last_debug.get("last_good_cmd_wz_deg", 0.0)), 4),
                        "hold_vx": round(float(self.last_debug.get("hold_vx", 0.0)), 4),
                        "hold_wz_deg": round(float(self.last_debug.get("hold_wz_deg", 0.0)), 4),
                        "rolling_center_error_median": round(float(self.last_debug.get("rolling_center_error_median", 0.0)), 4),
                        "suggested_center_y_target": round(float(self.last_debug.get("suggested_center_y_target", 0.0)), 4),
                        "raw_points_count": int(self.last_debug.get("raw_points_count", 0)),
                        "filtered_points_count": int(self.last_debug.get("filtered_points_count", 0)),
                        "left_points_count": int(self.last_debug.get("left_points_count", 0)),
                        "right_points_count": int(self.last_debug.get("right_points_count", 0)),
                        "center_y_jump": round(float(self.last_debug.get("center_y_jump", 0.0)), 4),
                        "heading_jump": round(float(self.last_debug.get("heading_jump", 0.0)), 4),
                        "center_jump_rejected": bool(self.last_debug.get("center_jump_rejected", False)),
                        "heading_jump_rejected": bool(self.last_debug.get("heading_jump_rejected", False)),
                        "left_valid": bool(self.last_debug.get("left_valid", False)),
                        "right_valid": bool(self.last_debug.get("right_valid", False)),
                        "left_reject_reason": self.last_debug.get("left_reject_reason", ""),
                        "right_reject_reason": self.last_debug.get("right_reject_reason", ""),
                        "parallel_angle_diff_deg": round(float(self.last_debug.get("parallel_angle_diff_deg", 0.0)), 4),
                        "width_error_m": round(float(self.last_debug.get("width_error_m", 0.0)), 4),
                        "boundary_width_tolerance_m": round(float(self.last_debug.get("boundary_width_tolerance_m", 0.0)), 4),
                        "left_residual_median": round(float(self.last_debug.get("left_residual_median", 0.0)), 4),
                        "right_residual_median": round(float(self.last_debug.get("right_residual_median", 0.0)), 4),
                        "left_consecutive_bins": int(self.last_debug.get("left_consecutive_bins", 0)),
                        "right_consecutive_bins": int(self.last_debug.get("right_consecutive_bins", 0)),
                        "boundary_source": self.last_debug.get("boundary_source", ""),
                        "control_using_last_good_line": bool(self.last_debug.get("control_using_last_good_line", False)),
                        "final_vx": round(float(self.last_debug.get("final_vx", 0.0)), 4),
                        "final_wz_deg": round(float(self.last_debug.get("final_wz_deg", 0.0)), 4),
                        "cmd_vx": round(float(self.last_cmd_vx), 4),
                        "cmd_wz_deg_s": round(float(self.last_debug.get("cmd_wz_deg_s", 0.0)), 4),
                        "fb_wz": round(float(self.last_debug.get("fb_wz", 0.0)), 4),
                        "wz_following": bool(self.last_debug.get("wz_following", False)),
                        "wz_not_following_count": int(self.last_debug.get("wz_not_following_count", 0)),
                        "send_body_wz": round(float(self.last_debug.get("send_body_wz", 0.0)), 4),
                        "send_body_wz_deg": round(float(self.last_debug.get("send_body_wz_deg", 0.0)), 4),
                        "send_steering_angle": self.last_debug.get("send_steering_angle", None),
                        "send_steering_speed": round(float(self.last_debug.get("send_steering_speed", 0.0)), 4),
                        "steering_cmd_present": bool(self.last_debug.get("steering_cmd_present", False)),
                        "send_steering_zeroed": bool(self.last_debug.get("send_steering_zeroed", False)),
                        "steering_assist_active": bool(self.last_debug.get("steering_assist_active", False)),
                        "auto_reverse_steering_assist": bool(self.last_debug.get("auto_reverse_steering_assist", False)),
                        "fb_steering_angle_deg": round(float(self.last_debug.get("fb_steering_angle_deg", 0.0)), 4),
                        "fb_steering_speed": round(float(self.last_debug.get("fb_steering_speed", 0.0)), 4),
                        "command_stale": bool(self.last_debug.get("command_stale", False)),
                        "command_age_s": round(float(self.last_debug.get("command_age_s", 0.0)), 4),
                        "requested_max_wz_deg": round(float(self.last_debug.get("requested_max_wz_deg", 0.0)), 4),
                        "effective_max_wz_deg": round(float(self.last_debug.get("effective_max_wz_deg", 0.0)), 4),
                        "max_wz_deg": round(float(self.last_debug.get("max_wz_deg", 0.0)), 4),
                        "reverse_steer_sign": round(float(self.last_debug.get("reverse_steer_sign", 0.0)), 4),
                        "raw_center_warning": self.last_debug.get("raw_center_warning", ""),
                        "reject_reason": self.last_debug.get("reject_reason", ""),
                        "stop_reason": self.last_debug.get("stop_reason", ""),
                },
                ensure_ascii=True,
                indent=2,
            )
            print(f"[debug]\n{debug_text}", flush=True)

    def stop(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        try:
            for _ in range(3):
                self._send_stop()
                time.sleep(0.05)
        finally:
            self.can_reader.close()
            self.sender.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use lidar to find plant-row centerline and follow it")
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument("--status-topic", default="/plant_row/status")
    parser.add_argument("--interface", default="socketcan")
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--bitrate", type=int, default=500000)
    parser.add_argument("--gear", default="4t4d", choices=["neutral", "4t4d", "crab", "6", "8"])
    parser.add_argument("--speed", type=float, default=0.12)
    parser.add_argument("--min-speed", type=float, default=0.04)
    parser.add_argument("--max-wz-deg", type=float, default=1.8)
    parser.add_argument("--max-heading-wz-deg", type=float, default=0.5)
    parser.add_argument("--k-lat", type=float, default=1.5)
    parser.add_argument("--k-heading", type=float, default=0.05)
    parser.add_argument("--center-y-target", type=float, default=0.0)
    parser.add_argument("--heading-conflict-error-y", type=float, default=0.012)
    parser.add_argument("--heading-conflict-scale", type=float, default=0.0)
    parser.add_argument("--row-width", type=float, default=0.60)
    parser.add_argument("--vehicle-width", type=float, default=0.40)
    parser.add_argument("--min-row-width", type=float, default=0.48)
    parser.add_argument("--max-row-width", type=float, default=0.78)
    parser.add_argument("--lookahead-x", type=float, default=0.75)
    parser.add_argument("--forward-lookahead-x", type=float, default=0.6)
    parser.add_argument("--forward-min", type=float, default=0.15)
    parser.add_argument("--forward-max", type=float, default=1.60)
    parser.add_argument("--lateral-limit", type=float, default=0.75)
    parser.add_argument("--range-min", type=float, default=0.05)
    parser.add_argument("--range-max", type=float, default=6.0)
    parser.add_argument("--bin-size", type=float, default=0.20)
    parser.add_argument("--min-points", type=int, default=16)
    parser.add_argument("--min-bins", type=int, default=2)
    parser.add_argument("--min-line-bins", type=int, default=4)
    parser.add_argument("--min-side-points-per-bin", type=int, default=2)
    parser.add_argument("--boundary-width-tolerance-m", type=float, default=0.0)
    parser.add_argument("--center-deadband", type=float, default=0.03)
    parser.add_argument("--left-percentile", type=float, default=20.0)
    parser.add_argument("--right-percentile", type=float, default=80.0)
    parser.add_argument("--sensor-yaw-deg", type=float, default=180.0)
    parser.add_argument("--lidar-yaw-correction-deg", type=float, default=0.0)
    parser.add_argument("--lidar-x-offset-m", type=float, default=0.0)
    parser.add_argument("--lidar-y-offset-m", type=float, default=0.0)
    parser.add_argument("--control-deadband-y", type=float, default=0.001)
    parser.add_argument("--slow-error-y", type=float, default=0.04)
    parser.add_argument("--stop-error-y", type=float, default=0.065)
    parser.add_argument("--slow-heading-rad", type=float, default=0.18)
    parser.add_argument("--safety-margin", type=float, default=0.03)
    parser.add_argument("--center-jump-reject", type=float, default=0.25)
    parser.add_argument("--one-side-center-jump-reject", type=float, default=0.30)
    parser.add_argument("--center-y-reject-abs", type=float, default=0.16)
    parser.add_argument("--raw-center-out-of-range", type=float, default=0.20)
    parser.add_argument("--one-side-raw-center-out-of-range", type=float, default=0.28)
    parser.add_argument("--one-side-stop-error-y", type=float, default=0.18)
    parser.add_argument("--one-side-safety-stop-band", type=float, default=0.18)
    parser.add_argument("--control-period", type=float, default=0.05)
    parser.add_argument("--status-period", type=float, default=0.20)
    parser.add_argument("--scan-timeout", type=float, default=1.0)
    parser.add_argument("--history-window-s", type=float, default=0.40)
    parser.add_argument("--min-center-history", type=int, default=2)
    parser.add_argument("--center-y-alpha", type=float, default=0.75)
    parser.add_argument("--center-y-max-jump", type=float, default=0.08)
    parser.add_argument("--lost-hold-s", type=float, default=0.15)
    parser.add_argument("--lost-hold-speed-scale", type=float, default=0.35)
    parser.add_argument("--lost-hold-wz-scale", type=float, default=0.0)
    parser.add_argument("--forward-lost-hold-sec", type=float, default=0.35)
    parser.add_argument("--forward-lost-stop-sec", type=float, default=0.50)
    parser.add_argument("--forward-lost-hold-wz-scale", type=float, default=0.5)
    parser.add_argument("--forward-lost-hold-max-wz-deg", type=float, default=0.6)
    parser.add_argument("--reverse-speed", type=float, default=0.15)
    parser.add_argument("--reverse-k-lat", type=float, default=14.0)
    parser.add_argument("--reverse-k-heading", type=float, default=0.10)
    parser.add_argument("--reverse-max-wz-deg", type=float, default=2.5)
    parser.add_argument("--reverse-steer-sign", type=float, default=-1.0)
    parser.add_argument("--reverse-wz-filter-alpha", type=float, default=0.30)
    parser.add_argument("--enable-4t4d-steering-assist", action="store_true")
    parser.add_argument("--steering-assist-wheelbase-m", type=float, default=0.85)
    parser.add_argument("--steering-assist-gain", type=float, default=1.0)
    parser.add_argument("--steering-assist-max-angle-deg", type=float, default=12.0)
    parser.add_argument("--steering-assist-min-speed-mps", type=float, default=0.05)
    parser.add_argument("--steering-assist-speed-mps", type=float, default=0.20)
    parser.add_argument("--low-beam", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    previous_threading_excepthook = threading.excepthook

    def _thread_excepthook(args_: threading.ExceptHookArgs) -> None:
        traceback.print_exception(args_.exc_type, args_.exc_value, args_.exc_traceback)
        if previous_threading_excepthook is not None:
            previous_threading_excepthook(args_)

    threading.excepthook = _thread_excepthook
    rclpy.init()
    node = PlantRowFollower(args)

    def _shutdown(*_args: object) -> None:
        node.get_logger().info("shutdown requested")
        node.stop()
        if rclpy.ok():
            rclpy.shutdown()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
