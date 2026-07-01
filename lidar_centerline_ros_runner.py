#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

PLANT_ROOT = Path("/home/orangepi/ugv/plant_lidar_centerline")
if str(PLANT_ROOT) not in sys.path:
    sys.path.insert(0, str(PLANT_ROOT))

from row_geometry import RowEstimate, RowFollowerConfig, blend_line, estimate_row  # type: ignore  # noqa: E402


def _history_median(history: list[tuple[float, float]]) -> float:
    if not history:
        return 0.0
    return float(np.median(np.asarray([value for _ts, value in history], dtype=np.float64)))


class PlantRowRosRunner(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("lidar_centerline_ros_runner")
        self.args = args
        self.last_scan: LaserScan | None = None
        self.last_scan_time = 0.0
        self.last_estimate = RowEstimate(found=False, mode="boot")
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
        self.last_debug: dict[str, object] = {}
        self.last_good_cmd = Twist()
        self.last_good_estimate = RowEstimate(found=False)
        self.last_good_center_line: tuple[float, float] | None = None
        self.last_good_left_line: tuple[float, float] | None = None
        self.last_good_right_line: tuple[float, float] | None = None
        self.last_good_mode = ""
        self.drive_enable = False
        self.reverse = False
        self.cruise_vx = abs(float(args.speed))
        self.low_beam = bool(args.low_beam)
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
        self.cmd_pub = self.create_publisher(Twist, args.cmd_vel_topic, 10)
        self.status_pub = self.create_publisher(String, args.status_topic, 10)
        self.create_subscription(LaserScan, args.scan_topic, self._on_scan, 10)
        self.create_subscription(String, args.drive_mode_topic, self._on_drive_mode, 10)
        self.create_timer(max(0.02, float(args.control_period)), self._on_control)
        self.create_timer(max(0.1, float(args.status_period)), self._publish_status)

    def _on_scan(self, msg: LaserScan) -> None:
        self.last_scan = msg
        self.last_scan_time = time.monotonic()

    def _on_drive_mode(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception:
            payload = {}
        self.drive_enable = bool(payload.get("enable", False))
        self.reverse = bool(payload.get("reverse", False))
        self.low_beam = bool(payload.get("low_beam", self.low_beam))
        try:
            self.cruise_vx = abs(float(payload.get("cruise_vx", self.args.speed)))
        except Exception:
            self.cruise_vx = abs(float(self.args.speed))

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
        estimate.left_line = blend_line(self.last_good_left_line, getattr(estimate, "left_line", None), keep_ratio=0.0)
        estimate.right_line = blend_line(self.last_good_right_line, getattr(estimate, "right_line", None), keep_ratio=0.0)
        estimate.center_line = blend_line(self.last_good_center_line, getattr(estimate, "center_line", None), keep_ratio=0.0)

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
        **extra: object,
    ) -> None:
        payload: dict[str, object] = {
            "found": found,
            "control_phase": control_phase,
            "reverse": bool(self.reverse),
            "drive_enable": bool(self.drive_enable),
            "raw_center_y": float(getattr(estimate, "raw_center_y", 0.0) or 0.0),
            "filtered_center_y": float(getattr(estimate, "center_y", 0.0) or 0.0),
            "center_y_target": float(center_y_target if center_y_target is not None else float(self.args.center_y_target)),
            "error_y": float(error_y),
            "heading_deg": math.degrees(float(heading_error)),
            "mode": str(estimate.mode or ""),
            "effective_mode": str(getattr(estimate, "effective_mode", estimate.mode) or ""),
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
        }
        payload.update(extra)
        self.last_debug = payload

    def _publish_cmd(self, vx: float, wz: float) -> None:
        msg = Twist()
        msg.linear.x = float(vx)
        msg.angular.z = float(wz)
        self.last_cmd_vx = float(vx)
        self.last_cmd_wz = float(wz)
        self.cmd_pub.publish(msg)

    def _build_cmd(self, vx: float, wz: float) -> Twist:
        msg = Twist()
        msg.linear.x = float(vx)
        msg.angular.z = float(wz)
        return msg

    def _make_command(self, estimate: RowEstimate) -> Twist:
        mode = str(getattr(estimate, "effective_mode", estimate.mode) or "")
        one_side_mode = mode in {"left_only", "right_only"}
        max_wz_deg = min(abs(float(self.args.max_wz_deg)), 5.0)
        max_wz_rad = math.radians(max_wz_deg)
        max_heading_wz_deg = max(0.0, float(self.args.max_heading_wz_deg))
        max_heading_wz_rad = math.radians(max_heading_wz_deg)
        requested_max_wz_deg = max_wz_deg
        target_speed = float(self.cruise_vx)
        center_y_target = float(self.args.center_y_target)
        line_fit = getattr(estimate, "center_line", None)
        reverse = bool(self.reverse)
        effective_lookahead_x = float(self.args.reverse_lookahead_x if reverse else self.args.forward_lookahead_x)
        target_y = float(estimate.center_y)
        center_y_direct_error = float(estimate.center_y)
        direct_error_y = float(center_y_direct_error - center_y_target)
        track_x = effective_lookahead_x
        line_y_at_track = float(estimate.center_y)
        track_error_y = float(direct_error_y)
        error_y = float(track_error_y)
        heading_error = float(math.atan(line_fit[0])) if line_fit is not None else float(estimate.heading_rad)
        heading_error_deg = math.degrees(heading_error)
        raw_center_y = float(estimate.raw_center_y)
        stop_reason = ""
        control_phase = "normal_reverse_tracking" if reverse else "forward_tracking"
        warning = str(getattr(estimate, "warning", "") or "")
        target_wz_deg = 0.0
        limited_wz_deg = 0.0
        wz_delta_limited = False
        reverse_wz_before_min_deg = 0.0
        reverse_wz_after_min_deg = 0.0
        reverse_wz_after_limit_deg = 0.0
        reverse_heading_conflict = False
        reverse_sign_flip_blocked = False
        wz_zeroed_reason = ""
        reverse_lat_term_deg = 0.0
        reverse_heading_term_deg = 0.0
        reverse_heading_term_applied_deg = 0.0

        if abs(error_y) <= float(self.args.control_deadband_y):
            error_y = 0.0

        k_lat_eff = float(self.args.k_lat)
        k_heading_eff = float(self.args.k_heading)
        base_wz = 0.0
        wz_raw = 0.0
        lat_term_deg = 0.0
        heading_term_deg = 0.0
        vx = -abs(target_speed) if reverse else max(float(self.args.min_speed), float(target_speed))
        wz = 0.0

        if estimate.reject_reason:
            stop_reason = f"reject:{estimate.reject_reason}"
            control_phase = "emergency_stop"
            vx = 0.0
            wz_raw = 0.0
        elif reverse:
            vx = -abs(target_speed)
            k_lat_eff = float(self.args.k_reverse_lat)
            k_heading_eff = 0.50
            max_wz_deg = min(5.0, abs(float(self.args.reverse_max_wz_deg)))
            if mode in {"left_only", "right_only"}:
                max_wz_deg = min(5.0, abs(float(self.args.reverse_one_side_max_wz_deg)))
                if abs(raw_center_y) > float(self.args.raw_center_out_of_range):
                    warning = "large_one_side_center"
            if line_fit is not None:
                track_x = float(self.args.reverse_lookahead_x)
                line_y_at_track = float(line_fit[0] * track_x + line_fit[1])
                track_error_y = float(line_y_at_track - center_y_target)
                error_y = track_error_y
                heading_error = float(math.atan(line_fit[0]))
                heading_error_deg = math.degrees(heading_error)
            reverse_lat_term_deg = float(self.args.reverse_steer_sign) * (k_lat_eff * track_error_y)
            reverse_heading_term_deg = float(self.args.reverse_steer_sign) * (k_heading_eff * heading_error_deg)
            reverse_heading_term_applied_deg = reverse_heading_term_deg
            lat_term_deg = reverse_lat_term_deg
            heading_term_deg = reverse_heading_term_applied_deg
            if abs(track_error_y) >= float(self.args.reverse_heading_conflict_error_y):
                if reverse_lat_term_deg * reverse_heading_term_deg < 0.0:
                    reverse_heading_term_applied_deg = 0.0
                    reverse_heading_conflict = True
                else:
                    heading_limit_deg = abs(reverse_lat_term_deg) * float(self.args.reverse_heading_max_ratio)
                    reverse_heading_term_applied_deg = max(-heading_limit_deg, min(heading_limit_deg, reverse_heading_term_deg))
            heading_term_deg = reverse_heading_term_applied_deg
            base_wz_deg = reverse_lat_term_deg + reverse_heading_term_applied_deg
            reverse_wz_before_min_deg = base_wz_deg
            last_wz_deg = math.degrees(float(self.last_cmd_wz))
            allow_min_wz = abs(track_error_y) >= float(self.args.reverse_min_wz_error_y)
            if (
                abs(track_error_y) < float(self.args.reverse_sign_flip_guard_error_y)
                and abs(last_wz_deg) >= float(self.args.reverse_sign_flip_guard_last_wz_deg)
                and abs(base_wz_deg) >= 1e-6
                and last_wz_deg * base_wz_deg < 0.0
            ):
                reverse_sign_flip_blocked = True
                target_wz_deg = 0.0
                wz_zeroed_reason = "reverse_sign_flip_guard"
            elif abs(track_error_y) <= 0.004 and abs(heading_error_deg) <= 0.5:
                wz_zeroed_reason = "reverse_deadband"
                target_wz_deg = 0.0
            elif abs(base_wz_deg) < 1e-9:
                wz_zeroed_reason = "reverse_zero_error"
                target_wz_deg = 0.0
            else:
                sign = 1.0 if base_wz_deg >= 0.0 else -1.0
                abs_target_wz_deg = abs(base_wz_deg)
                if allow_min_wz:
                    abs_target_wz_deg = max(abs_target_wz_deg, abs(float(self.args.reverse_min_wz_deg)))
                target_wz_deg = sign * min(abs_target_wz_deg, max_wz_deg)
            reverse_wz_after_min_deg = target_wz_deg
            reverse_wz_after_limit_deg = target_wz_deg
            max_delta = max(0.0, float(self.args.max_wz_delta_deg_per_cycle))
            limited_wz_deg = max(last_wz_deg - max_delta, min(last_wz_deg + max_delta, target_wz_deg))
            wz_delta_limited = abs(limited_wz_deg - target_wz_deg) > 1e-9
            final_wz_deg = limited_wz_deg
            if target_wz_deg == 0.0 and abs(track_error_y) <= 0.004 and abs(heading_error_deg) <= 0.5:
                final_wz_deg = 0.0
            wz_raw = math.radians(target_wz_deg)
            wz = math.radians(final_wz_deg)
        else:
            if line_fit is not None:
                track_x = float(self.args.forward_lookahead_x)
                line_y_at_track = float(line_fit[0] * track_x + line_fit[1])
                track_error_y = float(line_y_at_track - center_y_target)
                target_y = line_y_at_track
                error_y = track_error_y
                heading_error = float(math.atan(line_fit[0]))
                heading_error_deg = math.degrees(heading_error)
            if one_side_mode:
                max_wz_deg = min(max_wz_deg, 0.8)
                max_wz_rad = math.radians(max_wz_deg)
                k_lat_eff *= 0.5
                k_heading_eff *= 0.3
                max_heading_wz_deg = min(max_heading_wz_deg, max_wz_deg)
                max_heading_wz_rad = math.radians(max_heading_wz_deg)
            heading_term_raw = k_heading_eff * heading_error
            heading_term_limited = max(-max_heading_wz_rad, min(max_heading_wz_rad, heading_term_raw))
            if abs(error_y) > float(self.args.heading_conflict_error_y) and (k_lat_eff * error_y) * heading_term_limited < 0.0:
                heading_term_limited *= float(self.args.heading_conflict_scale)
            base_wz = k_lat_eff * error_y + heading_term_limited
            lat_term_deg = math.degrees(k_lat_eff * error_y)
            heading_term_deg = math.degrees(heading_term_limited)
            wz_raw = base_wz
            if abs(error_y) < 0.012:
                base_wz = 0.0
                wz_raw = 0.0
                wz_zeroed_reason = "forward_deadband"
            wz_raw = max(-max_wz_rad, min(max_wz_rad, wz_raw))
            wz = wz_raw

        self.last_direct_error_y = direct_error_y
        self.last_error_y = error_y
        now = time.monotonic()
        self.direct_error_history = [(ts, value) for ts, value in self.direct_error_history if now - ts <= 3.0]
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
            track_x=track_x,
            line_y_at_track=line_y_at_track,
            track_error_y=track_error_y,
            heading_error_deg=heading_error_deg,
            k_lat_eff=k_lat_eff,
            k_heading_eff=k_heading_eff,
            lat_term_deg=lat_term_deg,
            heading_term_deg=heading_term_deg,
            target_y=target_y,
            effective_lookahead_x=effective_lookahead_x,
            requested_max_wz_deg=requested_max_wz_deg,
            effective_max_wz_deg=max_wz_deg,
            target_wz_deg=target_wz_deg,
            limited_wz_deg=limited_wz_deg,
            wz_delta_limited=wz_delta_limited,
            wz_cmd_deg=math.degrees(wz),
            wz_raw_deg=math.degrees(wz_raw),
            rolling_center_error_median=rolling_center_error_median,
            suggested_center_y_target=suggested_center_y_target,
            reverse_wz_before_min_deg=reverse_wz_before_min_deg,
            reverse_wz_after_min_deg=reverse_wz_after_min_deg,
            reverse_wz_after_limit_deg=reverse_wz_after_limit_deg,
            reverse_heading_conflict=reverse_heading_conflict,
            reverse_sign_flip_blocked=reverse_sign_flip_blocked,
            reverse_lat_term_deg=reverse_lat_term_deg,
            reverse_heading_term_deg=reverse_heading_term_deg,
            reverse_heading_term_applied_deg=reverse_heading_term_applied_deg,
            wz_zeroed_reason=wz_zeroed_reason,
        )
        return self._build_cmd(vx, wz)

    def _send_stop(self) -> None:
        self._publish_cmd(0.0, 0.0)

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
        hold_vx = float(self.last_good_cmd.linear.x)
        hold_wz_deg = math.degrees(float(self.last_good_cmd.angular.z)) * float(wz_scale)
        hold_wz_deg = max(-abs(float(wz_limit_deg)), min(abs(float(wz_limit_deg)), hold_wz_deg))
        hold_wz = math.radians(hold_wz_deg)
        self._set_debug_snapshot(
            estimate=estimate,
            found=False,
            control_phase=control_phase,
            stop_reason=stop_reason,
            final_vx=hold_vx,
            final_wz=hold_wz,
            warning=warning,
            last_good_age_sec=time.monotonic() - self.last_good_time if self.last_good_time > 0.0 else -1.0,
            last_good_error_y=float(self.last_error_y),
            last_good_heading_deg=float(self.last_good_heading_deg),
            last_good_cmd_wz_deg=math.degrees(float(self.last_good_cmd.angular.z)),
            hold_vx=hold_vx,
            hold_wz_deg=hold_wz_deg,
            control_using_last_good_line=True,
        )
        self._publish_cmd(hold_vx, hold_wz)
        return True

    def _on_control(self) -> None:
        now = time.monotonic()
        if not self.drive_enable:
            self.last_estimate = RowEstimate(found=False, mode="disabled")
            self._send_stop()
            return
        scan = self.last_scan
        scan_age = now - self.last_scan_time if self.last_scan is not None else float("inf")
        if scan is None or scan_age > float(self.args.scan_timeout):
            self.last_estimate = RowEstimate(found=False, mode="scan_timeout")
            hold_phase = "scan_timeout_hold_reverse" if self.reverse else "scan_timeout_hold_forward"
            hold_limit_deg = float(self.args.reverse_lost_soft_max_wz_deg) if self.reverse else float(self.args.forward_lost_hold_max_wz_deg)
            hold_scale = 1.0 if self.reverse else float(self.args.forward_lost_hold_wz_scale)
            if self._hold_last_good_command(
                estimate=self.last_estimate,
                control_phase=hold_phase,
                stop_reason=hold_phase,
                warning=hold_phase,
                wz_limit_deg=hold_limit_deg,
                wz_scale=hold_scale,
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
            self._send_stop()
            return

        estimate = self._estimate_row(scan)
        self.last_estimate = estimate
        if estimate.found:
            self.last_found_time = now
            self.last_good_time = now
            cmd = self._make_command(estimate)
            self.last_good_cmd = cmd
            self.last_good_estimate = estimate
            self.last_good_center_line = getattr(estimate, "center_line", None)
            self.last_good_left_line = getattr(estimate, "left_line", None)
            self.last_good_right_line = getattr(estimate, "right_line", None)
            self.last_good_mode = str(getattr(estimate, "effective_mode", estimate.mode) or "")
            self.last_good_heading_deg = math.degrees(float(getattr(estimate, "heading_rad", 0.0) or 0.0))
            self._publish_cmd(float(cmd.linear.x), float(cmd.angular.z))
            return

        last_good_age = now - self.last_good_time if self.last_good_time > 0.0 else float("inf")
        if (not self.reverse) and self._hold_last_good_command(
            estimate=estimate,
            control_phase="lost_hold_forward",
            stop_reason="lost_hold_forward",
            warning="lost_hold_forward",
            wz_limit_deg=float(self.args.forward_lost_hold_max_wz_deg),
            wz_scale=float(self.args.forward_lost_hold_wz_scale),
        ):
            return
        if self.reverse and self._hold_last_good_command(
            estimate=estimate,
            control_phase="lost_hold_reverse",
            stop_reason=estimate.reject_reason or estimate.mode or "lost_hold_reverse",
            warning=str(getattr(estimate, "warning", "") or "") or "lost_hold_reverse",
            wz_limit_deg=float(self.args.reverse_lost_soft_max_wz_deg),
            wz_scale=1.0,
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
            last_good_age_sec=last_good_age if self.last_good_time > 0.0 else -1.0,
            last_good_error_y=float(self.last_error_y),
            last_good_heading_deg=float(self.last_good_heading_deg),
            last_good_cmd_wz_deg=math.degrees(float(self.last_good_cmd.angular.z)),
            hold_vx=0.0,
            hold_wz_deg=0.0,
            control_using_last_good_line=False,
        )
        self._send_stop()

    def _publish_status(self) -> None:
        estimate = self.last_estimate
        state = "IDLE"
        if self.drive_enable:
            state = "TRACK" if estimate.found or abs(float(self.last_cmd_vx)) > 1e-6 or abs(float(self.last_cmd_wz)) > 1e-6 else "SEARCH"
        payload = {
            "state": state,
            "found": bool(estimate.found),
            "obstacle_blocked": False,
            "lost_frames": 0 if estimate.found else 1,
            "reverse": bool(self.reverse),
            "drive_enable": bool(self.drive_enable),
            "mode": estimate.mode,
            "effective_mode": str(getattr(estimate, "effective_mode", estimate.mode) or ""),
            "raw_center_y_m": round(float(estimate.raw_center_y), 4),
            "filtered_center_y_m": round(float(estimate.center_y), 4),
            "center_y_m": round(float(estimate.center_y), 4),
            "heading_deg": round(math.degrees(float(estimate.heading_rad)), 3),
            "row_width_m": round(float(estimate.row_width), 4),
            "left_bins": int(estimate.left_bins),
            "right_bins": int(estimate.right_bins),
            "candidate_bins": int(estimate.candidate_bins),
            "reject_reason": estimate.reject_reason,
            "cmd_vx_mps": round(float(self.last_cmd_vx), 4),
            "cmd_wz_rad_s": round(float(self.last_cmd_wz), 4),
            "cmd_wz_deg_s": round(math.degrees(float(self.last_cmd_wz)), 4),
        }
        if self.last_debug:
            payload.update({"debug": self.last_debug})
        self.status_pub.publish(String(data=json.dumps(payload, ensure_ascii=True)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROS runner using plant_lidar_centerline lidar logic")
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument("--cmd-vel-topic", default="/linerun/cmd_vel")
    parser.add_argument("--status-topic", default="/linerun/status")
    parser.add_argument("--drive-mode-topic", default="/linerun/drive_mode")
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
    parser.add_argument("--reverse-lookahead-x", type=float, default=-0.6)
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
    parser.add_argument("--forward-lost-hold-sec", type=float, default=0.35)
    parser.add_argument("--forward-lost-stop-sec", type=float, default=0.50)
    parser.add_argument("--forward-lost-hold-wz-scale", type=float, default=0.5)
    parser.add_argument("--forward-lost-hold-max-wz-deg", type=float, default=0.6)
    parser.add_argument("--reverse-min-speed", type=float, default=0.04)
    parser.add_argument("--reverse-one-side-speed", type=float, default=0.10)
    parser.add_argument("--reverse-both-sides-speed", type=float, default=0.12)
    parser.add_argument("--reverse-min-wz-deg", type=float, default=2.5)
    parser.add_argument("--reverse-max-wz-deg", type=float, default=5.0)
    parser.add_argument("--reverse-one-side-max-wz-deg", type=float, default=5.0)
    parser.add_argument("--reverse-wz-enable-error-y", type=float, default=0.004)
    parser.add_argument("--reverse-wz-enable-heading-deg", type=float, default=1.0)
    parser.add_argument("--reverse-min-wz-error-y", type=float, default=0.025)
    parser.add_argument("--reverse-sign-flip-guard-error-y", type=float, default=0.02)
    parser.add_argument("--reverse-sign-flip-guard-last-wz-deg", type=float, default=1.5)
    parser.add_argument("--reverse-sign-hold-error-y", type=float, default=0.0)
    parser.add_argument("--reverse-both-sides-k-lat", type=float, default=1.2)
    parser.add_argument("--reverse-both-sides-k-heading", type=float, default=0.15)
    parser.add_argument("--reverse-one-side-k-lat", type=float, default=1.0)
    parser.add_argument("--k-reverse-lat", type=float, default=25.0)
    parser.add_argument("--k-reverse-heading", type=float, default=0.03)
    parser.add_argument("--reverse-steer-sign", type=float, default=-1.0)
    parser.add_argument("--reverse-heading-conflict-error-y", type=float, default=0.01)
    parser.add_argument("--reverse-heading-max-ratio", type=float, default=0.35)
    parser.add_argument("--reverse-error-stop", type=float, default=0.28)
    parser.add_argument("--reverse-wz-smoothing-alpha", type=float, default=0.0)
    parser.add_argument("--reverse-lost-hold-sec", type=float, default=0.35)
    parser.add_argument("--reverse-lost-stop-sec", type=float, default=0.80)
    parser.add_argument("--reverse-lost-hold-max-wz-deg", type=float, default=1.5)
    parser.add_argument("--reverse-lost-soft-max-wz-deg", type=float, default=0.8)
    parser.add_argument("--max-wz-delta-deg-per-cycle", type=float, default=1.0)
    parser.add_argument("--low-beam", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = PlantRowRosRunner(args)

    def _shutdown(*_args: object) -> None:
        if rclpy.ok():
            rclpy.shutdown()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
