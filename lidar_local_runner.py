#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import signal
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from row_geometry import RowFollowerConfig, RowEstimate, estimate_row


class LidarLocalRunner(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("autorunlida_local_runner")
        self.args = args
        self.last_scan: LaserScan | None = None
        self.last_scan_time = 0.0
        self.last_estimate = RowEstimate(found=False, mode="boot")
        self.last_good_row_width = float(args.row_width)
        self.last_filtered_center_y = 0.0
        self.center_history: list[tuple[float, float]] = []
        self.drive_enable = False
        self.reverse = False
        self.cruise_vx = float(args.speed)
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
            vehicle_half_width=0.5 * float(args.vehicle_width),
            safety_margin=float(args.safety_margin),
            center_jump_reject=float(args.center_jump_reject),
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
        try:
            self.cruise_vx = abs(float(payload.get("cruise_vx", self.args.speed)))
        except Exception:
            self.cruise_vx = float(self.args.speed)

    def _estimate(self, scan: LaserScan) -> RowEstimate:
        estimate, _debug = estimate_row(scan, self.row_cfg, self.last_good_row_width)
        now = time.monotonic()
        window_s = float(self.args.history_window_s)
        self.center_history = [(ts, value) for ts, value in self.center_history if now - ts <= window_s]

        if not estimate.found:
            return estimate

        self.last_good_row_width = float(estimate.row_width or self.last_good_row_width)
        raw_center_y = float(estimate.raw_center_y)
        if abs(raw_center_y) > float(self.args.center_y_reject_abs):
            estimate.found = False
            estimate.mode = "reject"
            estimate.reject_reason = "raw_center_out_of_range"
            return estimate

        self.center_history.append((now, raw_center_y))
        if len(self.center_history) < int(self.args.min_center_history):
            estimate.found = False
            estimate.mode = "reject"
            estimate.reject_reason = "candidate_history_too_short"
            return estimate

        history_center_y = float(np.median(np.asarray([value for _ts, value in self.center_history], dtype=np.float64)))
        filtered_center_y = float(self.args.center_y_alpha) * float(self.last_filtered_center_y) + (
            1.0 - float(self.args.center_y_alpha)
        ) * history_center_y
        max_jump = float(self.args.center_y_max_jump)
        if abs(filtered_center_y - float(self.last_filtered_center_y)) > max_jump:
            filtered_center_y = float(self.last_filtered_center_y) + math.copysign(
                max_jump,
                filtered_center_y - float(self.last_filtered_center_y),
            )

        if abs(filtered_center_y) > float(self.args.center_y_reject_abs):
            estimate.found = False
            estimate.mode = "reject"
            estimate.reject_reason = "filtered_center_out_of_range"
            return estimate

        estimate.raw_center_y = history_center_y
        estimate.center_y = filtered_center_y
        self.last_filtered_center_y = filtered_center_y
        return estimate

    def _publish_cmd(self, vx: float, wz: float) -> None:
        msg = Twist()
        msg.linear.x = float(vx)
        msg.angular.z = float(wz)
        self.cmd_pub.publish(msg)

    def _make_cmd(self, estimate: RowEstimate) -> tuple[float, float]:
        mode = str(estimate.mode or "")
        one_side_mode = mode in {"left_only", "right_only", "single_side"}
        k_lat = float(self.args.k_lat)
        k_heading = float(self.args.k_heading)
        max_wz_deg = min(abs(float(self.args.max_wz_deg)), 5.0)
        max_wz_rad = math.radians(max_wz_deg)
        target_speed = float(self.cruise_vx)
        center_y_target = float(self.args.center_y_target)
        error_y = float(estimate.center_y - center_y_target)

        if one_side_mode:
            k_lat *= float(self.args.one_side_lat_scale)
            k_heading *= float(self.args.one_side_heading_scale)
            max_wz_rad *= float(self.args.one_side_wz_scale)
            target_speed *= float(self.args.one_side_speed_scale)

        if abs(error_y) <= float(self.args.control_deadband_y):
            error_y = 0.0

        lateral_term = k_lat * error_y
        heading_term = k_heading * float(estimate.heading_rad)
        wz_forward = lateral_term + heading_term
        wz = -wz_forward if self.reverse else wz_forward
        wz = max(-max_wz_rad, min(max_wz_rad, wz))

        slow_ratio = 1.0
        slow_ratio *= max(0.25, 1.0 - min(1.0, abs(error_y) / max(0.01, self.args.slow_error_y)))
        vx = target_speed * slow_ratio
        vx = max(float(self.args.min_speed), min(target_speed, vx))

        if estimate.reject_reason:
            vx = 0.0
            wz = 0.0

        safe_half_width = max(
            0.0,
            0.5 * max(float(estimate.row_width), float(self.args.row_width)) - 0.5 * float(self.args.vehicle_width),
        )
        safety_stop_band = max(0.0, safe_half_width - float(self.args.safety_margin))
        stop_error_y = float(self.args.stop_error_y)
        active_safety_stop_band = safety_stop_band
        if one_side_mode:
            stop_error_y = float(self.args.one_side_stop_error_y)
            active_safety_stop_band = max(active_safety_stop_band, float(self.args.one_side_safety_stop_band))
        if abs(error_y) > stop_error_y:
            vx = 0.0
            wz = 0.0
        elif active_safety_stop_band > 0.0 and abs(error_y) > active_safety_stop_band:
            vx = 0.0
            wz = 0.0

        if self.reverse:
            vx = -vx
        return vx, wz

    def _on_control(self) -> None:
        now = time.monotonic()
        if not self.drive_enable:
            self.last_estimate = RowEstimate(found=False, mode="disabled")
            self._publish_cmd(0.0, 0.0)
            return
        if self.last_scan is None or (now - self.last_scan_time) > float(self.args.scan_timeout):
            self.last_estimate = RowEstimate(found=False, mode="scan_timeout")
            self._publish_cmd(0.0, 0.0)
            return
        estimate = self._estimate(self.last_scan)
        self.last_estimate = estimate
        if not estimate.found:
            self._publish_cmd(0.0, 0.0)
            return
        vx, wz = self._make_cmd(estimate)
        self._publish_cmd(vx, wz)

    def _publish_status(self) -> None:
        estimate = self.last_estimate
        state = "TRACK" if self.drive_enable and estimate.found else ("IDLE" if not self.drive_enable else "SEARCH")
        payload = {
            "state": state,
            "found": bool(estimate.found),
            "obstacle_blocked": False,
            "lost_frames": 0 if estimate.found else 1,
            "reverse": bool(self.reverse),
            "drive_enable": bool(self.drive_enable),
            "mode": estimate.mode,
            "reject_reason": estimate.reject_reason,
            "raw_center_y_m": round(float(estimate.raw_center_y), 4),
            "filtered_center_y_m": round(float(estimate.center_y), 4),
            "center_y_m": round(float(estimate.center_y), 4),
            "heading_deg": round(math.degrees(float(estimate.heading_rad)), 3),
            "row_width_m": round(float(estimate.row_width), 4),
            "left_bins": int(estimate.left_bins),
            "right_bins": int(estimate.right_bins),
            "candidate_bins": int(estimate.candidate_bins),
        }
        self.status_pub.publish(String(data=json.dumps(payload, ensure_ascii=True)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="autorunlida lidar local row runner")
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument("--cmd-vel-topic", default="/linerun/cmd_vel")
    parser.add_argument("--status-topic", default="/linerun/status")
    parser.add_argument("--drive-mode-topic", default="/linerun/drive_mode")
    parser.add_argument("--speed", type=float, default=0.20)
    parser.add_argument("--min-speed", type=float, default=0.04)
    parser.add_argument("--max-wz-deg", type=float, default=1.2)
    parser.add_argument("--k-lat", type=float, default=1.2)
    parser.add_argument("--k-heading", type=float, default=0.45)
    parser.add_argument("--center-y-target", type=float, default=0.0)
    parser.add_argument("--row-width", type=float, default=0.60)
    parser.add_argument("--vehicle-width", type=float, default=0.40)
    parser.add_argument("--min-row-width", type=float, default=0.48)
    parser.add_argument("--max-row-width", type=float, default=0.78)
    parser.add_argument("--lookahead-x", type=float, default=0.75)
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
    parser.add_argument("--center-deadband", type=float, default=0.03)
    parser.add_argument("--left-percentile", type=float, default=20.0)
    parser.add_argument("--right-percentile", type=float, default=80.0)
    parser.add_argument("--sensor-yaw-deg", type=float, default=180.0)
    parser.add_argument("--control-deadband-y", type=float, default=0.004)
    parser.add_argument("--slow-error-y", type=float, default=0.04)
    parser.add_argument("--stop-error-y", type=float, default=0.065)
    parser.add_argument("--slow-heading-rad", type=float, default=0.18)
    parser.add_argument("--safety-margin", type=float, default=0.04)
    parser.add_argument("--center-jump-reject", type=float, default=0.16)
    parser.add_argument("--center-y-reject-abs", type=float, default=0.16)
    parser.add_argument("--one-side-speed-scale", type=float, default=0.55)
    parser.add_argument("--one-side-wz-scale", type=float, default=0.35)
    parser.add_argument("--one-side-lat-scale", type=float, default=0.75)
    parser.add_argument("--one-side-heading-scale", type=float, default=0.25)
    parser.add_argument("--one-side-stop-error-y", type=float, default=0.12)
    parser.add_argument("--one-side-safety-stop-band", type=float, default=0.12)
    parser.add_argument("--control-period", type=float, default=0.05)
    parser.add_argument("--status-period", type=float, default=0.20)
    parser.add_argument("--scan-timeout", type=float, default=0.30)
    parser.add_argument("--history-window-s", type=float, default=0.40)
    parser.add_argument("--min-center-history", type=int, default=2)
    parser.add_argument("--center-y-alpha", type=float, default=0.75)
    parser.add_argument("--center-y-max-jump", type=float, default=0.08)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = LidarLocalRunner(args)

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
