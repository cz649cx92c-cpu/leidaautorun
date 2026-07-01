#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

ROBOT_FRAME_FRONT = 0.22
ROBOT_FRAME_BACK = 0.40
ROBOT_FRAME_LEFT = 0.20
ROBOT_FRAME_RIGHT = 0.20


@dataclass
class RowFollowerConfig:
    row_width: float = 0.60
    min_row_width: float = 0.45
    max_row_width: float = 0.80
    lookahead_x: float = 0.60
    forward_min: float = 0.25
    forward_max: float = 1.20
    lateral_limit: float = 0.60
    range_min: float = 0.05
    range_max: float = 6.0
    bin_size: float = 0.20
    min_points: int = 8
    min_bins: int = 2
    min_line_bins: int = 4
    min_side_points_per_bin: int = 2
    center_deadband: float = 0.03
    left_percentile: float = 30.0
    right_percentile: float = 70.0
    sensor_yaw_deg: float = 0.0
    vehicle_half_width: float = 0.20
    safety_margin: float = 0.04
    center_jump_reject: float = 0.25
    one_side_center_jump_reject: float = 0.30
    lidar_yaw_correction_deg: float = 0.0
    lidar_x_offset_m: float = 0.0
    lidar_y_offset_m: float = 0.0
    boundary_max_gap_x: float = 0.45
    boundary_width_tolerance_m: float = 0.0


@dataclass
class RowEstimate:
    found: bool
    center_y: float = 0.0
    raw_center_y: float = 0.0
    heading_rad: float = 0.0
    row_width: float = 0.0
    left_bins: int = 0
    right_bins: int = 0
    candidate_bins: int = 0
    mode: str = "lost"
    reject_reason: str = ""
    warning: str = ""
    center_points: np.ndarray | None = None
    effective_mode: str = ""
    left_line: tuple[float, float] | None = None
    right_line: tuple[float, float] | None = None
    center_line: tuple[float, float] | None = None
    raw_points_count: int = 0
    filtered_points_count: int = 0
    left_points_count: int = 0
    right_points_count: int = 0
    center_jump: float = 0.0
    heading_jump_deg: float = 0.0
    center_jump_rejected: bool = False
    heading_jump_rejected: bool = False
    left_valid: bool = False
    right_valid: bool = False
    left_reject_reason: str = ""
    right_reject_reason: str = ""
    parallel_angle_diff_deg: float = 0.0
    width_error_m: float = 0.0
    left_residual_median: float = 0.0
    right_residual_median: float = 0.0
    left_consecutive_bins: int = 0
    right_consecutive_bins: int = 0
    boundary_source: str = ""


@dataclass
class RowDebugData:
    raw_points: np.ndarray
    web_points: np.ndarray
    points: np.ndarray
    left_points: np.ndarray
    right_points: np.ndarray
    center_points: np.ndarray
    virtual_left_points: np.ndarray
    virtual_right_points: np.ndarray
    virtual_center_points: np.ndarray
    left_line: tuple[float, float] | None = None
    right_line: tuple[float, float] | None = None
    center_line: tuple[float, float] | None = None
    candidate_left_line: tuple[float, float] | None = None
    candidate_right_line: tuple[float, float] | None = None


def _empty_debug(raw_points: np.ndarray, web_points: np.ndarray, points: np.ndarray) -> RowDebugData:
    return RowDebugData(
        raw_points=raw_points,
        web_points=web_points,
        points=points,
        left_points=np.empty((0, 2), dtype=np.float64),
        right_points=np.empty((0, 2), dtype=np.float64),
        center_points=np.empty((0, 2), dtype=np.float64),
        virtual_left_points=np.empty((0, 2), dtype=np.float64),
        virtual_right_points=np.empty((0, 2), dtype=np.float64),
        virtual_center_points=np.empty((0, 2), dtype=np.float64),
        left_line=None,
        right_line=None,
        center_line=None,
        candidate_left_line=None,
        candidate_right_line=None,
    )


def _exclude_robot_frame(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points
    x = points[:, 0]
    y = points[:, 1]
    mask = ~(
        (x > -ROBOT_FRAME_BACK)
        & (x < ROBOT_FRAME_FRONT)
        & (y > -ROBOT_FRAME_RIGHT)
        & (y < ROBOT_FRAME_LEFT)
    )
    return points[mask]


def _downsample_points(points: np.ndarray, max_points: int = 180) -> np.ndarray:
    if points.size == 0 or len(points) <= max_points:
        return points
    step = max(1, len(points) // max_points)
    return points[::step]


def raw_scan_to_points(scan, cfg: RowFollowerConfig) -> np.ndarray:
    ranges = np.asarray(scan.ranges, dtype=np.float64)
    if ranges.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    angles = scan.angle_min + np.arange(ranges.size, dtype=np.float64) * scan.angle_increment
    valid = np.isfinite(ranges)
    valid &= ranges >= max(float(scan.range_min), float(cfg.range_min))
    valid &= ranges <= min(float(scan.range_max), float(cfg.range_max))
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.float64)
    ranges = ranges[valid]
    angles = angles[valid]
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    points = np.column_stack((x, y))
    return transform_lidar_points(
        points,
        sensor_yaw_deg=float(cfg.sensor_yaw_deg),
        lidar_yaw_correction_deg=float(cfg.lidar_yaw_correction_deg),
        lidar_x_offset_m=float(cfg.lidar_x_offset_m),
        lidar_y_offset_m=float(cfg.lidar_y_offset_m),
    )


def transform_lidar_points(
    points: np.ndarray,
    sensor_yaw_deg: float,
    lidar_yaw_correction_deg: float,
    lidar_x_offset_m: float,
    lidar_y_offset_m: float,
) -> np.ndarray:
    if points.size == 0:
        return points
    total_yaw = math.radians(float(sensor_yaw_deg) + float(lidar_yaw_correction_deg))
    c = math.cos(total_yaw)
    s = math.sin(total_yaw)
    xr = c * points[:, 0] - s * points[:, 1]
    yr = s * points[:, 0] + c * points[:, 1]
    xr = xr + float(lidar_x_offset_m)
    yr = yr + float(lidar_y_offset_m)
    return np.column_stack((xr, yr))


def scan_to_points(scan, cfg: RowFollowerConfig) -> np.ndarray:
    raw_points = raw_scan_to_points(scan, cfg)
    return filter_points_for_row(raw_points, cfg)


def filter_points_for_row(raw_points: np.ndarray, cfg: RowFollowerConfig) -> np.ndarray:
    if raw_points.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    x = raw_points[:, 0]
    y = raw_points[:, 1]
    mask = x >= float(cfg.forward_min)
    mask &= x <= float(cfg.forward_max)
    mask &= np.abs(y) <= float(cfg.lateral_limit)
    if not np.any(mask):
        return np.empty((0, 2), dtype=np.float64)
    return raw_points[mask]


def _safe_inner_limit(cfg: RowFollowerConfig) -> float:
    return float(cfg.vehicle_half_width) + float(cfg.safety_margin)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights)
    cutoff = 0.5 * float(cdf[-1])
    idx = int(np.searchsorted(cdf, cutoff, side="left"))
    idx = max(0, min(idx, len(values) - 1))
    return float(values[idx])


def _fit_heading(center_points: np.ndarray) -> float:
    if len(center_points) < 2:
        return 0.0
    order = np.argsort(center_points[:, 0])
    pts = center_points[order]
    if float(pts[:, 0].max() - pts[:, 0].min()) < 0.20:
        return 0.0
    coeffs = np.polyfit(pts[:, 0], pts[:, 1], deg=1)
    return float(math.atan(float(coeffs[0])))


def _fit_line(points: np.ndarray) -> tuple[float, float] | None:
    if points is None or len(points) < 2:
        return None
    order = np.argsort(points[:, 0])
    pts = points[order]
    if float(pts[:, 0].max() - pts[:, 0].min()) < 0.20:
        return None
    coeffs = np.polyfit(pts[:, 0], pts[:, 1], deg=1)
    return (float(coeffs[0]), float(coeffs[1]))


def _line_stats(points: np.ndarray, line: tuple[float, float] | None, bin_size: float) -> tuple[float, int, float, float]:
    if points is None or len(points) == 0 or line is None:
        return float("inf"), 0, float("inf"), 0.0
    a, b = line
    residuals = np.abs(points[:, 1] - (a * points[:, 0] + b))
    xs = np.sort(points[:, 0])
    gaps = np.diff(xs) if len(xs) >= 2 else np.asarray([], dtype=np.float64)
    max_gap = float(np.max(gaps)) if gaps.size else float("inf")
    line_length = float(xs[-1] - xs[0]) if len(xs) >= 2 else 0.0
    consecutive = 1 if len(xs) > 0 else 0
    best = consecutive
    for gap in gaps:
        if gap <= float(bin_size) * 1.25:
            consecutive += 1
        else:
            best = max(best, consecutive)
            consecutive = 1
    best = max(best, consecutive)
    return float(np.median(residuals)), int(best), max_gap, line_length


def _validate_side(points: np.ndarray, line: tuple[float, float] | None, bin_size: float) -> tuple[bool, str, float, int, float, float]:
    residual_median, consecutive_bins, max_gap_x, line_length = _line_stats(points, line, bin_size)
    if points is None or len(points) < 4:
        return False, "too_few_points", residual_median, consecutive_bins, max_gap_x, line_length
    if consecutive_bins < 3:
        return False, "too_few_consecutive_bins", residual_median, consecutive_bins, max_gap_x, line_length
    if residual_median >= 0.04:
        return False, "high_residual", residual_median, consecutive_bins, max_gap_x, line_length
    if max_gap_x > max(0.45, float(bin_size) * 2.25):
        return False, "gap_too_large", residual_median, consecutive_bins, max_gap_x, line_length
    if line_length <= 0.35:
        return False, "line_too_short", residual_median, consecutive_bins, max_gap_x, line_length
    return True, "", residual_median, consecutive_bins, max_gap_x, line_length


def _soft_side_ok(
    points: np.ndarray,
    reject_reason: str,
    residual_median: float,
    consecutive_bins: int,
) -> bool:
    if points is None or len(points) < 4:
        return False
    if residual_median >= 0.04:
        return False
    if consecutive_bins < 3:
        return False
    return reject_reason in {"gap_too_large", "too_few_consecutive_bins"}


def blend_line(
    previous: tuple[float, float] | None,
    current: tuple[float, float] | None,
    keep_ratio: float = 0.7,
) -> tuple[float, float] | None:
    if current is None:
        return previous
    if previous is None:
        return current
    keep = max(0.0, min(1.0, float(keep_ratio)))
    new_ratio = 1.0 - keep
    return (
        keep * float(previous[0]) + new_ratio * float(current[0]),
        keep * float(previous[1]) + new_ratio * float(current[1]),
    )


def estimate_row_from_points(
    raw_points: np.ndarray,
    cfg: RowFollowerConfig,
    last_good_row_width: float,
) -> tuple[RowEstimate, RowDebugData]:
    web_points = _downsample_points(_exclude_robot_frame(raw_points), max_points=180)
    points = filter_points_for_row(raw_points, cfg)
    empty = _empty_debug(raw_points, web_points, points)
    if len(points) < int(cfg.min_points):
        return (
            RowEstimate(
                found=False,
                mode="too_few_points",
                reject_reason="too_few_points",
                raw_points_count=int(len(raw_points)),
                filtered_points_count=int(len(points)),
            ),
            empty,
        )

    safe_inner = _safe_inner_limit(cfg)
    del last_good_row_width
    row_width_ref = float(cfg.row_width)
    candidate_centers: list[list[float]] = []
    paired_center_samples: list[list[float]] = []
    left_samples: list[list[float]] = []
    right_samples: list[list[float]] = []
    reject_reasons: list[str] = []
    left_bins = 0
    right_bins = 0

    x_cursor = float(cfg.forward_min)
    while x_cursor < float(cfg.forward_max):
        x_next = min(float(cfg.forward_max), x_cursor + float(cfg.bin_size))
        x_mid = 0.5 * (x_cursor + x_next)
        mask = (points[:, 0] >= x_cursor) & (points[:, 0] < x_next)
        chunk = points[mask]
        if len(chunk) == 0:
            x_cursor = x_next
            continue

        left_chunk = chunk[chunk[:, 1] >= safe_inner]
        right_chunk = chunk[chunk[:, 1] <= -safe_inner]

        left_inner_y: float | None = None
        right_inner_y: float | None = None

        if len(left_chunk) >= int(cfg.min_side_points_per_bin):
            left_inner_y = float(np.percentile(left_chunk[:, 1], float(cfg.left_percentile)))
            left_samples.append([x_mid, left_inner_y])
            left_bins += 1
        if len(right_chunk) >= int(cfg.min_side_points_per_bin):
            right_inner_y = float(np.percentile(right_chunk[:, 1], float(cfg.right_percentile)))
            right_samples.append([x_mid, right_inner_y])
            right_bins += 1

        if left_inner_y is not None and right_inner_y is not None:
            center_y = 0.5 * (left_inner_y + right_inner_y)
            candidate_centers.append([x_mid, center_y, row_width_ref, 2.0])
            paired_center_samples.append([x_mid, center_y])
        elif left_inner_y is not None:
            center_y = float(left_inner_y - 0.5 * row_width_ref)
            candidate_centers.append([x_mid, center_y, row_width_ref, 1.0])
        elif right_inner_y is not None:
            center_y = float(right_inner_y + 0.5 * row_width_ref)
            candidate_centers.append([x_mid, center_y, row_width_ref, 1.0])
        else:
            reject_reasons.append("empty_bin")

        x_cursor = x_next

    left_points = np.asarray(left_samples, dtype=np.float64) if left_samples else np.empty((0, 2), dtype=np.float64)
    right_points = np.asarray(right_samples, dtype=np.float64) if right_samples else np.empty((0, 2), dtype=np.float64)
    candidate_left_line = _fit_line(left_points)
    candidate_right_line = _fit_line(right_points)
    left_line = candidate_left_line
    right_line = candidate_right_line
    left_valid, left_reject_reason, left_residual_median, left_consecutive_bins, _left_max_gap_x, _left_line_length = _validate_side(
        left_points, left_line, float(cfg.bin_size)
    )
    right_valid, right_reject_reason, right_residual_median, right_consecutive_bins, _right_max_gap_x, _right_line_length = _validate_side(
        right_points, right_line, float(cfg.bin_size)
    )
    parallel_angle_diff_deg = 0.0
    width_error_m = 0.0

    if left_valid and right_valid and left_line is not None and right_line is not None:
        parallel_angle_diff_deg = abs(math.degrees(math.atan(left_line[0])) - math.degrees(math.atan(right_line[0])))
        if parallel_angle_diff_deg > 10.0:
            if left_residual_median <= right_residual_median:
                right_valid = False
                right_reject_reason = "not_parallel"
                right_line = None
            else:
                left_valid = False
                left_reject_reason = "not_parallel"
                left_line = None
        if left_valid and right_valid and left_line is not None and right_line is not None:
            sample_xs = np.linspace(max(float(cfg.forward_min), 0.3), min(float(cfg.forward_max), 1.1), num=4)
            width_errors = []
            for x in sample_xs:
                width_i = abs((left_line[0] * x + left_line[1]) - (right_line[0] * x + right_line[1]))
                width_errors.append(abs(width_i - row_width_ref))
            width_error_m = float(max(width_errors)) if width_errors else 0.0
            if float(cfg.boundary_width_tolerance_m) > 0.0 and width_error_m > float(cfg.boundary_width_tolerance_m):
                if left_residual_median <= right_residual_median:
                    right_valid = False
                    right_reject_reason = "width_mismatch"
                    right_line = None
                else:
                    left_valid = False
                    left_reject_reason = "width_mismatch"
                    left_line = None

    if not left_valid and not right_valid:
        left_soft_ok = _soft_side_ok(left_points, left_reject_reason, left_residual_median, left_consecutive_bins)
        right_soft_ok = _soft_side_ok(right_points, right_reject_reason, right_residual_median, right_consecutive_bins)
        if left_soft_ok or right_soft_ok:
            if left_soft_ok and right_soft_ok:
                if left_residual_median <= right_residual_median:
                    left_valid = True
                    left_reject_reason = ""
                    right_line = None
                else:
                    right_valid = True
                    right_reject_reason = ""
                    left_line = None
            elif left_soft_ok:
                left_valid = True
                left_reject_reason = ""
                right_line = None
            else:
                right_valid = True
                right_reject_reason = ""
                left_line = None

    if not candidate_centers:
        reject_reason = reject_reasons[-1] if reject_reasons else "no_candidates"
        debug = _empty_debug(raw_points, web_points, points)
        debug.left_points = left_points
        debug.right_points = right_points
        debug.candidate_left_line = candidate_left_line
        debug.candidate_right_line = candidate_right_line
        debug.left_line = left_line
        debug.right_line = right_line
        return (
            RowEstimate(
                found=False,
                left_bins=left_bins,
                right_bins=right_bins,
                mode="no_edges",
                reject_reason=reject_reason,
                raw_points_count=int(len(raw_points)),
                filtered_points_count=int(len(points)),
                left_points_count=int(len(left_points)),
                right_points_count=int(len(right_points)),
                left_valid=left_valid,
                right_valid=right_valid,
                left_reject_reason=left_reject_reason,
                right_reject_reason=right_reject_reason,
                parallel_angle_diff_deg=parallel_angle_diff_deg,
                width_error_m=width_error_m,
                left_residual_median=left_residual_median if np.isfinite(left_residual_median) else 0.0,
                right_residual_median=right_residual_median if np.isfinite(right_residual_median) else 0.0,
                left_consecutive_bins=left_consecutive_bins,
                right_consecutive_bins=right_consecutive_bins,
                boundary_source="reject",
            ),
            debug,
        )

    mode = "reject"
    if left_valid and right_valid:
        mode = "both_sides"
    elif left_valid:
        mode = "left_only"
    elif right_valid:
        mode = "right_only"
    effective_mode = mode

    row_width = row_width_ref
    virtual_left_points = np.empty((0, 2), dtype=np.float64)
    virtual_right_points = np.empty((0, 2), dtype=np.float64)
    virtual_center_points = np.empty((0, 2), dtype=np.float64)
    if mode == "left_only" and len(left_points) > 0:
        virtual_right_points = left_points.copy()
        virtual_right_points[:, 1] -= row_width
        virtual_center_points = np.column_stack((left_points[:, 0], left_points[:, 1] - 0.5 * row_width))
    elif mode == "right_only" and len(right_points) > 0:
        virtual_left_points = right_points.copy()
        virtual_left_points[:, 1] += row_width
        virtual_center_points = np.column_stack((right_points[:, 0], right_points[:, 1] + 0.5 * row_width))
    paired_center_points = (
        np.asarray(paired_center_samples, dtype=np.float64)
        if paired_center_samples
        else np.empty((0, 2), dtype=np.float64)
    )
    if effective_mode == "both_sides" and len(paired_center_points) >= 2:
        center_fit_points = paired_center_points
    elif effective_mode in {"left_only", "right_only"} and len(virtual_center_points) >= 2:
        center_fit_points = virtual_center_points
    else:
        center_fit_points = np.empty((0, 2), dtype=np.float64)
    center_line = _fit_line(center_fit_points)
    if effective_mode == "left_only" and left_line is not None:
        center_line = (left_line[0], left_line[1] - 0.5 * row_width)
        right_line = (left_line[0], left_line[1] - row_width)
    elif effective_mode == "right_only" and right_line is not None:
        center_line = (right_line[0], right_line[1] + 0.5 * row_width)
        left_line = (right_line[0], right_line[1] + row_width)

    if center_line is not None:
        reference_x = max(float(cfg.forward_min), min(float(cfg.forward_max), float(cfg.lookahead_x)))
        center_y = float(center_line[0] * reference_x + center_line[1])
        heading_rad = float(math.atan(center_line[0]))
    else:
        candidates = np.asarray(candidate_centers, dtype=np.float64)
        center_values = candidates[:, 1]
        row_width_values = candidates[:, 2]
        weights = candidates[:, 3]
        center_y = _weighted_median(center_values, weights)
        row_width = _weighted_median(row_width_values, weights)
        heading_rad = _fit_heading(candidates[:, :2]) if len(candidates) >= int(cfg.min_line_bins) else 0.0

    center_points = center_fit_points if len(center_fit_points) else (
        np.asarray(candidate_centers, dtype=np.float64)[:, :2] if candidate_centers else np.empty((0, 2), dtype=np.float64)
    )
    estimate = RowEstimate(
        found=True,
        center_y=center_y,
        raw_center_y=center_y,
        heading_rad=heading_rad,
        row_width=row_width,
        left_bins=left_bins,
        right_bins=right_bins,
        candidate_bins=len(candidate_centers),
        mode=mode,
        reject_reason="",
        warning="",
        center_points=center_points,
        effective_mode=effective_mode,
        left_line=left_line,
        right_line=right_line,
        center_line=center_line,
        raw_points_count=int(len(raw_points)),
        filtered_points_count=int(len(points)),
        left_points_count=int(len(left_points)),
        right_points_count=int(len(right_points)),
        left_valid=left_valid,
        right_valid=right_valid,
        left_reject_reason=left_reject_reason,
        right_reject_reason=right_reject_reason,
        parallel_angle_diff_deg=parallel_angle_diff_deg,
        width_error_m=width_error_m,
        left_residual_median=left_residual_median if np.isfinite(left_residual_median) else 0.0,
        right_residual_median=right_residual_median if np.isfinite(right_residual_median) else 0.0,
        left_consecutive_bins=left_consecutive_bins,
        right_consecutive_bins=right_consecutive_bins,
        boundary_source=effective_mode,
    )
    debug = _empty_debug(raw_points, web_points, points)
    debug.left_points = left_points
    debug.right_points = right_points
    debug.center_points = center_points
    debug.virtual_left_points = virtual_left_points
    debug.virtual_right_points = virtual_right_points
    debug.virtual_center_points = virtual_center_points
    debug.left_line = left_line
    debug.right_line = right_line
    debug.center_line = center_line
    debug.candidate_left_line = candidate_left_line
    debug.candidate_right_line = candidate_right_line
    return estimate, debug


def estimate_row(scan, cfg: RowFollowerConfig, last_good_row_width: float) -> tuple[RowEstimate, RowDebugData]:
    raw_points = raw_scan_to_points(scan, cfg)
    return estimate_row_from_points(raw_points, cfg, last_good_row_width)
