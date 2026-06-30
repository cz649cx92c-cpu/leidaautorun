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
    forward_min: float = 0.15
    forward_max: float = 1.20
    lateral_limit: float = 0.75
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
    center_jump_reject: float = 0.16


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
    center_points: np.ndarray | None = None


@dataclass
class RowDebugData:
    raw_points: np.ndarray
    web_points: np.ndarray
    points: np.ndarray
    left_points: np.ndarray
    right_points: np.ndarray
    center_points: np.ndarray


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
    angles = angles[valid] + math.radians(float(cfg.sensor_yaw_deg))
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    return np.column_stack((x, y))


def scan_to_points(scan, cfg: RowFollowerConfig) -> np.ndarray:
    raw_points = raw_scan_to_points(scan, cfg)
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


def estimate_row(scan, cfg: RowFollowerConfig, last_good_row_width: float) -> tuple[RowEstimate, RowDebugData]:
    raw_points = raw_scan_to_points(scan, cfg)
    web_points = _downsample_points(_exclude_robot_frame(raw_points), max_points=180)
    points = scan_to_points(scan, cfg)
    empty = RowDebugData(
        raw_points=raw_points,
        web_points=web_points,
        points=points,
        left_points=np.empty((0, 2), dtype=np.float64),
        right_points=np.empty((0, 2), dtype=np.float64),
        center_points=np.empty((0, 2), dtype=np.float64),
    )
    if len(points) < int(cfg.min_points):
        return RowEstimate(found=False, mode="too_few_points", reject_reason="too_few_points"), empty

    safe_inner = _safe_inner_limit(cfg)
    row_width_ref = float(last_good_row_width or cfg.row_width)
    candidate_centers: list[list[float]] = []
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
            width = float(left_inner_y - right_inner_y)
            if width < float(cfg.min_row_width) or width > float(cfg.max_row_width):
                reject_reasons.append("bad_width_bin")
                x_cursor = x_next
                continue
            center_y = 0.5 * (left_inner_y + right_inner_y)
            candidate_centers.append([x_mid, center_y, width, 2.0])
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

    if not candidate_centers:
        reject_reason = reject_reasons[-1] if reject_reasons else "no_candidates"
        debug = RowDebugData(
            raw_points=raw_points,
            web_points=web_points,
            points=points,
            left_points=left_points,
            right_points=right_points,
            center_points=np.empty((0, 2), dtype=np.float64),
        )
        return (
            RowEstimate(
                found=False,
                left_bins=left_bins,
                right_bins=right_bins,
                mode="no_edges",
                reject_reason=reject_reason,
            ),
            debug,
        )

    candidates = np.asarray(candidate_centers, dtype=np.float64)
    center_values = candidates[:, 1]
    row_width_values = candidates[:, 2]
    weights = candidates[:, 3]
    raw_center_y = _weighted_median(center_values, weights)
    row_width = _weighted_median(row_width_values, weights)

    if abs(raw_center_y) > float(cfg.center_jump_reject):
        debug = RowDebugData(
            raw_points=raw_points,
            web_points=web_points,
            points=points,
            left_points=left_points,
            right_points=right_points,
            center_points=candidates[:, :2],
        )
        return (
            RowEstimate(
                found=False,
                raw_center_y=raw_center_y,
                row_width=row_width,
                left_bins=left_bins,
                right_bins=right_bins,
                candidate_bins=len(candidates),
                mode="jump_reject",
                reject_reason="center_jump_too_large",
                center_points=candidates[:, :2],
            ),
            debug,
        )

    mode = "single_side"
    if left_bins >= int(cfg.min_bins) and right_bins >= int(cfg.min_bins):
        mode = "both_sides"
    elif left_bins >= int(cfg.min_bins):
        mode = "left_only"
    elif right_bins >= int(cfg.min_bins):
        mode = "right_only"

    heading_rad = 0.0
    if len(candidates) >= int(cfg.min_line_bins):
        heading_rad = _fit_heading(candidates[:, :2])

    estimate = RowEstimate(
        found=True,
        center_y=raw_center_y,
        raw_center_y=raw_center_y,
        heading_rad=heading_rad,
        row_width=row_width,
        left_bins=left_bins,
        right_bins=right_bins,
        candidate_bins=len(candidates),
        mode=mode,
        reject_reason="",
        center_points=candidates[:, :2],
    )
    debug = RowDebugData(
        raw_points=raw_points,
        web_points=web_points,
        points=points,
        left_points=left_points,
        right_points=right_points,
        center_points=candidates[:, :2],
    )
    return estimate, debug
