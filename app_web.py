#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import queue
import re
import signal
import socket
import subprocess
import shutil
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np
from sensor_msgs.msg import LaserScan as RosLaserScan

ROOT = Path(__file__).resolve().parent.parent
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import (  # type: ignore
    DEFAULT_BODY_X_OFFSET_M,
    DEFAULT_BODY_Y_OFFSET_M,
    DEFAULT_PITCH_GAIN,
    DEFAULT_ROLL_GAIN,
    DEFAULT_SENSOR_HEIGHT_M,
    MAIN_SCRIPT,
    MAPDATA_DIR,
    MISSIONS_DIR,
    PROJECT_ROOT,
    ProcessWorker,
    ROS_PYTHON,
    RosImageMonitor,
    RosPoseDebugMonitor,
    SETTINGS_PATH,
    UVC_PREVIEW_SCRIPT,
    UVC_PREVIEW_TOPIC,
    ensure_ros_monitor_node,
    generated_name,
    missions_for_map,
    normalize_device_path,
    now_text,
    project_ground_pose,
)
from control.official_fwmini_compat import get_bridge
from row_geometry import RowFollowerConfig, estimate_row
from yhs_can_interfaces.msg import ChassisInfoFb


HOST = "0.0.0.0"
PORT = 8765
LOG_LIMIT = 250
ODIN_USB_VENDOR = "2207"
ODIN_USB_PRODUCT = "0019"
DEFAULT_LIDAR_YAW_CORRECTION_DEG = "-3.0"
DEFAULT_LIDAR_X_OFFSET_M = "0.035"
DEFAULT_LIDAR_Y_OFFSET_M = "0.0"
LIDAR_SCAN_TOPIC = "/scan"
LIDAR_CALIBRATION_PATH = PROJECT_ROOT / "config" / "lidar_calibration.json"
LIDAR_DRIVER_ROOT = Path("/home/orangepi/ugv")
LIDAR_DRIVER_BIN = LIDAR_DRIVER_ROOT / "install" / "lidar_pkg" / "lib" / "lidar_pkg" / "lidar_node"
LIDAR_DRIVER_PARAMS = (
    LIDAR_DRIVER_ROOT / "install" / "lidar_pkg" / "share" / "lidar_pkg" / "config" / "lidar_params.yaml"
)
GAMEPAD_COMMAND_TIMEOUT_S = 0.45
GAMEPAD_CONTROL_PERIOD_S = 0.05
GAMEPAD_SPEED_LIMITS: dict[str, tuple[float, float]] = {
    "low": (0.15, 25.0),
    "medium": (0.30, 50.0),
    "high": (0.50, 75.0),
}


def load_lidar_preview_calibration() -> dict[str, float]:
    calibration = {
        "lidar_yaw_correction_deg": float(DEFAULT_LIDAR_YAW_CORRECTION_DEG),
        "lidar_x_offset_m": float(DEFAULT_LIDAR_X_OFFSET_M),
        "lidar_y_offset_m": float(DEFAULT_LIDAR_Y_OFFSET_M),
    }
    try:
        payload = json.loads(LIDAR_CALIBRATION_PATH.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in tuple(calibration):
                value = float(payload.get(key, calibration[key]))
                if math.isfinite(value):
                    calibration[key] = value
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass
    return calibration


DEFAULT_SETTINGS: dict[str, Any] = {
    "can_channel": "can0",
    "can_bitrate": "500000",
    "sensor_height_m": DEFAULT_SENSOR_HEIGHT_M,
    "body_x_offset_m": DEFAULT_BODY_X_OFFSET_M,
    "body_y_offset_m": DEFAULT_BODY_Y_OFFSET_M,
    "roll_gain": DEFAULT_ROLL_GAIN,
    "pitch_gain": DEFAULT_PITCH_GAIN,
    "line_cruise_vx": "0.12",
    "line_target_center_offset_px": "0",
    "line_vehicle_direction_angle_deg": "0.0",
    "line_steer_sign": "-1.0",
    "line_kp_offset": "7.0",
    "line_kp_heading": "0.08",
    "line_max_wz": "1.6",
    "mapping_recorddata": False,
}


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>autorun_final Console</title>
  <style>
    :root {
      color-scheme: dark light;
      --radius-xl: 28px;
      --radius-lg: 20px;
      --radius-md: 16px;
      --radius-sm: 12px;
      --shadow-xl: 0 26px 70px rgba(8, 15, 30, 0.22);
      --shadow-lg: 0 16px 40px rgba(8, 15, 30, 0.16);
      --shadow-md: 0 8px 24px rgba(8, 15, 30, 0.10);
      --ring: rgba(94, 139, 255, 0.26);
      --trans-fast: 160ms ease;
      --trans-med: 260ms ease;
    }
    body[data-theme="dark"] {
      --bg: #07101d;
      --bg-soft: #0d1626;
      --bg-elev: #101b2d;
      --panel: rgba(11, 20, 34, 0.82);
      --panel-2: rgba(17, 28, 46, 0.90);
      --panel-3: rgba(23, 36, 58, 0.92);
      --line: rgba(89, 103, 132, 0.22);
      --line-strong: rgba(109, 127, 156, 0.34);
      --text: #e7eef9;
      --muted: #93a3bf;
      --accent: #5e8bff;
      --accent-2: #33b6a5;
      --accent-3: #a774ff;
      --warn: #e26b77;
      --good: #29c575;
      --map-tint: rgba(94, 139, 255, 0.14);
      --loc-tint: rgba(51, 182, 165, 0.14);
      --record-tint: rgba(234, 175, 72, 0.14);
      --drive-tint: rgba(167, 116, 255, 0.14);
      --input-bg: rgba(7, 14, 26, 0.78);
      --hero-a: rgba(15, 24, 42, 0.96);
      --hero-b: rgba(8, 14, 24, 0.98);
      --mesh-a: rgba(94, 139, 255, 0.11);
      --mesh-b: rgba(51, 182, 165, 0.09);
      --mesh-c: rgba(167, 116, 255, 0.08);
    }
    body[data-theme="light"] {
      --bg: #edf3fb;
      --bg-soft: #f6f9fd;
      --bg-elev: #ffffff;
      --panel: rgba(255, 255, 255, 0.82);
      --panel-2: rgba(248, 250, 252, 0.96);
      --panel-3: rgba(242, 246, 251, 0.98);
      --line: rgba(148, 163, 184, 0.22);
      --line-strong: rgba(121, 137, 160, 0.34);
      --text: #122033;
      --muted: #5b6d85;
      --accent: #466fff;
      --accent-2: #16998b;
      --accent-3: #845df2;
      --warn: #d74f62;
      --good: #1fa568;
      --map-tint: rgba(70, 111, 255, 0.10);
      --loc-tint: rgba(22, 153, 139, 0.10);
      --record-tint: rgba(213, 145, 38, 0.12);
      --drive-tint: rgba(132, 93, 242, 0.12);
      --input-bg: rgba(255, 255, 255, 0.92);
      --hero-a: rgba(255, 255, 255, 0.92);
      --hero-b: rgba(244, 248, 252, 0.96);
      --mesh-a: rgba(70, 111, 255, 0.09);
      --mesh-b: rgba(22, 153, 139, 0.06);
      --mesh-c: rgba(132, 93, 242, 0.06);
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at 14% 12%, var(--mesh-a), transparent 24%),
        radial-gradient(circle at 88% 18%, var(--mesh-b), transparent 22%),
        radial-gradient(circle at 78% 78%, var(--mesh-c), transparent 18%),
        linear-gradient(180deg, var(--bg) 0%, var(--bg-soft) 100%);
      transition: background var(--trans-med), color var(--trans-fast);
    }
    .page {
      min-height: 100vh;
      padding: 20px;
      display: grid;
      gap: 18px;
    }
    h1, h2, h3, p { margin: 0; }
    h1 { font-size: clamp(28px, 3vw, 38px); line-height: 1.05; letter-spacing: 0; }
    h2 { font-size: 17px; line-height: 1.25; }
    h3 { font-size: 14px; line-height: 1.3; }
    .shell {
      display: grid;
      grid-template-columns: minmax(0, 1.5fr) minmax(310px, 0.85fr);
      gap: 18px;
      align-items: stretch;
    }
    .hero, .sidebar, .panel, .workflow-step, .metric-card, .console-card {
      border: 1px solid var(--line);
      background: var(--panel);
      backdrop-filter: blur(16px);
      box-shadow: var(--shadow-lg);
    }
    .hero {
      position: relative;
      overflow: hidden;
      border-radius: var(--radius-xl);
      padding: 24px;
      background:
        linear-gradient(180deg, var(--hero-a), var(--hero-b)),
        linear-gradient(135deg, rgba(255,255,255,0.02), transparent);
    }
    .hero::after {
      content: "";
      position: absolute;
      inset: auto -12% -34% 34%;
      height: 240px;
      background: radial-gradient(circle, rgba(94, 139, 255, 0.20), transparent 62%);
      pointer-events: none;
      filter: blur(18px);
    }
    .hero-top {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      flex-wrap: wrap;
      position: relative;
      z-index: 1;
    }
    .eyebrow {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      color: var(--accent);
      margin-bottom: 10px;
      font-weight: 700;
    }
    .subtitle {
      margin-top: 10px;
      max-width: 720px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.6;
    }
    .hero-actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
    }
    .pill, .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-height: 36px;
      padding: 0 12px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.04);
      font-size: 12px;
      font-weight: 600;
      color: var(--text);
      white-space: nowrap;
    }
    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: var(--good);
      box-shadow: 0 0 0 4px rgba(34, 197, 94, 0.12);
      flex: 0 0 auto;
    }
    .hero-stats {
      margin-top: 22px;
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      position: relative;
      z-index: 1;
    }
    .metric-card {
      padding: 16px;
      border-radius: var(--radius-lg);
      background: rgba(255,255,255,0.03);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
    }
    .metric-label {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.10em;
      color: var(--muted);
      margin-bottom: 8px;
      font-weight: 700;
    }
    .metric-value {
      font-size: 15px;
      line-height: 1.4;
      word-break: break-word;
    }
    .sidebar {
      border-radius: var(--radius-xl);
      padding: 18px;
      display: grid;
      gap: 14px;
      align-content: start;
    }
    .stack { display: grid; gap: 14px; }
    .panel {
      border-radius: var(--radius-lg);
      padding: 18px;
    }
    .panel-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
      margin-bottom: 14px;
    }
    .panel-sub {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.55;
      margin-top: 4px;
    }
    .route-preview-wrap {
      margin-top: 10px;
      border-radius: var(--radius-md);
      border: 1px solid var(--line);
      background: var(--input-bg);
      padding: 12px;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
    }
    .route-preview-wrap canvas {
      width: 100%;
      height: 260px;
      display: block;
      border-radius: 12px;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.00)),
        color-mix(in srgb, var(--bg) 92%, black);
    }
    .mini-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .mini-card {
      padding: 13px 14px;
      border-radius: var(--radius-md);
      border: 1px solid var(--line);
      background: var(--panel-2);
      display: grid;
      gap: 6px;
    }
    .mini-card .label {
      font-size: 11px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .mini-card .value {
      font-size: 14px;
      line-height: 1.4;
    }
    .segmented {
      display: inline-flex;
      gap: 6px;
      padding: 6px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: var(--panel-2);
      box-shadow: var(--shadow-md);
      width: fit-content;
      max-width: 100%;
      flex-wrap: wrap;
    }
    .segmented button {
      width: auto;
      min-width: 108px;
      min-height: 38px;
      padding: 0 14px;
      border-radius: 999px;
      border: 1px solid transparent;
      background: transparent;
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
      cursor: pointer;
      transition: background var(--trans-fast), color var(--trans-fast), transform var(--trans-fast), box-shadow var(--trans-fast);
    }
    .segmented button.active {
      background: linear-gradient(180deg, color-mix(in srgb, var(--accent) 95%, white), color-mix(in srgb, var(--accent) 82%, black));
      color: white;
      transform: translateY(-1px);
      box-shadow: 0 10px 24px rgba(70, 111, 255, 0.22);
    }
    .tab-panel {
      display: none;
      opacity: 0;
      transform: translateY(10px);
    }
    .tab-panel.active {
      display: block;
      opacity: 1;
      transform: translateY(0);
      animation: panelIn 260ms ease;
    }
    @keyframes panelIn {
      from { opacity: 0; transform: translateY(12px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .dashboard-layout {
      display: grid;
      grid-template-columns: minmax(0, 1.34fr) minmax(320px, 0.86fr);
      gap: 18px;
      align-items: start;
    }
    .preview-stage {
      overflow: hidden;
      border-radius: var(--radius-lg);
      border: 1px solid var(--line);
      background: color-mix(in srgb, var(--bg) 90%, black);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
    }
    .preview {
      width: 100%;
      aspect-ratio: 16 / 9;
      object-fit: contain;
      display: block;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.02), transparent),
        color-mix(in srgb, var(--bg) 92%, black);
    }
    .preview-footer {
      padding: 12px 14px;
      display: flex;
      justify-content: space-between;
      gap: 10px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 12px;
      border-top: 1px solid var(--line);
    }
    .preview-footer strong { color: var(--text); }
    .toast-stack {
      position: fixed;
      top: 18px;
      right: 18px;
      z-index: 1200;
      display: grid;
      gap: 10px;
      pointer-events: none;
    }
    .toast {
      min-width: 260px;
      max-width: 420px;
      padding: 12px 14px;
      border-radius: 14px;
      border: 1px solid var(--line-strong);
      background: color-mix(in srgb, var(--panel) 94%, black);
      box-shadow: var(--shadow-xl);
      color: var(--text);
      font-size: 13px;
      line-height: 1.45;
      opacity: 0;
      transform: translateY(-6px);
      transition: opacity 180ms ease, transform 180ms ease;
    }
    .toast.show {
      opacity: 1;
      transform: translateY(0);
    }
    .toast.success { border-color: rgba(52, 199, 89, 0.40); }
    .toast.error { border-color: rgba(255, 107, 107, 0.45); }
    .console-card textarea {
      min-height: 276px;
      resize: vertical;
    }
    .workflow-shell {
      display: grid;
      gap: 18px;
    }
    .workflow-board {
      display: grid;
      gap: 16px;
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }
    .workflow-step {
      border-radius: var(--radius-lg);
      padding: 18px;
      display: grid;
      gap: 14px;
      position: relative;
      overflow: hidden;
      transition: transform var(--trans-fast), box-shadow var(--trans-fast), border-color var(--trans-fast);
    }
    .workflow-step:hover {
      transform: translateY(-2px);
      box-shadow: var(--shadow-xl);
    }
    .workflow-step::before {
      content: "";
      position: absolute;
      inset: 0;
      background: linear-gradient(180deg, rgba(255,255,255,0.03), transparent 45%);
      pointer-events: none;
    }
    .workflow-step.map { background: linear-gradient(180deg, var(--panel), var(--map-tint)); }
    .workflow-step.loc { background: linear-gradient(180deg, var(--panel), var(--loc-tint)); }
    .workflow-step.record { background: linear-gradient(180deg, var(--panel), var(--record-tint)); }
    .workflow-step.drive { background: linear-gradient(180deg, var(--panel), var(--drive-tint)); }
    .step-number {
      width: 34px;
      height: 34px;
      border-radius: 999px;
      display: inline-grid;
      place-items: center;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.06);
      font-size: 13px;
      font-weight: 700;
      color: var(--text);
    }
    .task-tag {
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      padding: 0 10px;
      border-radius: 999px;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.10em;
      font-weight: 700;
      border: 1px solid transparent;
    }
    .task-tag.map { color: var(--accent); background: var(--map-tint); border-color: color-mix(in srgb, var(--accent) 22%, transparent); }
    .task-tag.loc { color: var(--accent-2); background: var(--loc-tint); border-color: color-mix(in srgb, var(--accent-2) 22%, transparent); }
    .task-tag.record { color: #c98a24; background: var(--record-tint); border-color: rgba(201, 138, 36, 0.25); }
    .task-tag.drive { color: var(--accent-3); background: var(--drive-tint); border-color: color-mix(in srgb, var(--accent-3) 24%, transparent); }
    .workflow-copy {
      display: grid;
      gap: 6px;
      min-height: 78px;
      align-content: start;
    }
    .workflow-copy p {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.55;
    }
    .step-dependency {
      padding: 11px 12px;
      border-radius: var(--radius-md);
      border: 1px dashed var(--line-strong);
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
      background: rgba(255,255,255,0.03);
    }
    .step-dependency strong { color: var(--text); }
    .fields {
      display: grid;
      gap: 12px;
    }
    .field {
      display: grid;
      gap: 7px;
    }
    .field-inline {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    label {
      display: block;
      font-size: 12px;
      color: var(--muted);
      font-weight: 700;
      letter-spacing: 0.01em;
    }
    input, select, button, textarea {
      width: 100%;
      min-height: 44px;
      border-radius: var(--radius-sm);
      border: 1px solid var(--line-strong);
      background: var(--input-bg);
      color: var(--text);
      padding: 10px 13px;
      font: inherit;
      transition: border-color var(--trans-fast), box-shadow var(--trans-fast), transform var(--trans-fast), background var(--trans-fast), opacity var(--trans-fast);
    }
    input, select, textarea { font-size: 14px; }
    input:focus, select:focus, textarea:focus {
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 4px var(--ring);
    }
    button {
      cursor: pointer;
      font-size: 14px;
      font-weight: 700;
      border-color: transparent;
      color: white;
      background: linear-gradient(180deg, color-mix(in srgb, var(--accent) 96%, white), color-mix(in srgb, var(--accent) 82%, black));
      box-shadow: 0 10px 20px rgba(70, 111, 255, 0.18);
    }
    button.secondary {
      background: transparent;
      color: var(--text);
      border-color: var(--line-strong);
      box-shadow: none;
    }
    button.warn {
      background: linear-gradient(180deg, color-mix(in srgb, var(--warn) 94%, white), color-mix(in srgb, var(--warn) 82%, black));
    }
    button:disabled {
      opacity: 0.52;
      cursor: not-allowed;
      box-shadow: none;
      transform: none;
    }
    button:not(:disabled):hover {
      transform: translateY(-1px);
      box-shadow: 0 14px 28px rgba(70, 111, 255, 0.22);
    }
    button.secondary:not(:disabled):hover {
      border-color: var(--accent);
      box-shadow: none;
    }
    textarea {
      font-family: "Cascadia Mono", "SFMono-Regular", Consolas, monospace;
      line-height: 1.5;
      font-size: 12px;
      background: color-mix(in srgb, var(--input-bg) 92%, black);
    }
    .toggle-row {
      display: flex;
      gap: 12px;
      align-items: center;
      padding: 12px;
      border-radius: var(--radius-sm);
      border: 1px solid var(--line-strong);
      background: var(--input-bg);
    }
    .toggle-row input {
      width: 18px;
      min-height: 18px;
      padding: 0;
      margin: 0;
      accent-color: var(--accent);
    }
    .toggle-row span {
      font-size: 13px;
      color: var(--text);
    }
    .button-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .notes-grid {
      display: grid;
      gap: 12px;
    }
    .note {
      padding: 14px 15px;
      border-radius: var(--radius-md);
      border: 1px solid var(--line);
      background: var(--panel-2);
      color: var(--muted);
      font-size: 12px;
      line-height: 1.65;
    }
    .note strong, .note code { color: var(--text); }
    .properties {
      display: grid;
      grid-template-columns: 124px 1fr;
      gap: 10px 12px;
      font-size: 13px;
    }
    .properties > div:nth-child(odd) { color: var(--muted); }
    .status-line {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
      padding: 12px 14px;
      border-radius: var(--radius-md);
      border: 1px solid var(--line);
      background: var(--panel-2);
    }
    .status-line strong { font-size: 13px; }
    .status-badge {
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      padding: 0 10px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.04);
      color: var(--muted);
    }
    .status-badge.ready { color: var(--good); border-color: rgba(41, 197, 117, 0.34); background: rgba(41, 197, 117, 0.10); }
    .status-badge.wait { color: var(--accent); border-color: rgba(94, 139, 255, 0.30); background: rgba(94, 139, 255, 0.10); }
    .status-badge.warn { color: var(--warn); border-color: rgba(226, 107, 119, 0.30); background: rgba(226, 107, 119, 0.10); }
    .workflow-arrow {
      display: flex;
      justify-content: center;
      align-items: center;
      color: var(--muted);
      font-size: 18px;
      margin: -4px 0;
    }
    .deferred-stage {
      display: none;
      gap: 18px;
      opacity: 0;
      transform: translateY(12px);
    }
    .deferred-stage.visible {
      display: grid;
      opacity: 1;
      transform: translateY(0);
      animation: panelIn 280ms ease;
    }
    .deferred-intro {
      display: flex;
      justify-content: space-between;
      gap: 14px;
      align-items: center;
      padding: 14px 18px;
      border-radius: var(--radius-lg);
      border: 1px solid var(--line);
      background: var(--panel);
      box-shadow: var(--shadow-md);
    }
    .locked-stage {
      display: grid;
      gap: 12px;
      padding: 18px;
      border-radius: var(--radius-lg);
      border: 1px dashed var(--line-strong);
      background: color-mix(in srgb, var(--panel) 88%, transparent);
    }
    .locked-stage p {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
    }
    .settings-layout {
      display: grid;
      gap: 18px;
      grid-template-columns: minmax(0, 1.16fr) minmax(320px, 0.84fr);
      align-items: start;
    }
    .settings-group {
      display: grid;
      gap: 16px;
    }
    .tuning-layout {
      grid-template-columns: minmax(0, 1fr);
    }
    .tuning-main-stack {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 16px;
      align-content: start;
      width: 100%;
    }
    .tuning-main-stack > .panel {
      height: 100%;
    }
    .tuning-notes {
      grid-column: 1 / -1;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      align-items: stretch;
    }
    .tuning-notes .note {
      height: 100%;
    }
    .subpanel {
      padding: 16px;
      border-radius: var(--radius-lg);
      border: 1px solid var(--line);
      background: var(--panel-2);
      display: grid;
      gap: 14px;
    }
    .subpanel-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
    }
    .model-path {
      padding: 13px 14px;
      border-radius: var(--radius-md);
      border: 1px solid var(--line);
      background: var(--panel-3);
      font-family: "Cascadia Mono", "SFMono-Regular", Consolas, monospace;
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    .muted { color: var(--muted); }
    code {
      font-family: "Cascadia Mono", "SFMono-Regular", Consolas, monospace;
      color: color-mix(in srgb, var(--accent) 72%, white);
    }
    @media (max-width: 1240px) {
      .shell, .dashboard-layout, .settings-layout { grid-template-columns: 1fr; }
      .hero-stats, .workflow-board { grid-template-columns: 1fr 1fr; }
      .tuning-main-stack { grid-template-columns: 1fr; }
      .tuning-notes { grid-template-columns: 1fr; }
    }
    @media (max-width: 860px) {
      .page { padding: 14px; }
      .hero { padding: 18px; }
      .sidebar { padding: 16px; }
      .hero-stats, .workflow-board, .mini-grid, .button-row, .field-inline { grid-template-columns: 1fr; }
      .segmented { width: 100%; }
      .segmented button { flex: 1 1 0; min-width: 0; }
    }

    /* 2026-06 UI polish overrides START */
    :root {
      --ui-bg: #090b10;
      --ui-bg-2: #0d1118;
      --ui-panel: #11151d;
      --ui-panel-2: #151b25;
      --ui-panel-3: #1b2431;
      --ui-strong-surface: #0b0f16;
      --ui-strong-surface-2: #0a0d14;
      --ui-line: rgba(255,255,255,0.08);
      --ui-line-soft: rgba(255,255,255,0.05);
      --ui-text: #f4f7fb;
      --ui-muted: #93a0b4;
      --ui-accent: #6d7dff;
      --ui-accent-soft: rgba(109,125,255,0.16);
      --ui-success: #29c06f;
      --ui-warn: #ef6b7b;
      --ui-shadow: 0 12px 32px rgba(0, 0, 0, 0.34);
      --ui-overlay-sheet: rgba(17, 21, 29, 0.72);
      --ui-overlay-dim: rgba(0, 0, 0, 0.20);
      --ui-overlay-shadow: rgba(0, 0, 0, 0.34);
      --ui-radius: 16px;
      --ui-radius-sm: 12px;
      --ui-radius-xs: 10px;
    }
    body[data-theme="light"] {
      --ui-bg: #eef3fb;
      --ui-bg-2: #f6f9fd;
      --ui-panel: #ffffff;
      --ui-panel-2: #f7f9fd;
      --ui-panel-3: #edf2ff;
      --ui-strong-surface: #f4f7fc;
      --ui-strong-surface-2: #edf2f8;
      --ui-line: rgba(28, 45, 78, 0.12);
      --ui-line-soft: rgba(28, 45, 78, 0.08);
      --ui-text: #172033;
      --ui-muted: #5e6b80;
      --ui-accent: #5a6bff;
      --ui-accent-soft: rgba(90,107,255,0.12);
      --ui-success: #1fa15c;
      --ui-warn: #d85d6f;
      --ui-shadow: 0 10px 28px rgba(24, 42, 70, 0.08);
      --ui-overlay-sheet: rgba(255, 255, 255, 0.72);
      --ui-overlay-dim: rgba(48, 63, 88, 0.10);
      --ui-overlay-shadow: rgba(24, 42, 70, 0.18);
    }
    body {
      background:
        radial-gradient(circle at top left, rgba(109,125,255,0.14), transparent 24%),
        radial-gradient(circle at top right, rgba(55,95,190,0.10), transparent 22%),
        linear-gradient(180deg, var(--ui-bg) 0%, var(--ui-bg-2) 100%);
      color: var(--ui-text);
      font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
      font-feature-settings: "cv01", "ss03";
      transition: background 180ms ease, color 180ms ease;
    }
    body::before { display: none; }
    .page.control-root {
      width: min(1720px, 100%);
      padding: 10px 12px 12px;
      margin: 0 auto;
      display: grid;
      gap: 10px;
    }
    .topbar {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
      align-items: stretch;
      padding: 14px 16px;
      border: 1px solid var(--ui-line);
      border-radius: 18px;
      background: linear-gradient(180deg, color-mix(in srgb, var(--ui-panel) 92%, transparent), color-mix(in srgb, var(--ui-panel-2) 96%, transparent));
      box-shadow: var(--ui-shadow);
    }
    .brand-block { display: grid; gap: 10px; min-width: 0; }
    .brand-kicker {
      font-size: 11px;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: color-mix(in srgb, var(--ui-accent) 78%, white 22%);
      font-weight: 700;
    }
    .brand-row {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
    }
    h1 {
      margin: 0;
      font-size: clamp(26px, 2.2vw, 36px);
      line-height: 1;
      font-weight: 650;
      letter-spacing: -0.04em;
      color: var(--ui-text);
    }
    h2 {
      margin: 0;
      font-size: 15px;
      line-height: 1.15;
      font-weight: 620;
      letter-spacing: -0.02em;
      color: var(--ui-text);
    }
    .top-status-cluster,
    .topbar-actions {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    .status-pill, .pill {
      min-height: 34px;
      padding: 0 12px;
      border-radius: 999px;
      border: 1px solid var(--ui-line);
      background: color-mix(in srgb, var(--ui-panel-2) 88%, transparent);
      color: var(--ui-text);
      font-size: 12px;
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }
    .status-pill.compact strong { font-size: 12px; }
    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--ui-success);
      box-shadow: 0 0 0 4px color-mix(in srgb, var(--ui-success) 18%, transparent);
      flex: 0 0 auto;
    }
    .summary-strip {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
    }
    .summary-cell {
      min-width: 0;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid var(--ui-line-soft);
      background: color-mix(in srgb, var(--ui-panel-2) 84%, transparent);
      display: grid;
      gap: 5px;
    }
    .summary-cell.stretch { grid-column: span 1; }
    .summary-label {
      font-size: 10px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--ui-muted);
    }
    .summary-cell strong {
      font-size: 13px;
      line-height: 1.35;
      font-weight: 620;
      color: var(--ui-text);
      word-break: break-word;
    }
    .tab-dock { display: flex; justify-content: flex-start; }
    .segmented.nav-tabs {
      width: auto;
      display: inline-flex;
      gap: 4px;
      padding: 4px;
      border-radius: 14px;
      background: color-mix(in srgb, var(--ui-panel-2) 86%, transparent);
      border: 1px solid var(--ui-line-soft);
      box-shadow: none;
    }
    .segmented.nav-tabs button {
      min-height: 34px;
      padding: 0 14px;
      font-size: 12px;
      font-weight: 620;
      border-radius: 10px;
      box-shadow: none;
      background: transparent;
      color: var(--ui-muted);
      border: 0;
    }
    .segmented.nav-tabs button.active {
      background: var(--ui-panel-3);
      color: var(--ui-text);
      box-shadow: inset 0 0 0 1px var(--ui-line-soft);
      transform: none;
    }
    .control-grid {
      display: grid;
      grid-template-columns: 310px minmax(0, 1fr);
      gap: 10px;
      align-items: start;
    }
    .ops-rail,
    .workspace {
      min-width: 0;
      display: grid;
      gap: 10px;
    }
    .rail-panel,
    .panel,
    .subpanel,
    .workflow-step,
    .deferred-intro,
    .locked-stage,
    .note,
    .route-preview-wrap {
      background: linear-gradient(180deg, color-mix(in srgb, var(--ui-panel) 96%, transparent), color-mix(in srgb, var(--ui-panel-2) 96%, transparent));
      border: 1px solid var(--ui-line);
      border-radius: var(--ui-radius);
      box-shadow: var(--ui-shadow);
    }
    .panel,
    .rail-panel,
    .subpanel,
    .workflow-step,
    .deferred-intro,
    .locked-stage {
      padding: 14px;
    }
    .rail-title-row,
    .panel-head.tight {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 10px;
      margin-bottom: 10px;
    }
    .panel-head.tight { margin-bottom: 10px; }
    .panel-sub {
      margin-top: 4px;
      color: var(--ui-muted);
      font-size: 11px;
      line-height: 1.45;
      max-width: none;
    }
    .rail-metrics,
    .gate-stack,
    .stack,
    .settings-group,
    .workflow-shell,
    .fields,
    .field,
    .button-row,
    .field-inline { display: grid; gap: 8px; }
    .rail-metric {
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid var(--ui-line-soft);
      background: color-mix(in srgb, var(--ui-panel-2) 88%, transparent);
      display: grid;
      gap: 4px;
    }
    .rail-metric span {
      font-size: 10px;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--ui-muted);
    }
    .rail-metric strong {
      font-size: 12px;
      line-height: 1.35;
      color: var(--ui-text);
      word-break: break-word;
    }
    .status-line.compact {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid var(--ui-line-soft);
      background: color-mix(in srgb, var(--ui-panel-2) 88%, transparent);
    }
    .status-badge {
      min-height: 26px;
      padding: 0 9px;
      border-radius: 999px;
      border: 1px solid var(--ui-line);
      background: color-mix(in srgb, var(--ui-panel-3) 86%, transparent);
      color: var(--ui-text);
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      display: inline-flex;
      align-items: center;
      white-space: nowrap;
    }
    .status-badge.warn { color: color-mix(in srgb, var(--ui-warn) 75%, var(--ui-text)); border-color: color-mix(in srgb, var(--ui-warn) 35%, transparent); background: color-mix(in srgb, var(--ui-warn) 12%, transparent); }
    .status-badge.ready { color: color-mix(in srgb, var(--ui-success) 72%, var(--ui-text)); border-color: color-mix(in srgb, var(--ui-success) 35%, transparent); background: color-mix(in srgb, var(--ui-success) 12%, transparent); }
    .overview-grid {
      display: grid;
      grid-template-columns: minmax(0, 1.42fr) minmax(360px, 0.9fr);
      gap: 10px;
      align-items: start;
    }
    .preview-panel { grid-row: span 2; }
    .console-panel textarea {
      width: 100%;
      min-height: 260px;
      resize: vertical;
      border-radius: 12px;
      font-size: 11px;
      background: var(--ui-strong-surface);
      border: 1px solid var(--ui-line);
      color: var(--ui-text);
      padding: 10px 12px;
    }
    .preview-stage {
      overflow: hidden;
      border-radius: 14px;
      border: 1px solid var(--ui-line);
      background: var(--ui-strong-surface);
      box-shadow: inset 0 1px 0 color-mix(in srgb, var(--ui-text) 4%, transparent);
    }
    .preview {
      width: 100%;
      aspect-ratio: 16 / 9;
      object-fit: contain;
      display: block;
      background: linear-gradient(180deg, color-mix(in srgb, var(--ui-text) 3%, transparent), transparent), var(--ui-strong-surface-2);
    }
    .preview-footer {
      padding: 10px 12px;
      display: flex;
      justify-content: space-between;
      gap: 10px;
      flex-wrap: wrap;
      color: var(--ui-muted);
      font-size: 11px;
      border-top: 1px solid var(--ui-line);
    }
    .preview-footer strong { color: var(--ui-text); }
    .lidar-example {
      margin-top: 18px;
      padding-top: 16px;
      border-top: 1px solid var(--ui-line-soft);
    }
    .lidar-example-head {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 12px;
    }
    .lidar-example-title {
      display: flex;
      align-items: center;
      gap: 9px;
      margin-bottom: 4px;
    }
    .lidar-example-title h2 { margin: 0; }
    .lidar-sample-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      min-height: 24px;
      padding: 4px 9px;
      border: 1px solid color-mix(in srgb, var(--ui-warn) 34%, var(--ui-line));
      border-radius: 999px;
      background: color-mix(in srgb, var(--ui-warn) 10%, transparent);
      color: color-mix(in srgb, var(--ui-warn) 78%, var(--ui-text));
      font-size: 9.5px;
      font-weight: 760;
      letter-spacing: .07em;
      text-transform: uppercase;
      white-space: nowrap;
    }
    .lidar-sample-badge::before {
      content: "";
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--ui-warn);
      box-shadow: 0 0 0 4px color-mix(in srgb, var(--ui-warn) 13%, transparent);
    }
    .lidar-sample-badge.live {
      border-color: color-mix(in srgb, var(--ui-success) 38%, var(--ui-line));
      background: color-mix(in srgb, var(--ui-success) 11%, transparent);
      color: color-mix(in srgb, var(--ui-success) 75%, var(--ui-text));
    }
    .lidar-sample-badge.live::before {
      background: var(--ui-success);
      box-shadow: 0 0 0 4px color-mix(in srgb, var(--ui-success) 13%, transparent);
      animation: lidarLivePulse 1.8s ease-in-out infinite;
    }
    @keyframes lidarLivePulse {
      50% { transform: scale(1.35); opacity: .7; }
    }
    .lidar-geometry {
      display: flex;
      justify-content: flex-end;
      gap: 6px;
      flex-wrap: wrap;
    }
    .lidar-chip {
      padding: 6px 9px;
      border: 1px solid var(--ui-line-soft);
      border-radius: 8px;
      background: var(--ui-strong-surface);
      color: var(--ui-muted);
      font-size: 11px;
      line-height: 1.1;
      white-space: nowrap;
    }
    .lidar-chip strong { color: var(--ui-text); font-weight: 740; }
    .lidar-canvas-wrap {
      position: relative;
      min-height: 300px;
      overflow: hidden;
      border: 1px solid var(--ui-line);
      border-radius: 12px;
      background: var(--ui-strong-surface-2);
      box-shadow: inset 0 1px 0 color-mix(in srgb, var(--ui-text) 4%, transparent);
    }
    #lidarExampleCanvas {
      display: block;
      width: 100%;
      height: 320px;
    }
    .lidar-axis-note {
      position: absolute;
      top: 10px;
      left: 12px;
      pointer-events: none;
      color: var(--ui-muted);
      font-size: 9.5px;
      letter-spacing: .02em;
    }
    .lidar-example-foot {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px 16px;
      flex-wrap: wrap;
      padding: 10px 2px 0;
      color: var(--ui-muted);
      font-size: 10px;
    }
    .lidar-legend {
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }
    .lidar-legend span {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      white-space: nowrap;
    }
    .lidar-key {
      width: 12px;
      height: 3px;
      border-radius: 999px;
      background: #64748b;
    }
    .lidar-key.left { background: #16a34a; }
    .lidar-key.right { background: #ea580c; }
    .lidar-key.center { background: #e11d48; }
    .lidar-example-note strong { color: var(--ui-text); }
    .properties {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px 12px;
    }
    .properties > div {
      padding: 8px 10px;
      border-radius: 10px;
      border: 1px solid var(--ui-line-soft);
      background: color-mix(in srgb, var(--ui-panel-2) 88%, transparent);
      font-size: 11px;
      color: var(--ui-text);
    }
    .fresh-workflow,
    .deferred-stage { display: grid; gap: 10px; }
    .workflow-board {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .workflow-step {
      display: grid;
      gap: 10px;
      align-content: start;
    }
    .step-number {
      width: 28px;
      height: 28px;
      border-radius: 50%;
      display: grid;
      place-items: center;
      background: var(--ui-accent-soft);
      border: 1px solid color-mix(in srgb, var(--ui-accent) 32%, transparent);
      color: color-mix(in srgb, var(--ui-accent) 70%, var(--ui-text));
      font-size: 11px;
      font-weight: 700;
    }
    .task-tag {
      width: fit-content;
      min-height: 24px;
      padding: 0 8px;
      border-radius: 999px;
      border: 1px solid var(--ui-line);
      background: color-mix(in srgb, var(--ui-panel-3) 86%, transparent);
      color: var(--ui-text);
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      display: inline-flex;
      align-items: center;
    }
    .workflow-copy { display: grid; gap: 4px; min-height: 0; }
    .workflow-copy p,
    .locked-stage p { margin: 0; color: var(--ui-muted); font-size: 11px; line-height: 1.45; }
    .step-dependency,
    .note {
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid var(--ui-line-soft);
      background: color-mix(in srgb, var(--ui-panel-2) 88%, transparent);
      font-size: 11px;
      line-height: 1.45;
      color: var(--ui-muted);
    }
    .deferred-intro,
    .compact-locked {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    .settings-layout {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      align-items: start;
    }
    .subpanel { display: grid; gap: 10px; }
    label {
      font-size: 11px;
      font-weight: 600;
      color: var(--ui-text);
    }
    input, select, textarea {
      min-height: 38px;
      width: 100%;
      padding: 8px 10px;
      border-radius: 10px;
      border: 1px solid var(--ui-line);
      background: var(--ui-strong-surface);
      color: var(--ui-text);
      font-size: 12px;
    }
    textarea { min-height: 110px; }
    input:focus, select:focus, textarea:focus {
      outline: none;
      border-color: color-mix(in srgb, var(--ui-accent) 55%, transparent);
      box-shadow: 0 0 0 3px color-mix(in srgb, var(--ui-accent) 14%, transparent);
    }
    .toggle-row {
      display: flex;
      gap: 8px;
      align-items: center;
      font-size: 11px;
      color: var(--ui-muted);
    }
    .toggle-row input { width: auto; min-height: auto; }
    .field-inline {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .button-row {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    button {
      min-height: 38px;
      padding: 0 14px;
      border-radius: 10px;
      border: 1px solid color-mix(in srgb, var(--ui-accent) 32%, transparent);
      background: linear-gradient(180deg, color-mix(in srgb, var(--ui-accent) 88%, white 12%), color-mix(in srgb, var(--ui-accent) 84%, black 16%));
      color: white;
      font-size: 12px;
      font-weight: 650;
      box-shadow: 0 8px 20px color-mix(in srgb, var(--ui-accent) 26%, transparent);
    }
    button.secondary {
      background: color-mix(in srgb, var(--ui-panel-2) 88%, transparent);
      color: var(--ui-text);
      border-color: var(--ui-line);
      box-shadow: none;
    }
    button.warn {
      background: linear-gradient(180deg, color-mix(in srgb, var(--ui-warn) 88%, white 12%), color-mix(in srgb, var(--ui-warn) 84%, black 16%));
      border-color: color-mix(in srgb, var(--ui-warn) 32%, transparent);
    }
    button.slim { min-height: 34px; padding: 0 12px; }
    button:not(:disabled):hover { transform: translateY(-1px); }
    .route-preview-wrap {
      margin-top: 8px;
      padding: 10px;
      background: var(--ui-strong-surface);
    }
    .route-preview-wrap canvas {
      width: 100%;
      height: 240px;
      display: block;
      border-radius: 10px;
      background: var(--ui-strong-surface-2);
    }
    .notes-grid {
      display: grid;
      gap: 10px;
    }
    .toast {
      font-size: 11.5px;
      border-radius: 12px;
      border-color: var(--ui-line);
      background: var(--ui-panel);
      color: var(--ui-text);
    }
    .tab-panel { animation: none; }
    .panel:hover,
    .workflow-step:hover,
    .rail-panel:hover,
    .subpanel:hover,
    .deferred-intro:hover,
    .locked-stage:hover { transform: none; }
    @media (max-width: 1360px) {
      .control-grid,
      .overview-grid,
      .settings-layout { grid-template-columns: 1fr; }
      .preview-panel { grid-row: auto; }
      .ops-rail { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 920px) {
      .summary-strip,
      .workflow-board,
      .field-inline,
      .properties,
      .ops-rail { grid-template-columns: 1fr; }
      .topbar { grid-template-columns: 1fr; }
      .topbar-actions { justify-content: flex-start; }
      .settings-layout { grid-template-columns: 1fr; }
    }
    @media (max-width: 640px) {
      .page.control-root { padding: 8px; }
      .topbar,
      .panel,
      .rail-panel,
      .workflow-step,
      .subpanel,
      .deferred-intro,
      .locked-stage { padding: 12px; }
      .brand-row,
      .deferred-intro,
      .compact-locked,
      .status-line.compact { grid-template-columns: 1fr; display: grid; }
      .segmented.nav-tabs { width: 100%; display: grid; grid-template-columns: 1fr 1fr; }
      .segmented.nav-tabs button { width: 100%; }
      .button-row { display: grid; }
    }
    /* 2026-06 UI polish overrides END */

    /* v7 layout refresh — preserves the established blue / violet palette */
    .page.control-root {
      width: min(1840px, 100%);
      min-height: 100vh;
      padding: 18px clamp(16px, 2vw, 34px) 30px;
      gap: 14px;
    }
    .topbar {
      position: relative;
      top: auto;
      z-index: auto;
      grid-template-columns: minmax(0, 1fr) auto;
      align-items: center;
      padding: 16px 18px;
      border-radius: 20px;
      background: color-mix(in srgb, var(--ui-panel) 88%, transparent);
      backdrop-filter: blur(20px);
    }
    .brand-block { gap: 12px; }
    .brand-row { align-items: flex-end; }
    .summary-strip {
      grid-template-columns: 116px 150px minmax(180px, 1fr) minmax(180px, 1fr);
      gap: 0;
      border-top: 1px solid var(--ui-line-soft);
      border-bottom: 1px solid var(--ui-line-soft);
    }
    .summary-cell {
      padding: 9px 14px;
      border: 0;
      border-radius: 0;
      background: transparent;
      border-right: 1px solid var(--ui-line-soft);
    }
    .summary-cell:last-child { border-right: 0; }
    .summary-cell strong { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .topbar-actions { align-self: end; padding-bottom: 0; }
    .tab-dock {
      position: relative;
      top: auto;
      z-index: auto;
      padding: 4px 0;
      pointer-events: auto;
    }
    .segmented.nav-tabs { box-shadow: var(--ui-shadow); }
    .segmented.nav-tabs button { min-width: 94px; }
    .control-grid {
      grid-template-columns: 276px minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }
    .ops-rail {
      position: sticky;
      top: 158px;
      gap: 12px;
    }
    .workspace { gap: 18px; }
    .rail-panel, .panel, .workflow-step, .subpanel, .deferred-intro, .locked-stage {
      box-shadow: none;
    }
    .rail-panel {
      padding: 15px;
      border-radius: 14px;
      background: color-mix(in srgb, var(--ui-panel) 92%, transparent);
    }
    .rail-metric {
      padding: 11px 0;
      border: 0;
      border-radius: 0;
      border-bottom: 1px solid var(--ui-line-soft);
      background: transparent;
    }
    .rail-metric:last-child { border-bottom: 0; padding-bottom: 0; }
    .status-line.compact { padding: 10px 0; border: 0; border-bottom: 1px solid var(--ui-line-soft); border-radius: 0; background: transparent; }
    .status-line.compact:last-child { border-bottom: 0; padding-bottom: 0; }
    .overview-grid {
      grid-template-columns: minmax(0, 1.56fr) minmax(310px, .74fr);
      gap: 18px;
    }
    .preview-panel {
      grid-row: span 2;
      padding: 16px;
      border-radius: 18px;
    }
    .preview-stage { border-radius: 12px; }
    .preview { aspect-ratio: 16 / 8.5; }
    .status-panel, .console-panel { border-radius: 14px; }
    .properties { grid-template-columns: 104px 1fr; gap: 0; border-top: 1px solid var(--ui-line-soft); }
    .properties > div { padding: 10px 0; border: 0; border-bottom: 1px solid var(--ui-line-soft); border-radius: 0; background: transparent; }
    .properties > div:nth-child(even) { font-weight: 620; }
    .console-panel textarea { min-height: 224px; }
    .workflow-shell { gap: 16px; }
    .workflow-board.primary-workflow { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 14px; }
    .workflow-board.secondary-workflow { grid-template-columns: minmax(0, .82fr) minmax(0, 1.18fr); gap: 14px; }
    .workflow-step {
      min-height: 270px;
      padding: 18px;
      border-radius: 16px;
    }
    .workflow-step.drive { min-height: 0; }
    .step-number { margin-bottom: 4px; }
    .workflow-copy { min-height: 64px; }
    .workflow-copy h2 { font-size: 18px; }
    .step-dependency { margin-top: auto; }
    .locked-stage { border-style: solid; border-radius: 16px; }
    .settings-layout { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 18px; }
    .tuning-main-stack { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 18px; }
    .settings-layout.tuning-layout { grid-template-columns: minmax(0, 1fr); }
    .tuning-main-stack > .panel { padding: 18px; }
    .subpanel { padding:14px; background: var(--ui-strong-surface); }
    .note { box-shadow: none; }
    button { letter-spacing: .01em; }
    .status-dot { animation: uiStatusPulse 2.4s ease-in-out infinite; }
    @keyframes uiStatusPulse { 50% { transform: scale(1.18); opacity: .72; } }
    .tab-panel.active { animation: workspaceEnter 280ms cubic-bezier(.2,.7,.2,1); }
    @keyframes workspaceEnter { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
    .workflow-step, .panel { transition: border-color 180ms ease, transform 180ms ease; }
    .workflow-step:hover, .panel:hover { transform: translateY(-2px); border-color: var(--ui-line); }
    @media (max-width: 1360px) {
      .topbar, .control-grid, .overview-grid, .settings-layout { grid-template-columns: 1fr; }
      .topbar { position: relative; top: auto; }
      .tab-dock, .ops-rail { position: relative; top: auto; }
      .ops-rail { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .preview-panel { grid-row: auto; }
    }
    @media (max-width: 820px) {
      .page.control-root { padding: 10px; }
      .summary-strip, .workflow-board.primary-workflow, .workflow-board.secondary-workflow, .tuning-main-stack { grid-template-columns: 1fr; }
      .summary-cell { border-right: 0; border-bottom: 1px solid var(--ui-line-soft); }
      .summary-cell:last-child { border-bottom: 0; }
      .ops-rail { grid-template-columns: 1fr; }
      .topbar-actions { justify-content: flex-start; }
      .lidar-example-head { flex-direction: column; }
      .lidar-geometry { justify-content: flex-start; }
      #lidarExampleCanvas { height: 270px; }
      .lidar-canvas-wrap { min-height: 250px; }
    }

    /* visual QA: prevent dense status labels from competing for the same line */
    h2 { font-size: 16px; line-height: 1.25; }
    .brand-kicker, .summary-label, .rail-metric span { font-size: 10.5px; }
    .panel-sub, .workflow-copy p, .locked-stage p { font-size: 12px; line-height: 1.5; }
    .rail-title-row, .panel-head.tight, .brand-row {
      flex-wrap: wrap;
      align-items: center;
    }
    .rail-title-row h2, .panel-head.tight > div:first-child { min-width: 0; }
    .status-badge {
      max-width: 100%;
      min-height: 28px;
      padding: 5px 10px;
      line-height: 1.15;
      text-align: center;
      white-space: normal;
      overflow-wrap: anywhere;
    }
    .status-line.compact { align-items: flex-start; }
    .status-line.compact > div:first-child { min-width: 0; }
    .status-line.compact .status-badge { flex: 0 1 auto; }
    .workflow-copy h2 { font-size: 19px; line-height: 1.15; }
    .workflow-copy p { max-width: 34ch; }
    .field label, label { font-size: 12px; }
    input, select, textarea { font-size: 13px; }
    .summary-cell { min-width: 0; }
    @media (max-width: 820px) {
      h1 { font-size: clamp(28px, 8vw, 34px); }
      .top-status-cluster { width: 100%; justify-content: flex-start; }
      .rail-title-row, .panel-head.tight { align-items: flex-start; }
      .rail-title-row .status-badge { width: fit-content; }
      .status-line.compact { display: grid; grid-template-columns: 1fr; gap: 7px; }
      .status-line.compact .status-badge { width: fit-content; max-width: 100%; }
      .summary-cell strong { white-space: normal; overflow: visible; text-overflow: clip; }
    }

    /* Primary driving display: camera first, lidar and instruments alongside it. */
    .page.control-root {
      width: 100%;
      padding: 10px 14px 12px;
      gap: 10px;
    }
    .topbar {
      padding: 10px 16px;
    }
    .brand-row h1 {
      font-size: clamp(22px, 1.8vw, 29px);
    }
    .control-grid {
      grid-template-columns: minmax(0, 1fr);
    }
    .workspace {
      width: 100%;
    }
    .dashboard-metrics {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      grid-template-rows: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin: 0;
      min-height: 0;
    }
    .run-metric {
      position: relative;
      min-width: 0;
      min-height: 0;
      height: 100%;
      padding: 8px 12px;
      overflow: hidden;
      border: 1px solid var(--ui-line);
      border-radius: 13px;
      background: linear-gradient(180deg, color-mix(in srgb, var(--ui-panel) 97%, transparent), color-mix(in srgb, var(--ui-panel-2) 94%, transparent));
      display: grid;
      align-content: center;
    }
    .run-metric::after {
      content: none;
    }
    .run-metric span {
      display: block;
      color: var(--ui-muted);
      font-size: 9px;
      font-weight: 650;
      letter-spacing: .08em;
      text-transform: uppercase;
    }
    .run-metric strong {
      display: block;
      margin-top: 4px;
      color: var(--ui-text);
      font-size: 13px;
      line-height: 1.1;
      font-weight: 760;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .run-metric small {
      margin-left: 4px;
      color: var(--ui-muted);
      font-size: 8.5px;
      font-weight: 600;
    }
    .overview-grid {
      grid-template-columns: minmax(0, 1.12fr) minmax(450px, 1fr);
      grid-template-rows: minmax(360px, 1.38fr) minmax(190px, .62fr);
      grid-template-areas:
        "camera lidar"
        "camera status";
      gap: 10px;
      height: calc(100vh - 92px);
      min-height: 680px;
      align-items: stretch;
    }
    .preview-panel {
      grid-area: camera;
      padding: 10px;
      border-radius: 16px;
      min-height: 0;
      display: flex;
      flex-direction: column;
    }
    .preview-panel .preview-stage {
      min-height: 0;
      flex: 1;
      display: flex;
      flex-direction: column;
    }
    .preview-panel .preview {
      min-height: 0;
      height: 100%;
      flex: 1;
      aspect-ratio: auto;
      object-fit: contain;
    }
    .preview-panel .preview-footer {
      padding: 8px 10px;
      font-size: 10px;
    }
    .preview-panel .panel-head.tight {
      margin-bottom: 7px;
    }
    .preview-panel .panel-head.tight h2,
    .lidar-panel .lidar-example-title h2 {
      font-size: 15px;
    }
    .status-panel {
      grid-area: status;
      position: relative;
      min-height: 0;
      padding: 12px 126px 12px 12px;
      border-radius: 16px;
      display: grid;
      grid-template-rows: minmax(0, 2fr) minmax(0, 1fr);
      gap: 8px;
    }
    #vehicleStatus {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      grid-template-rows: minmax(0, 1fr);
      gap: 8px;
      border-top: 0;
      min-height: 0;
    }
    #vehicleStatus .vehicle-metric {
      position: relative;
      min-width: 0;
      min-height: 0;
      height: 100%;
      padding: 8px 12px;
      overflow: hidden;
      border: 1px solid var(--ui-line);
      border-radius: 13px;
      background: linear-gradient(180deg, color-mix(in srgb, var(--ui-panel) 97%, transparent), color-mix(in srgb, var(--ui-panel-2) 94%, transparent));
      display: grid;
      align-content: center;
      gap: 5px;
    }
    #vehicleStatus .vehicle-metric::after {
      content: none;
    }
    #vehicleStatus .vehicle-metric.charging {
      border-color: var(--ui-line);
      background: linear-gradient(180deg, color-mix(in srgb, var(--ui-panel) 97%, transparent), color-mix(in srgb, var(--ui-panel-2) 94%, transparent));
    }
    #vehicleStatus .vehicle-metric.charging strong {
      color: color-mix(in srgb, var(--ui-success) 76%, var(--ui-text));
    }
    #vehicleStatus .vehicle-metric span {
      color: var(--ui-muted);
      font-size: 9px;
      font-weight: 650;
      letter-spacing: .06em;
      text-transform: uppercase;
    }
    #vehicleStatus .vehicle-metric strong {
      min-width: 0;
      color: var(--ui-text);
      font-size: 13px;
      line-height: 1.1;
      font-weight: 760;
      text-align: left;
      overflow-wrap: anywhere;
    }
    .floating-tools {
      position: absolute;
      top: 12px;
      right: 12px;
      bottom: 12px;
      width: 102px;
      display: grid;
      grid-template-rows: repeat(3, minmax(0, 1fr));
      align-content: stretch;
      gap: 8px;
      padding: 9px;
      border: 1px solid var(--ui-line);
      border-radius: 13px;
      background: linear-gradient(180deg, color-mix(in srgb, var(--ui-panel) 97%, transparent), color-mix(in srgb, var(--ui-panel-2) 94%, transparent));
    }
    .floating-tools button {
      width: 100%;
      min-height: 0;
      height: 100%;
      padding: 0 8px;
      border-radius: 9px;
      font-size: 9px;
      font-weight: 650;
    }
    .lidar-panel {
      grid-area: lidar;
      padding: 10px;
      border-radius: 16px;
      min-height: 0;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }
    .lidar-panel .lidar-example {
      margin-top: 0;
      padding-top: 0;
      border-top: 0;
      min-height: 0;
      height: 100%;
      display: flex;
      flex-direction: column;
    }
    .lidar-panel .lidar-example-head {
      margin-bottom: 7px;
      display: block;
    }
    .lidar-panel .lidar-example-title {
      margin-bottom: 7px;
    }
    .lidar-panel .lidar-geometry {
      width: 100%;
      display: grid;
      grid-template-columns: repeat(6, max-content);
      justify-content: space-between;
      align-items: center;
      gap: 6px;
    }
    .lidar-panel .lidar-example-title h2 {
      white-space: nowrap;
    }
    .lidar-panel .lidar-canvas-wrap {
      min-height: 260px;
      flex: 1;
    }
    .lidar-panel #lidarExampleCanvas {
      height: 100%;
    }
    .lidar-panel .lidar-example-foot {
      padding-top: 8px;
    }
    details.console-panel { display: none; }
    details.console-panel summary {
      padding: 11px 14px;
      color: var(--ui-muted);
      font-size: 11px;
      font-weight: 650;
      cursor: pointer;
      list-style-position: inside;
    }
    details.console-panel[open] summary {
      border-bottom: 1px solid var(--ui-line-soft);
    }
    details.console-panel textarea {
      height: 170px;
      min-height: 170px;
      border: 0;
      border-radius: 0;
      resize: vertical;
    }
    @media (max-width: 1180px) {
      .overview-grid {
        grid-template-columns: 1fr;
        grid-template-rows: auto;
        grid-template-areas:
          "camera"
          "lidar"
          "status";
        height: auto;
        min-height: 0;
      }
      .status-panel {
        min-height: 250px;
      }
    }
    @media (max-width: 640px) {
      .dashboard-metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .run-metric {
        min-height: 72px;
        padding: 11px;
      }
      .preview-panel .preview {
        height: auto;
        aspect-ratio: 16 / 9;
      }
      #vehicleStatus {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
      .lidar-panel #lidarExampleCanvas {
        height: 250px;
      }
      .lidar-panel .lidar-geometry {
        grid-template-columns: repeat(2, max-content);
        justify-content: start;
      }
      .status-panel { padding-right: 12px; padding-bottom: 146px; }
      .floating-tools {
        top: auto;
        left: 12px;
        right: 12px;
        bottom: 12px;
        width: auto;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        grid-template-rows: minmax(0, 1fr);
      }
    }
    #tab-dashboard { display: block; opacity: 1; transform: none; }
    .overlay-panel {
      position: fixed;
      z-index: 120;
      top: 74px;
      left: 50%;
      width: min(1420px, calc(100vw - 40px));
      height: fit-content;
      min-height: 0;
      max-height: calc(100vh - 94px);
      padding: 18px;
      overflow: auto;
      transform: translate(-50%, 12px);
      border: 1px solid var(--ui-line);
      border-radius: 18px;
      background: var(--ui-overlay-sheet);
      box-shadow:
        0 0 0 100vmax var(--ui-overlay-dim),
        0 28px 90px var(--ui-overlay-shadow);
      backdrop-filter: blur(14px) saturate(1.08);
    }
    .overlay-panel.active {
      display: block;
      opacity: 1;
      transform: translate(-50%, 0);
      animation: overlayEnter 180ms ease;
    }
    .overlay-panel::before {
      content: none;
    }
    button.secondary.overlay-close {
      position: sticky;
      z-index: 4;
      top: 0;
      float: right;
      width: auto;
      min-height: 34px;
      padding: 0 13px;
      margin: 0 0 10px 12px;
      border-color: color-mix(in srgb, var(--ui-warn) 42%, var(--ui-line));
      background: color-mix(in srgb, var(--ui-warn) 16%, var(--ui-panel));
      color: color-mix(in srgb, var(--ui-warn) 82%, var(--ui-text));
      box-shadow: none;
    }
    button.secondary.overlay-close:not(:disabled):hover {
      border-color: color-mix(in srgb, var(--ui-warn) 58%, var(--ui-line));
      background: color-mix(in srgb, var(--ui-warn) 23%, var(--ui-panel));
      color: color-mix(in srgb, var(--ui-warn) 90%, var(--ui-text));
      box-shadow: none;
    }
    body.workflow-overlay-open {
      overflow: hidden;
    }
    #tab-tasks.overlay-panel {
      top: 66px;
      width: min(1540px, calc(100vw - 32px));
      height: auto;
      max-height: none;
      overflow: visible;
    }
    #tab-tasks .workflow-shell,
    #tab-tasks .deferred-stage {
      gap: 10px;
    }
    #tab-tasks .workflow-board.primary-workflow,
    #tab-tasks .workflow-board.secondary-workflow {
      gap: 10px;
    }
    #tab-tasks .workflow-step {
      min-height: 0;
      padding: 14px;
      gap: 8px;
    }
    #tab-tasks .workflow-copy {
      min-height: 0;
    }
    #tab-tasks .workflow-copy h2 {
      font-size: 16px;
    }
    #tab-tasks .fields,
    #tab-tasks .field,
    #tab-tasks .field-inline {
      gap: 6px;
    }
    #tab-tasks input,
    #tab-tasks select,
    #tab-tasks button:not(.overlay-close) {
      min-height: 34px;
    }
    #tab-tasks .step-dependency {
      padding: 8px 10px;
      margin-top: 0;
    }
    #tab-tasks .deferred-intro,
    #tab-tasks .locked-stage {
      padding: 10px 14px;
    }
    @media (max-width: 820px), (max-height: 720px) {
      body.workflow-overlay-open { overflow: hidden; }
      #tab-tasks.overlay-panel {
        max-height: calc(100vh - 82px);
        overflow-y: auto;
      }
    }
    @keyframes overlayEnter {
      from { opacity: 0; transform: translate(-50%, 12px); }
      to { opacity: 1; transform: translate(-50%, 0); }
    }
    #tab-logs.overlay-panel {
      width: min(1120px, calc(100vw - 40px));
    }
    .log-shell {
      display: grid;
      gap: 10px;
      clear: both;
    }
    .log-toolbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .log-toolbar button {
      width: auto;
      min-height: 34px;
      padding: 0 13px;
    }
    .log-console {
      width: 100%;
      height: min(62vh, 560px);
      min-height: 320px;
      resize: none;
      border: 1px solid var(--ui-line);
      border-radius: 13px;
      padding: 12px 14px;
      background: color-mix(in srgb, var(--ui-strong-surface) 94%, transparent);
      color: var(--ui-text);
      font-family: "Cascadia Mono", "SFMono-Regular", Consolas, monospace;
      font-size: 11px;
      line-height: 1.5;
    }

  </style>
</head>
<body data-theme="dark">
  <div id="toastStack" class="toast-stack"></div>
  <div class="page control-root">
    <header class="topbar">
      <div class="brand-block">
        <div class="brand-row">
          <h1>Vehicle Motion Monitor</h1>
          <div class="top-status-cluster">
            <div class="status-pill"><span class="status-dot"></span><span>autorun online</span></div>
            <div class="status-pill compact"><strong id="controlModeStatus">Control: Idle</strong></div>
          </div>
        </div>
      </div>
      <div class="topbar-actions">
        <button class="secondary slim" style="width:auto" onclick="toggleTheme()">Theme</button>
        <button id="canConnectTopBtn" class="secondary slim" style="width:auto" onclick="connectCan()">Connect CAN</button>
        <button id="tabBtn-logs" class="secondary slim" style="width:auto" onclick="selectTab('logs')">Logs</button>
      </div>
    </header>

    <section class="control-grid">
      <main class="workspace">
        <section id="tab-dashboard" class="tab-panel active">
          <div class="overview-grid">
            <div class="panel preview-panel">
              <div class="panel-head tight">
                <div>
                  <h2>UVC Live View</h2>
                </div>
              </div>
              <div class="preview-stage">
                <img id="preview" class="preview" alt="preview">
                <div class="preview-footer">
                  <span>Source <strong id="previewSourceDashboard">Waiting</strong></span>
                  <span>Camera <strong id="cameraStatusDashboard">Stopped</strong></span>
                </div>
              </div>
            </div>

            <div class="panel status-panel" aria-label="Vehicle instruments">
              <div class="dashboard-metrics">
                <article class="run-metric">
                  <span>Speed</span>
                  <strong id="metricSpeed">--<small>m/s</small></strong>
                </article>
                <article class="run-metric">
                  <span>Steering</span>
                  <strong id="metricSteering">--<small>°</small></strong>
                </article>
                <article class="run-metric">
                  <span>Localization</span>
                  <strong id="metricLocalization">Not started</strong>
                </article>
                <article class="run-metric">
                  <span>Guidance</span>
                  <strong id="metricGuidance">Waiting /scan</strong>
                </article>
                <article class="run-metric">
                  <span>Center Offset</span>
                  <strong id="metricCenterOffset">--<small>m</small></strong>
                </article>
                <article class="run-metric">
                  <span>Battery</span>
                  <strong id="metricBattery">--<small>%</small></strong>
                </article>
              </div>
              <div id="vehicleStatus"></div>
              <nav class="floating-tools" aria-label="Control panels">
                <button id="tabBtn-tasks" class="secondary" onclick="selectTab('tasks')">Workflow</button>
                <button id="tabBtn-library" class="secondary" onclick="selectTab('library')">Library</button>
                <button id="tabBtn-settings" class="secondary" onclick="selectTab('settings')">Tuning</button>
              </nav>
            </div>

            <div class="panel lidar-panel">
              <section class="lidar-example" aria-labelledby="lidarExampleTitle">
                <div class="lidar-example-head">
                  <div>
                    <div class="lidar-example-title">
                      <h2 id="lidarExampleTitle">Lidar Channel Geometry</h2>
                      <span class="lidar-sample-badge" id="lidarDataBadge">Waiting for /scan</span>
                    </div>
                  </div>
                  <div class="lidar-geometry" aria-label="Lidar geometry parameters">
                    <span class="lidar-chip">Channel <strong id="lidarChannelState">--</strong></span>
                    <span class="lidar-chip">Row <strong id="lidarRowWidth">--</strong></span>
                    <span class="lidar-chip">Vehicle <strong>0.40 × 0.62 m</strong></span>
                    <span class="lidar-chip">Lookahead <strong>0.60 m</strong></span>
                    <span class="lidar-chip">Points <strong id="lidarPointCount">--</strong></span>
                    <span class="lidar-chip">Scan <strong id="lidarScanRate">-- Hz</strong></span>
                  </div>
                </div>
                <div class="lidar-canvas-wrap">
                  <canvas id="lidarExampleCanvas" aria-label="Example lidar point cloud with row boundaries, centerline and scaled vehicle footprint"></canvas>
                  <div class="lidar-axis-note">+X forward · lateral crop ±0.75 m</div>
                </div>
                <div class="lidar-example-foot">
                  <div class="lidar-legend">
                    <span><i class="lidar-key"></i>Raw scan</span>
                    <span><i class="lidar-key left"></i>Left fit</span>
                    <span><i class="lidar-key right"></i>Right fit</span>
                    <span><i class="lidar-key center"></i>Centerline</span>
                  </div>
                  <span class="lidar-example-note" id="lidarDataNote">Detection window <strong>0.15–1.60 m</strong> · side clearance <strong>0.10 m / side</strong></span>
                </div>
              </section>
            </div>

          </div>
        </section>

        <section id="tab-tasks" class="tab-panel overlay-panel">
          <button class="secondary overlay-close" onclick="selectTab('dashboard')">Close</button>
          <div class="workflow-shell fresh-workflow">
            <div class="workflow-board primary-workflow">
              <article class="workflow-step map">
                <div class="workflow-copy">
                  <span class="task-tag map">Mapping</span>
                  <h2>Build map</h2>
                  <p>Create a fresh Odin map.</p>
                </div>
                <div class="fields">
                  <div class="field">
                    <label for="mapName">Map Name</label>
                    <input id="mapName">
                  </div>
                  <label class="toggle-row"><input type="checkbox" id="mappingRecorddata"><span>Enable recorddata during mapping</span></label>
                </div>
                <button id="mappingActionBtn" onclick="toggleMapping()">Start Mapping</button>
              </article>

              <article class="workflow-step loc">
                <div class="workflow-copy">
                  <span class="task-tag loc">Relocalize</span>
                  <h2>Lock map</h2>
                  <p>Select the active session map.</p>
                </div>
                <div class="fields">
                  <div class="field">
                    <label for="recordMap">Map</label>
                    <select id="recordMap" onchange="onRecordMapChanged()"></select>
                  </div>
                  <div class="step-dependency"><strong>Session Map</strong><br>Recording and drive reuse this localized map.</div>
                </div>
                <button id="localizationActionBtn" onclick="toggleLocalization()">Start Localization</button>
              </article>
            </div>

            <div id="lockedStage" class="locked-stage panel compact-locked">
              <div class="status-badge warn" style="width:fit-content">Waiting for relocalization</div>
              <h2>Record + Drive unlock after map lock</h2>
              <p>Once localization succeeds, the remaining two operation cards appear here.</p>
            </div>

            <div id="deferredStage" class="deferred-stage">
              <div class="deferred-intro panel">
                <div>
                  <h2>Localization ready</h2>
                  <div class="panel-sub">Teach a path or run hybrid drive on the active localized map.</div>
                </div>
                <div class="status-badge ready">Unlocked</div>
              </div>

              <div class="workflow-board secondary-workflow">
                <article class="workflow-step record">
                  <div class="workflow-copy">
                    <span class="task-tag record">Recording</span>
                    <h2>Teach path</h2>
                    <p>Record a mission after localization settles.</p>
                  </div>
                  <div class="fields">
                    <div class="field">
                      <label for="missionName">Mission Name</label>
                      <input id="missionName">
                    </div>
                    <div class="step-dependency" id="recordDependency"><strong>Localization Ready</strong><br>You can record on the active map.</div>
                  </div>
                  <button id="recordingActionBtn" onclick="toggleRecording()">Start Recording</button>
                </article>

                <article class="workflow-step drive">
                  <div class="panel-head tight">
                    <div class="workflow-copy">
                      <span class="task-tag drive">Hybrid Drive</span>
                      <h2>Replay mission</h2>
                      <p>Global mission tracking with local row guidance.</p>
                    </div>
                    <div class="status-badge" id="driveStageBadge">Standby</div>
                  </div>
                  <div class="field-inline">
                    <div class="field">
                      <label for="driveMap">Map</label>
                      <select id="driveMap" onchange="onDriveMapChanged()"></select>
                    </div>
                    <div class="field">
                      <label for="missionSelect">Mission</label>
                      <select id="missionSelect" onchange="onMissionChanged()"></select>
                    </div>
                  </div>
                  <div class="field-inline">
                    <div class="field">
                      <label>Control Strategy</label>
                      <div class="step-dependency"><strong>Lidar in-row / staged global</strong><br>Global control is used only for mission transition stages.</div>
                    </div>
                  </div>
                  <div class="step-dependency" id="driveDependency"><strong>Localization Ready</strong><br>Choose a mission on the active map, then begin hybrid drive.</div>
                  <button id="driveActionBtn" onclick="toggleDrive()">Start Hybrid Drive</button>
                </article>
              </div>
            </div>
          </div>
        </section>

        <section id="tab-library" class="tab-panel overlay-panel">
          <button class="secondary overlay-close" onclick="selectTab('dashboard')">Close</button>
          <div class="settings-layout library-layout">
            <div class="settings-group">
              <div class="panel">
                <div class="panel-head tight">
                  <div>
                    <h2>Map Library</h2>
                    <div class="panel-sub">Delete full map folders.</div>
                  </div>
                </div>
                <div class="subpanel">
                  <div class="field">
                    <label for="libraryMapSelect">Map</label>
                    <select id="libraryMapSelect" onchange="onLibraryMapChanged()"></select>
                  </div>
                  <div class="note"><strong>Selected Map</strong><br><span id="mapDeleteSummary">No map selected.</span></div>
                  <button class="secondary" onclick="deleteSelectedMap()">Delete Selected Map Folder</button>
                </div>
              </div>
            </div>
            <div class="settings-group">
              <div class="panel">
                <div class="panel-head tight">
                  <div>
                    <h2>Mission Library</h2>
                    <div class="panel-sub">Delete recorded missions and paired csv files.</div>
                  </div>
                </div>
                <div class="subpanel">
                  <div class="field">
                    <label for="libraryMissionSelect">Mission</label>
                    <select id="libraryMissionSelect" onchange="onLibraryMissionChanged()"></select>
                  </div>
                  <div class="note"><strong>Selected Mission</strong><br><span id="missionDeleteSummary">No mission selected.</span></div>
                  <div class="note">
                    <strong>Mission Preview</strong><br>
                    <div class="route-preview-wrap">
                      <canvas id="missionPreviewCanvas" width="640" height="260"></canvas>
                      <div id="missionPreviewMeta" class="panel-sub" style="margin-top:10px">Select a mission to preview its path.</div>
                    </div>
                  </div>
                  <button class="secondary" onclick="deleteSelectedMission()">Delete Selected Mission</button>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section id="tab-settings" class="tab-panel overlay-panel">
          <button class="secondary overlay-close" onclick="selectTab('dashboard')">Close</button>
          <div class="settings-layout tuning-layout">
            <div class="tuning-main-stack">
              <div class="panel">
                <div class="panel-head tight">
                  <div>
                    <h2>Line Guidance</h2>
                    <div class="panel-sub">Normal row-follow speed. Control-source switching is stage based.</div>
                  </div>
                  <div class="status-badge">Saved globally</div>
                </div>
                <div class="subpanel">
                  <div class="field-inline">
                    <div class="field"><label for="lineCruiseVx">Cruise vx</label><input id="lineCruiseVx"></div>
                  </div>
                  <div class="note"><strong>Control source</strong><br>Normal row travel uses lidar guidance. Startup, row-end, reverse transition, and crab row-change stages use global path control.</div>
                </div>
              </div>

              <div class="panel">
                <div class="panel-head tight">
                  <div>
                    <h2>Ground Projection</h2>
                    <div class="panel-sub">Project sensor pose back to the vehicle center.</div>
                  </div>
                </div>
                <div class="subpanel">
                  <div class="field-inline">
                    <div class="field"><label for="sensorHeight">Sensor Height (m)</label><input id="sensorHeight"></div>
                    <div class="field"><label for="bodyXOffset">Body X Offset (m)</label><input id="bodyXOffset"></div>
                  </div>
                  <div class="field-inline">
                    <div class="field"><label for="bodyYOffset">Body Y Offset (m)</label><input id="bodyYOffset"></div>
                    <div class="field"><label for="rollGain">Roll Gain</label><input id="rollGain"></div>
                  </div>
                  <div class="field-inline">
                    <div class="field"><label for="pitchGain">Pitch Gain</label><input id="pitchGain"></div>
                    <div class="field"></div>
                  </div>
                  <div class="button-row">
                    <button onclick="captureProjectionAnchor()">Capture Anchor</button>
                    <button class="secondary" onclick="saveSettings()">Save Tuning</button>
                  </div>
                </div>
              </div>

              <div class="notes-grid tuning-notes">
                <div class="note">Fields remain editable while background refresh runs. Unsaved values are preserved until you press <code>Save Tuning</code>.</div>
                <div class="note">Saved values are written into <code>gui_settings.json</code> and reused by mapping, recording, and hybrid drive.</div>
              </div>
            </div>
          </div>
        </section>

        <section id="tab-logs" class="tab-panel overlay-panel">
          <button class="secondary overlay-close" onclick="selectTab('dashboard')">Close</button>
          <div class="log-shell">
            <div class="log-toolbar">
              <div>
                <h2>Logs</h2>
                <div class="panel-sub">Backend events and task transitions.</div>
              </div>
              <button class="secondary" onclick="copyAllLogs()">Copy All</button>
            </div>
            <textarea id="console" class="log-console" readonly></textarea>
          </div>
        </section>
      </main>
    </section>
  </div>
<script>
    let stateCache = null;
    let lidarFrame = null;
    let lidarRequestInFlight = false;
    let lidarLastDrawAt = 0;
    let activeTab = 'dashboard';
    const dirtyFields = new Set();
    let consoleAutoFollow = true;
    const defaultTheme = window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
    const editableFieldIds = [
      'mapName', 'missionName', 'lineCruiseVx',
      'mappingRecorddata', 'sensorHeight', 'bodyXOffset',
      'bodyYOffset', 'rollGain', 'pitchGain'
    ];
    const gamepadClientId = sessionStorage.getItem('autorun_gamepad_client_id') ||
      ('web-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2));
    sessionStorage.setItem('autorun_gamepad_client_id', gamepadClientId);
    const browserGamepad = {
      clientId: gamepadClientId,
      enabled: false,
      index: null,
      gear: '4t4d',
      speedMode: 'low',
      previousButtons: {},
      requestInFlight: false,
      lastSendAt: 0,
      lastErrorAt: 0,
      driveAxis: 0,
      steerAxis: 0,
      deadman: false,
      faulted: false,
      claimedThisPage: false,
      autoClaimInFlight: false,
      autoArmNeedsRtRelease: true,
      nextAutoClaimAt: 0,
    };

    function setTheme(theme) {
      document.body.setAttribute('data-theme', theme);
      localStorage.setItem('autorun_final_theme', theme);
    }
    function toggleTheme() {
      const current = document.body.getAttribute('data-theme') || defaultTheme;
      setTheme(current === 'dark' ? 'light' : 'dark');
    }
    async function api(path, method='GET', body=null) {
      const options = { method, headers: {} };
      if (body !== null) {
        options.headers['Content-Type'] = 'application/json';
        options.body = JSON.stringify(body);
      }
      const res = await fetch(path, options);
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || ('HTTP ' + res.status));
      }
      const contentType = res.headers.get('content-type') || '';
      if (contentType.includes('application/json')) return await res.json();
      return await res.text();
    }

    function gamepadButtonValue(gamepad, index) {
      const button = gamepad?.buttons?.[index];
      if (!button) return 0;
      return Math.max(0, Math.min(1, Number(button.value ?? (button.pressed ? 1 : 0)) || 0));
    }

    function gamepadButtonPressed(gamepad, index) {
      const button = gamepad?.buttons?.[index];
      return !!button && (!!button.pressed || gamepadButtonValue(gamepad, index) >= 0.5);
    }

    function gamepadAxis(value, deadzone=0.12) {
      const number = Math.max(-1, Math.min(1, Number(value) || 0));
      const magnitude = Math.abs(number);
      if (magnitude <= deadzone) return 0;
      return Math.sign(number) * (magnitude - deadzone) / (1 - deadzone);
    }

    function isXboxGamepad(gamepad) {
      const id = String(gamepad?.id || '').toLowerCase();
      return id.includes('xbox') || id.includes('x-box') || id.includes('xinput') || id.includes('045e');
    }

    function currentBrowserGamepad() {
      if (!('getGamepads' in navigator)) return null;
      try {
        const pads = Array.from(navigator.getGamepads() || [])
          .filter(pad => !!pad && pad.connected && isXboxGamepad(pad));
        if (browserGamepad.index !== null) {
          const selected = pads.find(pad => pad.index === browserGamepad.index);
          if (selected) return selected;
        }
        const first = pads[0] || null;
        if (first) browserGamepad.index = first.index;
        return first;
      } catch (err) {
        return null;
      }
    }

    function setBrowserGamepadGear(gear) {
      if (!['4t4d', 'crab', 'park', 'neutral'].includes(gear)) return;
      browserGamepad.gear = gear;
    }

    function stepBrowserGamepadSpeed(delta) {
      const modes = ['low', 'medium', 'high'];
      const current = Math.max(0, modes.indexOf(browserGamepad.speedMode));
      browserGamepad.speedMode = modes[Math.max(0, Math.min(modes.length - 1, current + delta))];
    }

    function handleBrowserGamepadButtons(gamepad) {
      const actions = {
        0: () => setBrowserGamepadGear('4t4d'),
        1: () => setBrowserGamepadGear('crab'),
        2: () => setBrowserGamepadGear('park'),
        3: () => setBrowserGamepadGear('neutral'),
        12: () => stepBrowserGamepadSpeed(+1),
        13: () => stepBrowserGamepadSpeed(-1),
      };
      for (const [indexText, action] of Object.entries(actions)) {
        const index = Number(indexText);
        const pressed = gamepadButtonPressed(gamepad, index);
        if (pressed && !browserGamepad.previousButtons[index]) action();
        browserGamepad.previousButtons[index] = pressed;
      }
    }

    function updateBrowserGamepadReadout(gamepad) {
      const device = document.getElementById('gamepadDeviceName');
      const deadman = document.getElementById('gamepadDeadmanValue');
      const drive = document.getElementById('gamepadDriveValue');
      const steer = document.getElementById('gamepadSteerValue');
      if (device) device.textContent = gamepad ? gamepad.id : 'Not detected; press a controller button';
      if (deadman) deadman.textContent = browserGamepad.deadman ? 'Held' : 'Released';
      if (drive) drive.textContent = browserGamepad.driveAxis.toFixed(2);
      if (steer) steer.textContent = browserGamepad.steerAxis.toFixed(2);
    }

    async function sendBrowserGamepadCommand(gamepad, now) {
      if (!browserGamepad.enabled || browserGamepad.requestInFlight) return;
      if (now - browserGamepad.lastSendAt < 80) return;
      browserGamepad.lastSendAt = now;
      browserGamepad.requestInFlight = true;
      try {
        await api('/api/gamepad/command', 'POST', {
          client_id: browserGamepad.clientId,
          connected: !!gamepad,
          device_name: gamepad?.id || '',
          deadman: !!gamepad && browserGamepad.deadman,
          gear: browserGamepad.gear,
          speed_mode: browserGamepad.speedMode,
          drive_axis: browserGamepad.driveAxis,
          steer_axis: browserGamepad.steerAxis,
        });
      } catch (err) {
        releaseBrowserGamepadBeacon('Command channel error');
        browserGamepad.enabled = false;
        browserGamepad.faulted = true;
        const stamp = performance.now();
        if (stamp - browserGamepad.lastErrorAt > 1500) {
          showToast(String(err), 'error');
          browserGamepad.lastErrorAt = stamp;
        }
      } finally {
        browserGamepad.requestInFlight = false;
      }
    }

    async function autoClaimBrowserGamepad(gamepad, now) {
      if (!gamepad || browserGamepad.enabled || browserGamepad.autoClaimInFlight) return;
      if (now < browserGamepad.nextAutoClaimAt) return;
      const rtValue = gamepadButtonValue(gamepad, 7);
      if (browserGamepad.autoArmNeedsRtRelease) {
        if (rtValue >= 0.10) return;
        browserGamepad.autoArmNeedsRtRelease = false;
      }
      browserGamepad.autoClaimInFlight = true;
      browserGamepad.nextAutoClaimAt = now + 2000;
      try {
        await api('/api/gamepad/claim', 'POST', {
          client_id: browserGamepad.clientId,
          connected: true,
          device_name: gamepad.id || 'Xbox Controller',
        });
        browserGamepad.enabled = true;
        browserGamepad.faulted = false;
        browserGamepad.claimedThisPage = true;
        browserGamepad.lastSendAt = 0;
        showToast('Xbox controller connected. Hold RT to move.');
      } catch (err) {
        browserGamepad.enabled = false;
        browserGamepad.claimedThisPage = false;
      } finally {
        browserGamepad.autoClaimInFlight = false;
      }
    }

    function pollBrowserGamepad(now) {
      const gamepad = currentBrowserGamepad();
      if (gamepad) {
        handleBrowserGamepadButtons(gamepad);
        const rightYAxis = gamepad.mapping === 'standard' ? 3 : (gamepad.axes.length > 4 ? 4 : 3);
        browserGamepad.driveAxis = -gamepadAxis(gamepad.axes[rightYAxis] || 0);
        browserGamepad.steerAxis = -gamepadAxis(gamepad.axes[0] || 0);
        browserGamepad.deadman = gamepadButtonValue(gamepad, 7) >= 0.35;
      } else {
        browserGamepad.driveAxis = 0;
        browserGamepad.steerAxis = 0;
        browserGamepad.deadman = false;
      }
      updateBrowserGamepadReadout(gamepad);
      autoClaimBrowserGamepad(gamepad, now);
      sendBrowserGamepadCommand(gamepad, now);
      window.requestAnimationFrame(pollBrowserGamepad);
    }

    async function releaseBrowserGamepad(reason='Released') {
      const wasEnabled = browserGamepad.enabled;
      browserGamepad.enabled = false;
      browserGamepad.faulted = true;
      browserGamepad.claimedThisPage = false;
      browserGamepad.autoArmNeedsRtRelease = true;
      browserGamepad.deadman = false;
      browserGamepad.driveAxis = 0;
      browserGamepad.steerAxis = 0;
      updateBrowserGamepadReadout(currentBrowserGamepad());
      try {
        await api('/api/gamepad/release', 'POST', {
          client_id: browserGamepad.clientId,
          reason,
        });
        if (wasEnabled) showToast('Browser controller stopped and released.');
      } catch (err) {
        if (wasEnabled) showToast(String(err), 'error');
      }
    }

    function releaseBrowserGamepadBeacon(reason) {
      if (!browserGamepad.enabled) return;
      browserGamepad.enabled = false;
      browserGamepad.faulted = true;
      browserGamepad.claimedThisPage = false;
      browserGamepad.autoArmNeedsRtRelease = true;
      const body = JSON.stringify({client_id: browserGamepad.clientId, reason});
      try {
        navigator.sendBeacon('/api/gamepad/release', new Blob([body], {type: 'application/json'}));
      } catch (err) {
        // The backend command timeout remains the final fail-safe.
      }
    }

    function updateGamepadControlState(control) {
      const state = control || {};
      const owner = String(state.owner_client_id || '');
      const enabled = !!state.enabled;
      const sameBrowserLease = enabled && owner === browserGamepad.clientId;
      const ownedByThisBrowser =
        sameBrowserLease && browserGamepad.claimedThisPage && !browserGamepad.faulted;
      browserGamepad.enabled = ownedByThisBrowser;
      if (ownedByThisBrowser) {
        setBrowserGamepadGear(String(state.gear || browserGamepad.gear));
        browserGamepad.speedMode = String(state.speed_mode || browserGamepad.speedMode);
      }
      const label = String(state.control_label || 'Idle');
      const topStatus = document.getElementById('controlModeStatus');
      if (topStatus) topStatus.textContent = 'Control: ' + label;
    }

    function setOptions(selectId, items, selected) {
      const sel = document.getElementById(selectId);
      if (!sel) return;
      if (document.activeElement === sel) {
        return;
      }
      const previousValue = sel.value || '';
      const desiredValue = selected || previousValue || '';
      const signature = JSON.stringify((items || []).map(item => [item.id, item.label]));
      if (sel.dataset.optionsSignature === signature && previousValue === desiredValue) {
        return;
      }
      sel.innerHTML = '';
      for (const item of items) {
        const opt = document.createElement('option');
        opt.value = item.id;
        opt.textContent = item.label;
        sel.appendChild(opt);
      }
      const desiredExists = (items || []).some(item => item.id === desiredValue);
      if (desiredExists) {
        sel.value = desiredValue;
      }
      if (!sel.value && previousValue && (items || []).some(item => item.id === previousValue)) {
        sel.value = previousValue;
      }
      if (!sel.value && sel.options.length > 0) {
        sel.selectedIndex = 0;
      }
      sel.dataset.optionsSignature = signature;
    }

    function showToast(message, kind='success') {
      const stack = document.getElementById('toastStack');
      if (!stack) return;
      const toast = document.createElement('div');
      toast.className = 'toast ' + kind;
      toast.textContent = message;
      stack.appendChild(toast);
      requestAnimationFrame(() => toast.classList.add('show'));
      setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 220);
      }, 2800);
    }

    function updateConsoleBox(lines) {
      const consoleBox = document.getElementById('console');
      if (!consoleBox) return;
      const nearBottom = (consoleBox.scrollHeight - consoleBox.scrollTop - consoleBox.clientHeight) < 24;
      consoleBox.value = (lines || []).join('\\n');
      if (consoleAutoFollow || nearBottom) {
        consoleBox.scrollTop = consoleBox.scrollHeight;
        consoleAutoFollow = true;
      }
    }

    async function copyAllLogs() {
      const consoleBox = document.getElementById('console');
      const text = String(consoleBox?.value || '');
      if (!text) {
        showToast('No logs to copy.');
        return;
      }
      try {
        if (navigator.clipboard && window.isSecureContext) {
          await navigator.clipboard.writeText(text);
        } else {
          consoleBox.focus();
          consoleBox.select();
          const copied = document.execCommand('copy');
          consoleBox.setSelectionRange(text.length, text.length);
          if (!copied) throw new Error('Copy command was rejected');
        }
        showToast('All logs copied.');
      } catch (err) {
        showToast('Unable to copy logs. Select the text and copy it manually.', 'error');
      }
    }

    function formatNumber(value, digits=2) {
      const num = Number(value);
      if (!Number.isFinite(num)) return value ?? '--';
      return num.toFixed(digits).replace(/\.?0+$/, '');
    }

    function formatDriveMode(value) {
      const normalized = String(value ?? '').trim().toLowerCase();
      const labels = {
        '1': 'Park',
        '2': 'Neutral',
        '5': '4-wheel steer',
        '6': '4-wheel steer',
        '7': 'Crab',
        '8': 'Crab',
        park: 'Park',
        neutral: 'Neutral',
        '4t4d': '4-wheel steer',
        crab: 'Crab',
      };
      return labels[normalized] || (normalized ? String(value) : '--');
    }

    function formatLocalizationLabel(value) {
      const normalized = String(value || '').trim().toLowerCase();
      if (normalized === 'ready') return 'Localized';
      if (normalized === 'not started') return 'Not started';
      if (normalized === 'stopped') return 'Stopped';
      if (normalized.includes('fail')) return 'Failed';
      return value || '--';
    }

    function setTextContent(id, value) {
      const element = document.getElementById(id);
      if (element) element.textContent = value ?? '--';
    }

    function setMetricValue(id, value, unit='') {
      const element = document.getElementById(id);
      if (!element) return;
      element.textContent = String(value ?? '--');
      if (unit) {
        const suffix = document.createElement('small');
        suffix.textContent = unit;
        element.appendChild(suffix);
      }
    }

    function updateVehicleStatus(status) {
      const root = document.getElementById('vehicleStatus');
      if (!root) return;
      const vxRaw = Number(status.motion?.vx_mps ?? status.motion?.vx);
      const vyRaw = Number(status.motion?.vy_mps ?? status.motion?.vy);
      const speed = Number.isFinite(vxRaw) && Number.isFinite(vyRaw)
        ? formatNumber(Math.hypot(vxRaw, vyRaw), 2)
        : '--';
      const steering = formatNumber(status.steering?.wheel_angle_deg ?? '--', 1);
      const soc = status.battery?.soc_pct ?? '--';
      setMetricValue('metricSpeed', speed, 'm/s');
      setMetricValue('metricSteering', steering, '°');
      setMetricValue('metricBattery', soc, '%');
      const charging = status.battery?.charging;
      const currentMode = formatDriveMode(status.motion?.gear ?? status.steering?.gear);
      const entries = [
        ['Current Mode', currentMode, ''],
        ['Charging', charging === true ? 'Charging' : charging === false ? 'Not charging' : '--', charging === true ? 'charging' : ''],
      ];
      root.innerHTML = entries
        .map(([k, v, className]) => `<div class="vehicle-metric ${className}"><span>${k}</span><strong>${v}</strong></div>`)
        .join('');
    }

    function updateProjectionDebug(debug) {
      const root = document.getElementById('projectionDebug');
      if (!root) return;
      const entries = [
        ['Monitor', debug?.monitor_status ?? '--'],
        ['Raw Pose', debug?.raw_xy ?? '--'],
        ['Raw Attitude', debug?.raw_rp ?? '--'],
        ['Anchor', debug?.anchor ?? '--'],
        ['Projected Pose', debug?.proj_xy ?? '--'],
        ['Projected Delta', debug?.proj_delta ?? '--'],
        ['Projected RPY', debug?.proj_rpy ?? '--'],
      ];
      root.innerHTML = entries.map(([k, v]) => `<div>${k}</div><div>${v}</div>`).join('');
    }

    function setActionButton(id, active, startLabel, stopLabel) {
      const btn = document.getElementById(id);
      if (!btn) return;
      btn.textContent = active ? stopLabel : startLabel;
      btn.classList.toggle('warn', active);
      btn.classList.remove('secondary');
      if (id === 'localizationActionBtn' && !active) btn.classList.add('secondary');
    }

    function setBadge(id, text, mode='wait') {
      const el = document.getElementById(id);
      if (!el) return;
      el.textContent = text;
      el.className = 'status-badge ' + mode;
    }

    function isLocalizationActive(statusText) {
      const text = String(statusText || '').toLowerCase();
      return !['not started', 'stopped', 'failed'].includes(text);
    }

    function isLocalizationReady(statusText) {
      return String(statusText || '').toLowerCase() === 'ready';
    }

    function isTaskRunning(taskStatus, name) {
      return String(taskStatus || '') === name;
    }

    function selectTab(tabName) {
      const overlayTabs = ['tasks', 'library', 'settings', 'logs'];
      const requested = overlayTabs.includes(tabName) ? tabName : 'dashboard';
      const nextTab = requested !== 'dashboard' && activeTab === requested ? 'dashboard' : requested;
      activeTab = nextTab;
      document.body.classList.toggle('workflow-overlay-open', nextTab === 'tasks');
      document.getElementById('tab-dashboard')?.classList.add('active');
      for (const name of overlayTabs) {
        document.getElementById('tab-' + name)?.classList.toggle('active', name === nextTab);
        document.getElementById('tabBtn-' + name)?.classList.toggle('active', name === nextTab);
      }
    }

    document.addEventListener('keydown', event => {
      if (event.key === 'Escape' && activeTab !== 'dashboard') selectTab('dashboard');
    });

    async function onRecordMapChanged() {
      const mapId = document.getElementById('recordMap').value;
      await api('/api/select_map', 'POST', { role: 'record', map_id: mapId });
      await refreshState();
    }

    async function onDriveMapChanged() {
      const mapId = document.getElementById('driveMap').value;
      await api('/api/select_map', 'POST', { role: 'drive', map_id: mapId });
      await refreshState();
    }

    async function onMissionChanged() {
      const missionId = document.getElementById('missionSelect').value;
      await api('/api/select_mission', 'POST', { mission_id: missionId });
      await refreshState();
    }

    async function onLibraryMapChanged() {
      const mapId = document.getElementById('libraryMapSelect').value;
      await api('/api/select_map', 'POST', { role: 'library', map_id: mapId });
      await refreshState();
    }

    async function onLibraryMissionChanged() {
      const missionId = document.getElementById('libraryMissionSelect').value;
      await api('/api/select_mission', 'POST', { role: 'library', mission_id: missionId });
      await refreshState();
    }

    function updateFieldIfClean(id, value, isCheckbox=false) {
      const el = document.getElementById(id);
      if (!el) return;
      if (dirtyFields.has(id)) return;
      if (document.activeElement === el) return;
      if (isCheckbox) {
        el.checked = !!value;
      } else {
        el.value = value ?? '';
      }
    }

    function markFieldDirty(id) {
      const el = document.getElementById(id);
      if (!el) return;
      const evt = (el.type === 'checkbox') ? 'change' : 'input';
      el.addEventListener(evt, () => dirtyFields.add(id));
    }

    function updateWorkflowState(data) {
      const ready = isLocalizationReady(data.localization_status);
      const activeLocalizationMapId = data.active_localization_map_id || '';
      const canState = String(data.can_status || 'Unknown').toUpperCase();
      const canReady = ['UP', 'UNKNOWN'].includes(canState);
      const selectedMap = (data.maps || []).find(
        item => item.id === (activeLocalizationMapId || data.selected_replay_map_id || data.selected_record_map_id)
      );
      const selectedMission = (data.missions || []).find(item => item.id === data.selected_mission_id);
      setTextContent('workflowGateSummary', ready ? 'Ready for record or drive' : 'Waiting for map lock');
      const workflowGateTuning = document.getElementById('workflowGateTuning');
      if (workflowGateTuning) workflowGateTuning.textContent = ready ? 'Ready for record or drive' : 'Waiting for map lock';
      document.getElementById('recordDependency').innerHTML = ready
        ? '<strong>Localization ready</strong><br>You can record on the current localized map.'
        : '<strong>Needs localization</strong><br>Start relocalization first, then record on the same map.';
      document.getElementById('driveDependency').innerHTML = ready
        ? '<strong>Localization ready</strong><br>' + (selectedMission ? 'Mission ' + selectedMission.label + ' can now be driven on ' + (selectedMap ? selectedMap.label : 'the active map') + '.' : 'Choose a mission that belongs to the active map.')
        : '<strong>Needs localization and a recorded mission</strong><br>Choose a mission that belongs to the active map, then begin hybrid drive.';

      setBadge('localizationGateBadge', ready ? 'Ready' : 'Waiting', ready ? 'ready' : 'wait');
      setBadge('recordingGateBadge', ready ? 'Unlocked' : 'Locked', ready ? 'ready' : 'warn');
      setBadge('driveGateBadge', ready ? 'Unlocked' : 'Locked', ready ? 'ready' : 'warn');

      const driveMode = isTaskRunning(data.task_status, 'Hybrid Drive') ? 'ready' : (ready ? 'wait' : 'warn');
      const driveLabel = isTaskRunning(data.task_status, 'Hybrid Drive') ? 'Running' : (ready ? 'Armed' : 'Standby');
      setBadge('driveStageBadge', driveLabel, driveMode);

      const recordingActive = isTaskRunning(data.task_status, 'Path Recording');
      const driveActive = isTaskRunning(data.task_status, 'Hybrid Drive');
      const recordingBtn = document.getElementById('recordingActionBtn');
      const driveBtn = document.getElementById('driveActionBtn');
      const missionSelect = document.getElementById('missionSelect');
      const driveMapSelect = document.getElementById('driveMap');
      const canBtn = document.getElementById('canConnectBtn');
      const canTopBtn = document.getElementById('canConnectTopBtn');
      const deferredStage = document.getElementById('deferredStage');
      const lockedStage = document.getElementById('lockedStage');
      if (deferredStage) deferredStage.classList.toggle('visible', ready || recordingActive || driveActive);
      if (lockedStage) lockedStage.style.display = (ready || recordingActive || driveActive) ? 'none' : 'grid';
      if (recordingBtn) recordingBtn.disabled = !ready && !recordingActive;
      if (driveBtn) driveBtn.disabled = !ready && !driveActive;
      if (missionSelect) missionSelect.disabled = !ready && !driveActive;
      if (driveMapSelect) driveMapSelect.disabled = ready || driveActive;
      if (canBtn) {
        canBtn.textContent = canReady ? 'Reconnect CAN' : 'Connect CAN';
        canBtn.classList.toggle('secondary', canReady);
      }
      if (canTopBtn) {
        canTopBtn.textContent = canReady ? 'Reconnect CAN' : 'Connect CAN';
        canTopBtn.classList.toggle('secondary', canReady);
      }
    }

    function updateLidarPreview(preview) {
      lidarFrame = preview && typeof preview === 'object' ? preview : null;
      const live = !!(lidarFrame && lidarFrame.live);
      const badge = document.getElementById('lidarDataBadge');
      const pointCount = document.getElementById('lidarPointCount');
      const scanRate = document.getElementById('lidarScanRate');
      const channelState = document.getElementById('lidarChannelState');
      const rowWidth = document.getElementById('lidarRowWidth');
      const note = document.getElementById('lidarDataNote');
      const geometry = lidarFrame?.geometry && typeof lidarFrame.geometry === 'object'
        ? lidarFrame.geometry
        : {};
      const channelFound = live && geometry.found === true;
      const controlAccepted = channelFound && geometry.control_accepted === true;
      const centerOffset = Number(geometry.center_y_m);
      setMetricValue(
        'metricCenterOffset',
        channelFound && Number.isFinite(centerOffset) ? centerOffset.toFixed(3) : '--',
        'm',
      );
      setTextContent(
        'metricGuidance',
        controlAccepted ? 'Channel tracking' : live ? 'Lidar scanning' : 'Waiting /scan',
      );
      if (badge) {
        badge.classList.toggle('live', controlAccepted);
        badge.textContent = controlAccepted
          ? `CHANNEL · ${Number(lidarFrame.scan_hz || 0).toFixed(1)} Hz`
          : channelFound
            ? `GEOMETRY ONLY · ${Number(lidarFrame.scan_hz || 0).toFixed(1)} Hz`
          : live
            ? `NO CHANNEL · ${Number(lidarFrame.scan_hz || 0).toFixed(1)} Hz`
          : String(lidarFrame?.status || 'Waiting for /scan');
      }
      if (pointCount) pointCount.textContent = live ? String(lidarFrame.visible_count ?? lidarFrame.points?.length ?? 0) : '--';
      if (scanRate) scanRate.textContent = live ? `${Number(lidarFrame.scan_hz || 0).toFixed(1)} Hz` : '-- Hz';
      if (channelState) {
        channelState.textContent = controlAccepted
          ? 'TRACKABLE'
          : channelFound
            ? 'GEOMETRY ONLY'
            : 'NOT DETECTED';
      }
      if (rowWidth) {
        rowWidth.textContent = channelFound && Number.isFinite(Number(geometry.row_width_m))
          ? `${Number(geometry.row_width_m).toFixed(2)} m`
          : '--';
      }
      if (note) {
        note.innerHTML = live
          ? channelFound
            ? `Two-side fit · left bins <strong>${Number(geometry.left_bins || 0)}</strong> · right bins <strong>${Number(geometry.right_bins || 0)}</strong> · measured width <strong>${Number(geometry.row_width_m || 0).toFixed(3)} m</strong>${controlAccepted ? '' : ' · guidance <strong>rejected</strong>'}`
            : `Raw <strong>${Number(lidarFrame.raw_count || 0)}</strong> · valid <strong>${Number(lidarFrame.valid_count || 0)}</strong> · only raw points are shown`
          : `Detection window <strong>0.15–1.60 m</strong> · side clearance <strong>0.10 m / side</strong>`;
      }
    }

    function applyState(data) {
      stateCache = data;
      const activeLocalizationMapId = data.active_localization_map_id || '';
      const localizationText = data.localization_status || 'Not started';
      setTextContent('taskStatus', data.task_status || 'Idle');
      setTextContent('localizationStatus', localizationText);
      setTextContent('metricLocalization', formatLocalizationLabel(localizationText));
      setTextContent('cameraStatus', data.camera_status || 'Stopped');
      setTextContent('cameraStatusDashboard', data.camera_status || 'Stopped');
      setTextContent('previewSource', data.preview_source || 'Waiting');
      setTextContent('previewSourceDashboard', data.preview_source || 'Waiting');
      setTextContent('previewModeBadge', String(data.preview_source || 'Preview'));
      const previewSourceTuning = document.getElementById('previewSourceTuning');
      if (previewSourceTuning) previewSourceTuning.textContent = data.preview_source || 'Waiting for preview stream';
      const previewModeBadgeTuning = document.getElementById('previewModeBadgeTuning');
      if (previewModeBadgeTuning) previewModeBadgeTuning.textContent = String(data.preview_source || 'Preview');
      setTextContent('canStateSummary', data.can_status || 'Unknown');

      setOptions('recordMap', data.maps || [], activeLocalizationMapId || data.selected_record_map_id);
      setOptions('driveMap', data.maps || [], activeLocalizationMapId || data.selected_replay_map_id);
      setOptions('libraryMapSelect', data.maps || [], data.selected_library_map_id);
      setOptions('missionSelect', data.missions || [], data.selected_mission_id);
      setOptions('libraryMissionSelect', data.library_missions || [], data.selected_library_mission_id);

      const selectedMap = (data.maps || []).find(
        item => item.id === (activeLocalizationMapId || data.selected_replay_map_id || data.selected_record_map_id)
      );
      const selectedMission = (data.missions || []).find(item => item.id === data.selected_mission_id);
      const libraryMap = (data.maps || []).find(item => item.id === data.selected_library_map_id);
      const libraryMission = (data.library_missions || []).find(item => item.id === data.selected_library_mission_id);
      setTextContent('selectedMapSummary', selectedMap ? selectedMap.label : '--');
      setTextContent('selectedMissionSummary', selectedMission ? selectedMission.label : '--');
      document.getElementById('mapDeleteSummary').textContent = libraryMap ? libraryMap.label : 'No map selected.';
      document.getElementById('missionDeleteSummary').textContent = libraryMission ? libraryMission.label : 'No mission selected.';
      renderMissionPreview(data.selected_library_mission_preview || null);

      const projectionSummary = document.getElementById('projectionSummary');
      if (projectionSummary) {
        projectionSummary.textContent =
          'h=' + (data.settings.sensor_height_m || '--') +
          ', x=' + (data.settings.body_x_offset_m || '--') +
          ', y=' + (data.settings.body_y_offset_m || '--');
      }
      const projectionSummaryTuning = document.getElementById('projectionSummaryTuning');
      if (projectionSummaryTuning) {
        projectionSummaryTuning.textContent =
          'h=' + (data.settings.sensor_height_m || '--') +
          ', x=' + (data.settings.body_x_offset_m || '--') +
          ', y=' + (data.settings.body_y_offset_m || '--');
      }

      updateFieldIfClean('mapName', data.mapping_name || '');
      updateFieldIfClean('missionName', data.mission_name || '');
      updateFieldIfClean('mappingRecorddata', !!data.settings.mapping_recorddata, true);
      updateFieldIfClean('lineCruiseVx', data.settings.line_cruise_vx || '');
      updateFieldIfClean('sensorHeight', data.settings.sensor_height_m || '');
      updateFieldIfClean('bodyXOffset', data.settings.body_x_offset_m || '');
      updateFieldIfClean('bodyYOffset', data.settings.body_y_offset_m || '');
      updateFieldIfClean('rollGain', data.settings.roll_gain || '');
      updateFieldIfClean('pitchGain', data.settings.pitch_gain || '');

      updateConsoleBox(data.logs || []);

      updateVehicleStatus(data.vehicle_status || {});
      updateGamepadControlState(data.gamepad_control || {});
      updateProjectionDebug(data.pose_debug || {});
      updateWorkflowState(data);

      const taskStatus = String(data.task_status || '');
      setActionButton('mappingActionBtn', taskStatus === 'Mapping', 'Start Mapping', 'Stop Mapping');
      setActionButton('localizationActionBtn', isLocalizationActive(localizationText), 'Start Localization', 'Stop Localization');
      setActionButton('recordingActionBtn', taskStatus === 'Path Recording', 'Start Recording', 'Stop Recording');
      setActionButton('driveActionBtn', taskStatus === 'Hybrid Drive', 'Start Hybrid Drive', 'Stop Hybrid Drive');
    }

    async function refreshState() {
      try {
        const data = await api('/api/state');
        applyState(data);
      } catch (err) {
        console.error(err);
      }
    }

    async function refreshLidarPreview() {
      if (lidarRequestInFlight || document.hidden || activeTab !== 'dashboard') return;
      lidarRequestInFlight = true;
      try {
        updateLidarPreview(await api('/api/lidar'));
      } catch (err) {
        console.error(err);
      } finally {
        lidarRequestInFlight = false;
      }
    }

    async function saveSettings() {
      await api('/api/settings', 'POST', {
        line_cruise_vx: document.getElementById('lineCruiseVx').value,
        sensor_height_m: document.getElementById('sensorHeight').value,
        body_x_offset_m: document.getElementById('bodyXOffset').value,
        body_y_offset_m: document.getElementById('bodyYOffset').value,
        roll_gain: document.getElementById('rollGain').value,
        pitch_gain: document.getElementById('pitchGain').value,
        mapping_recorddata: document.getElementById('mappingRecorddata').checked,
      });
      for (const id of editableFieldIds) dirtyFields.delete(id);
      await refreshState();
    }

    async function captureProjectionAnchor() {
      await api('/api/capture_anchor', 'POST', {});
      await refreshState();
    }

    async function connectCan() {
      await api('/api/connect_can', 'POST', {});
      await refreshState();
    }

    async function deleteSelectedMap() {
      const targetId = document.getElementById('libraryMapSelect')?.value || '';
      if (!targetId) return;
      const selectedMap = (stateCache.maps || []).find(item => item.id === targetId);
      const label = selectedMap ? selectedMap.label : 'the selected map';
      if (!window.confirm('Delete map folder and all files?\\n' + label)) return;
      try {
        await api('/api/delete_map', 'POST', { map_id: targetId });
        await refreshState();
        showToast('Map folder deleted: ' + label, 'success');
      } catch (err) {
        console.error(err);
        showToast('Delete map failed: ' + (err?.message || err), 'error');
      }
    }

    async function deleteSelectedMission() {
      const targetId = document.getElementById('libraryMissionSelect')?.value || '';
      if (!targetId) return;
      const selectedMission = (stateCache.library_missions || []).find(item => item.id === targetId);
      const label = selectedMission ? selectedMission.label : 'the selected mission';
      if (!window.confirm('Delete mission files?\\n' + label)) return;
      try {
        await api('/api/delete_mission', 'POST', { mission_id: targetId });
        await refreshState();
        showToast('Mission deleted: ' + label, 'success');
      } catch (err) {
        console.error(err);
        showToast('Delete mission failed: ' + (err?.message || err), 'error');
      }
    }

    function renderMissionPreview(preview) {
      const canvas = document.getElementById('missionPreviewCanvas');
      const meta = document.getElementById('missionPreviewMeta');
      if (!canvas || !meta) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const dpr = Math.max(1, window.devicePixelRatio || 1);
      const cssWidth = canvas.clientWidth || 640;
      const cssHeight = canvas.clientHeight || 260;
      if (canvas.width !== Math.round(cssWidth * dpr) || canvas.height !== Math.round(cssHeight * dpr)) {
        canvas.width = Math.round(cssWidth * dpr);
        canvas.height = Math.round(cssHeight * dpr);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, cssWidth, cssHeight);

      const styles = getComputedStyle(document.body);
      const grid = styles.getPropertyValue('--line').trim() || 'rgba(120,140,170,0.18)';
      const line = styles.getPropertyValue('--accent').trim() || '#5e8bff';
      const start = styles.getPropertyValue('--good').trim() || '#29c575';
      const end = styles.getPropertyValue('--warn').trim() || '#e26b77';
      const muted = styles.getPropertyValue('--muted').trim() || '#93a3bf';
      const pad = 18;
      const w = cssWidth;
      const h = cssHeight;

      for (let i = 0; i < 6; i += 1) {
        const y = pad + ((h - pad * 2) / 5) * i;
        const x = pad + ((w - pad * 2) / 5) * i;
        ctx.strokeStyle = grid;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(pad, y);
        ctx.lineTo(w - pad, y);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x, pad);
        ctx.lineTo(x, h - pad);
        ctx.stroke();
      }

      if (!preview || !preview.points || preview.points.length < 2) {
        ctx.fillStyle = muted;
        ctx.font = '13px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('No preview available for this mission yet.', w / 2, h / 2);
        meta.textContent = 'Select a mission to preview its path.';
        return;
      }

      const points = preview.points;
      const minX = Number(preview.min_x);
      const maxX = Number(preview.max_x);
      const minY = Number(preview.min_y);
      const maxY = Number(preview.max_y);
      const spanX = Math.max(0.001, maxX - minX);
      const spanY = Math.max(0.001, maxY - minY);
      const scale = Math.min((w - pad * 2) / spanX, (h - pad * 2) / spanY);
      const offsetX = (w - spanX * scale) / 2 - minX * scale;
      const offsetY = (h - spanY * scale) / 2 - minY * scale;
      const project = (pt) => {
        const px = pt.x * scale + offsetX;
        const py = h - (pt.y * scale + offsetY);
        return [px, py];
      };

      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.strokeStyle = line;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      points.forEach((pt, idx) => {
        const [px, py] = project(pt);
        if (idx === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.stroke();

      const drawDot = (pt, color, radius) => {
        const [px, py] = project(pt);
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(px, py, radius, 0, Math.PI * 2);
        ctx.fill();
      };
      drawDot(points[0], start, 5);
      drawDot(points[points.length - 1], end, 5);

      const startPt = points[0];
      const endPt = points[points.length - 1];
      meta.textContent =
        `Samples ${preview.sample_count} · span ${spanX.toFixed(2)}m x ${spanY.toFixed(2)}m · ` +
        `start (${startPt.x.toFixed(2)}, ${startPt.y.toFixed(2)}) · end (${endPt.x.toFixed(2)}, ${endPt.y.toFixed(2)})`;
    }

    async function startMapping() {
      await saveSettings();
      await api('/api/start_mapping', 'POST', { map_name: document.getElementById('mapName').value });
      await refreshState();
    }
    async function toggleMapping() {
      const text = document.getElementById('mappingActionBtn').textContent || '';
      if (text.toLowerCase().includes('stop')) return await stopTask();
      return await startMapping();
    }
    async function startLocalization() {
      await api('/api/start_localization', 'POST', { map_id: document.getElementById('recordMap').value });
      await refreshState();
    }
    async function stopLocalization() {
      await api('/api/stop_localization', 'POST', {});
      await refreshState();
    }
    async function toggleLocalization() {
      const text = document.getElementById('localizationActionBtn').textContent || '';
      if (text.toLowerCase().includes('stop')) return await stopLocalization();
      return await startLocalization();
    }
    async function startRecording() {
      await api('/api/start_recording', 'POST', {
        map_id: document.getElementById('recordMap').value,
        mission_name: document.getElementById('missionName').value,
      });
      await refreshState();
    }
    async function toggleRecording() {
      const text = document.getElementById('recordingActionBtn').textContent || '';
      if (text.toLowerCase().includes('stop')) return await stopTask();
      return await startRecording();
    }
    async function startDrive() {
      await saveSettings();
      await api('/api/start_drive', 'POST', {
        map_id: document.getElementById('driveMap').value,
        mission_id: document.getElementById('missionSelect').value,
      });
      await refreshState();
    }
    async function toggleDrive() {
      const text = document.getElementById('driveActionBtn').textContent || '';
      if (text.toLowerCase().includes('stop')) return await stopTask();
      return await startDrive();
    }
    async function stopTask() {
      await api('/api/stop', 'POST', {});
      await refreshState();
    }

    function refreshPreview() {
      const img = document.getElementById('preview');
      img.src = '/api/preview.jpg?t=' + Date.now();
    }

    function drawLidarExample(timestamp=0) {
      const canvas = document.getElementById('lidarExampleCanvas');
      if (!canvas) return;
      if (document.hidden || activeTab !== 'dashboard' || timestamp - lidarLastDrawAt < 66) {
        window.requestAnimationFrame(drawLidarExample);
        return;
      }
      lidarLastDrawAt = timestamp;
      const ctx = canvas.getContext('2d');
      const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
      const cssWidth = Math.max(320, canvas.clientWidth || 900);
      const cssHeight = Math.max(250, canvas.clientHeight || 320);
      const pixelWidth = Math.round(cssWidth * dpr);
      const pixelHeight = Math.round(cssHeight * dpr);
      if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
        canvas.width = pixelWidth;
        canvas.height = pixelHeight;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const style = getComputedStyle(document.body);
      const themeDark = document.body.dataset.theme === 'dark';
      const color = (name, fallback) => style.getPropertyValue(name).trim() || fallback;
      const bg = color('--ui-strong-surface-2', themeDark ? '#0b1422' : '#f8fafc');
      const text = color('--ui-text', themeDark ? '#e5edf8' : '#172033');
      const muted = color('--ui-muted', themeDark ? '#94a3b8' : '#64748b');
      const line = color('--ui-line-soft', themeDark ? '#26354a' : '#dfe5ee');
      const panel = color('--ui-strong-surface', themeDark ? '#111d2d' : '#ffffff');

      const w = cssWidth;
      const h = cssHeight;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, w, h);

      const bounds = { xMin: -0.48, xMax: 1.78, yMin: -2.5, yMax: 2.5 };
      const pad = { left: 44, right: 24, top: 28, bottom: 28 };
      const plotW = w - pad.left - pad.right;
      const plotH = h - pad.top - pad.bottom;
      const metersToPixels = Math.min(
        plotW / (bounds.yMax - bounds.yMin),
        plotH / (bounds.xMax - bounds.xMin),
      );
      const usedPlotW = (bounds.yMax - bounds.yMin) * metersToPixels;
      const usedPlotH = (bounds.xMax - bounds.xMin) * metersToPixels;
      const plotOffsetX = pad.left + (plotW - usedPlotW) / 2;
      const plotOffsetY = pad.top + (plotH - usedPlotH) / 2;
      const project = (x, y) => ({
        x: plotOffsetX + (y - bounds.yMin) * metersToPixels,
        y: plotOffsetY + (bounds.xMax - x) * metersToPixels,
      });
      const linePath = (points) => {
        ctx.beginPath();
        points.forEach((point, index) => {
          const p = project(point[0], point[1]);
          if (index === 0) ctx.moveTo(p.x, p.y);
          else ctx.lineTo(p.x, p.y);
        });
      };
      const roundedRect = (x, y, width, height, radius) => {
        const r = Math.min(radius, width / 2, height / 2);
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.arcTo(x + width, y, x + width, y + height, r);
        ctx.arcTo(x + width, y + height, x, y + height, r);
        ctx.arcTo(x, y + height, x, y, r);
        ctx.arcTo(x, y, x + width, y, r);
        ctx.closePath();
      };

      ctx.lineWidth = 1;
      ctx.font = '10px ui-sans-serif, system-ui, sans-serif';
      ctx.textBaseline = 'middle';
      for (let x = -0.4; x <= 1.61; x += 0.2) {
        const major = Math.abs((x * 10) % 5) < 0.01;
        const p0 = project(x, bounds.yMin);
        const p1 = project(x, bounds.yMax);
        ctx.strokeStyle = major ? line : (themeDark ? 'rgba(148,163,184,.08)' : 'rgba(100,116,139,.08)');
        ctx.beginPath();
        ctx.moveTo(p0.x, p0.y);
        ctx.lineTo(p1.x, p1.y);
        ctx.stroke();
        if (major && x >= 0) {
          ctx.fillStyle = muted;
          ctx.textAlign = 'right';
          ctx.fillText(x.toFixed(1) + ' m', pad.left - 7, p0.y);
        }
      }
      for (let y = -2.4; y <= 2.41; y += 0.2) {
        const major = Math.abs((y * 10) % 5) < 0.01;
        const p0 = project(bounds.xMin, y);
        const p1 = project(bounds.xMax, y);
        ctx.strokeStyle = major ? line : (themeDark ? 'rgba(148,163,184,.08)' : 'rgba(100,116,139,.08)');
        ctx.beginPath();
        ctx.moveTo(p0.x, p0.y);
        ctx.lineTo(p1.x, p1.y);
        ctx.stroke();
      }

      const cropTopLeft = project(1.60, -0.75);
      const cropBottomRight = project(0.15, 0.75);
      ctx.fillStyle = themeDark ? 'rgba(59,130,246,.035)' : 'rgba(59,130,246,.025)';
      ctx.strokeStyle = themeDark ? 'rgba(96,165,250,.28)' : 'rgba(37,99,235,.19)';
      ctx.setLineDash([5, 5]);
      ctx.fillRect(cropTopLeft.x, cropTopLeft.y, cropBottomRight.x - cropTopLeft.x, cropBottomRight.y - cropTopLeft.y);
      ctx.strokeRect(cropTopLeft.x, cropTopLeft.y, cropBottomRight.x - cropTopLeft.x, cropBottomRight.y - cropTopLeft.y);
      ctx.setLineDash([]);

      const hasLiveScan = !!(lidarFrame && lidarFrame.live);
      const geometry = hasLiveScan && lidarFrame.geometry && typeof lidarFrame.geometry === 'object'
        ? lidarFrame.geometry
        : {};
      const channelFound = geometry.found === true;
      const controlAccepted = geometry.control_accepted === true;
      const lineFunction = (value) => {
        if (!Array.isArray(value) || value.length < 2) return null;
        const slope = Number(value[0]);
        const intercept = Number(value[1]);
        if (!Number.isFinite(slope) || !Number.isFinite(intercept)) return null;
        return x => slope * x + intercept;
      };
      const leftY = channelFound ? lineFunction(geometry.left_line) : null;
      const rightY = channelFound ? lineFunction(geometry.right_line) : null;
      const centerY = channelFound ? lineFunction(geometry.center_line) : null;
      const renderChannel = !!(channelFound && leftY && rightY && centerY);
      const xs = [];
      for (let x = 0.10; x <= 1.64; x += 0.04) xs.push(x);

      if (renderChannel) {
        const corridor = [];
        xs.forEach(x => corridor.push([x, leftY(x)]));
        [...xs].reverse().forEach(x => corridor.push([x, rightY(x)]));
        linePath(corridor);
        ctx.closePath();
        ctx.fillStyle = controlAccepted
          ? (themeDark ? 'rgba(34,197,94,.07)' : 'rgba(22,163,74,.055)')
          : (themeDark ? 'rgba(245,158,11,.08)' : 'rgba(217,119,6,.06)');
        ctx.fill();
      }

      const rawPoint = (x, y, radius=1.45, alpha=.68) => {
        const p = project(x, y);
        ctx.beginPath();
        ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = themeDark ? `rgba(148,163,184,${alpha})` : `rgba(71,85,105,${alpha})`;
        ctx.fill();
      };
      const liveScanPoints = hasLiveScan && Array.isArray(lidarFrame.points)
        ? lidarFrame.points
        : [];
      if (hasLiveScan) {
        liveScanPoints.forEach(point => {
          if (!Array.isArray(point) || point.length < 2) return;
          const x = Number(point[0]);
          const y = Number(point[1]);
          if (!Number.isFinite(x) || !Number.isFinite(y)) return;
          const inDetectionWindow = x >= 0.15 && x <= 1.60 && Math.abs(y) <= 0.75;
          rawPoint(x, y, inDetectionWindow ? 1.85 : 1.25, inDetectionWindow ? .88 : .46);
        });
      } else {
        const waitPoint = project(0.95, 0);
        ctx.fillStyle = muted;
        ctx.font = '600 11px ui-sans-serif, system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Waiting for live /scan · simulated points are disabled', waitPoint.x, waitPoint.y);
      }

      const drawFit = (fn, stroke, width=2.2) => {
        linePath([[0.15, fn(0.15)], [1.60, fn(1.60)]]);
        ctx.strokeStyle = stroke;
        ctx.lineWidth = width;
        ctx.setLineDash(controlAccepted ? [] : [6, 4]);
        ctx.stroke();
        ctx.setLineDash([]);
      };
      if (renderChannel) {
        drawFit(leftY, '#16a34a', 2.2);
        drawFit(rightY, '#ea580c', 2.2);
        drawFit(centerY, '#e11d48', 2.8);
        linePath([[0, centerY(0)], [-0.43, centerY(-0.43)]]);
        ctx.strokeStyle = '#93c5fd';
        ctx.lineWidth = 1.6;
        ctx.setLineDash([6, 4]);
        ctx.stroke();
        ctx.setLineDash([]);
      } else if (hasLiveScan) {
        const statusPoint = project(0.82, 0);
        ctx.fillStyle = themeDark ? 'rgba(15,23,42,.78)' : 'rgba(255,255,255,.84)';
        roundedRect(statusPoint.x - 76, statusPoint.y - 13, 152, 26, 13);
        ctx.fill();
        ctx.fillStyle = muted;
        ctx.font = '600 10px ui-sans-serif, system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('No valid channel detected', statusPoint.x, statusPoint.y);
      }

      const sweepX = 0.15 + ((timestamp / 1800) % 1) * 1.45;
      const sweep = project(sweepX, 0);
      const sweepLeft = project(sweepX, -0.75);
      const sweepRight = project(sweepX, 0.75);
      const sweepGradient = ctx.createLinearGradient(sweepLeft.x, 0, sweepRight.x, 0);
      sweepGradient.addColorStop(0, 'rgba(59,130,246,0)');
      sweepGradient.addColorStop(.5, themeDark ? 'rgba(96,165,250,.38)' : 'rgba(37,99,235,.24)');
      sweepGradient.addColorStop(1, 'rgba(59,130,246,0)');
      ctx.strokeStyle = sweepGradient;
      ctx.lineWidth = 1.3;
      ctx.beginPath();
      ctx.moveTo(sweepLeft.x, sweep.y);
      ctx.lineTo(sweepRight.x, sweep.y);
      ctx.stroke();

      const vehicleFrontLeft = project(0.22, -0.20);
      const vehicleRearRight = project(-0.40, 0.20);
      const vehicleX = vehicleFrontLeft.x;
      const vehicleY = vehicleFrontLeft.y;
      const vehicleW = vehicleRearRight.x - vehicleFrontLeft.x;
      const vehicleH = vehicleRearRight.y - vehicleFrontLeft.y;
      roundedRect(vehicleX, vehicleY, vehicleW, vehicleH, 7);
      ctx.fillStyle = themeDark ? '#17263b' : '#f8fafc';
      ctx.fill();
      ctx.strokeStyle = themeDark ? '#dbeafe' : '#334155';
      ctx.lineWidth = 1.8;
      ctx.stroke();
      const vehicleCenter = project(-0.09, 0);
      const vehicleNose = project(0.17, 0);
      ctx.strokeStyle = '#0ea5e9';
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      ctx.moveTo(vehicleCenter.x, vehicleCenter.y);
      ctx.lineTo(vehicleNose.x, vehicleNose.y);
      ctx.stroke();
      ctx.fillStyle = '#0ea5e9';
      ctx.beginPath();
      ctx.moveTo(vehicleNose.x, vehicleNose.y - 5);
      ctx.lineTo(vehicleNose.x - 4, vehicleNose.y + 3);
      ctx.lineTo(vehicleNose.x + 4, vehicleNose.y + 3);
      ctx.closePath();
      ctx.fill();

      const lidar = project(0.035, 0);
      ctx.fillStyle = '#22d3ee';
      ctx.beginPath();
      ctx.arc(lidar.x, lidar.y, 3.4, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = themeDark ? 'rgba(34,211,238,.22)' : 'rgba(8,145,178,.18)';
      ctx.beginPath();
      ctx.arc(lidar.x, lidar.y, 7, 0, Math.PI * 2);
      ctx.stroke();

      if (renderChannel) {
        const lookahead = project(0.60, centerY(0.60));
        const pulse = 4.2 + 1.2 * (0.5 + 0.5 * Math.sin(timestamp / 320));
        ctx.fillStyle = panel;
        ctx.strokeStyle = '#e11d48';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(lookahead.x, lookahead.y, pulse, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = text;
        ctx.font = '600 10px ui-sans-serif, system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('0.60 m lookahead', lookahead.x + 9, lookahead.y - 8);
      }
      ctx.fillStyle = muted;
      ctx.font = '9.5px ui-sans-serif, system-ui, sans-serif';
      ctx.fillText('vehicle 0.40 × 0.62 m', vehicleRearRight.x + 8, vehicleRearRight.y - 8);

      if (renderChannel) {
        const widthX = 1.42;
        const leftWidthPoint = project(widthX, leftY(widthX));
        const rightWidthPoint = project(widthX, rightY(widthX));
        ctx.strokeStyle = themeDark ? 'rgba(226,232,240,.56)' : 'rgba(51,65,85,.46)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(leftWidthPoint.x, leftWidthPoint.y);
        ctx.lineTo(rightWidthPoint.x, rightWidthPoint.y);
        ctx.stroke();
        ctx.fillStyle = muted;
        ctx.textAlign = 'center';
        ctx.fillText(
          `${Number(geometry.row_width_m || 0).toFixed(2)} m row`,
          (leftWidthPoint.x + rightWidthPoint.x) / 2,
          leftWidthPoint.y - 9,
        );
      }
      window.requestAnimationFrame(drawLidarExample);
    }

    for (const id of editableFieldIds) markFieldDirty(id);
    const consoleBox = document.getElementById('console');
    if (consoleBox) {
      consoleBox.addEventListener('scroll', () => {
        const nearBottom = (consoleBox.scrollHeight - consoleBox.scrollTop - consoleBox.clientHeight) < 24;
        consoleAutoFollow = nearBottom;
      });
    }
    window.addEventListener('gamepadconnected', event => {
      if (!isXboxGamepad(event.gamepad)) return;
      browserGamepad.index = event.gamepad.index;
      browserGamepad.autoArmNeedsRtRelease = true;
      browserGamepad.nextAutoClaimAt = 0;
      showToast('Xbox controller detected. Connecting automatically.');
    });
    window.addEventListener('gamepaddisconnected', event => {
      if (browserGamepad.index === event.gamepad.index) browserGamepad.index = null;
      browserGamepad.deadman = false;
      browserGamepad.driveAxis = 0;
      browserGamepad.steerAxis = 0;
      browserGamepad.lastSendAt = 0;
      if (isXboxGamepad(event.gamepad)) {
        releaseBrowserGamepad('Xbox controller disconnected');
        showToast('Xbox controller disconnected. Vehicle stopping now.', 'error');
      }
    });
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) releaseBrowserGamepadBeacon('Page hidden');
    });
    window.addEventListener('pagehide', () => releaseBrowserGamepadBeacon('Page closed'));
    if (!('getGamepads' in navigator)) {
      showToast('This browser cannot read the Xbox controller. Use the latest Chrome or Edge.', 'error');
    }
    setTheme(localStorage.getItem('autorun_final_theme') || defaultTheme);
    setInterval(refreshState, 1000);
    setInterval(refreshLidarPreview, 200);
    setInterval(refreshPreview, 250);
    window.requestAnimationFrame(pollBrowserGamepad);
    window.requestAnimationFrame(drawLidarExample);
    refreshState();
    refreshLidarPreview();
    refreshPreview();
  </script>
</body>
</html>
"""


class RosLaserScanMonitor:
    def __init__(
        self,
        sink: queue.Queue[tuple[str, Any]],
        topic: str = LIDAR_SCAN_TOPIC,
    ) -> None:
        self.sink = sink
        self.topic = topic
        self.calibration = load_lidar_preview_calibration()
        self.running = False
        self.subscription: Any = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.last_emit = 0.0
        self.scan_times: deque[float] = deque(maxlen=40)
        self.scan_hz = 0.0
        self.min_emit_interval = 0.12
        self.last_good_row_width = 0.60
        self.row_cfg = RowFollowerConfig(
            row_width=0.60,
            min_row_width=0.48,
            max_row_width=0.78,
            lookahead_x=0.60,
            forward_min=0.15,
            forward_max=1.60,
            lateral_limit=0.75,
            range_min=0.05,
            range_max=6.0,
            bin_size=0.20,
            min_points=16,
            min_bins=2,
            min_line_bins=4,
            min_side_points_per_bin=2,
            center_deadband=0.03,
            left_percentile=20.0,
            right_percentile=80.0,
            sensor_yaw_deg=180.0,
            lidar_yaw_correction_deg=self.calibration["lidar_yaw_correction_deg"],
            lidar_x_offset_m=self.calibration["lidar_x_offset_m"],
            lidar_y_offset_m=self.calibration["lidar_y_offset_m"],
            boundary_max_gap_x=0.45,
            boundary_width_tolerance_m=0.0,
            vehicle_half_width=0.20,
            safety_margin=0.03,
            center_jump_reject=0.25,
            one_side_center_jump_reject=0.30,
        )

    def start(self) -> None:
        self.running = True
        self.thread.start()

    def stop(self) -> None:
        self.running = False

    def _run(self) -> None:
        try:
            ros_thread = ensure_ros_monitor_node()
            self.subscription = ros_thread.node.create_subscription(
                RosLaserScan,
                self.topic,
                self._on_scan,
                10,
            )
            self.sink.put(("log", f"{now_text()} Lidar preview monitor subscribed to {self.topic}."))
            self.sink.put(("lidar_status", f"Waiting for {self.topic}"))
            while self.running:
                time.sleep(0.1)
        except Exception as exc:
            self.sink.put(("log", f"{now_text()} Lidar preview monitor failed: {exc}"))
            self.sink.put(("lidar_status", "Monitor failed"))
        finally:
            self.sink.put(("lidar_status", "Monitor stopped"))

    def _on_scan(self, msg: RosLaserScan) -> None:
        if not self.running:
            return
        now = time.monotonic()
        self.scan_times.append(now)
        if len(self.scan_times) >= 2:
            elapsed = self.scan_times[-1] - self.scan_times[0]
            if elapsed > 1e-4:
                self.scan_hz = (len(self.scan_times) - 1) / elapsed
        if now - self.last_emit < self.min_emit_interval:
            return
        self.last_emit = now

        yaw_correction_deg = self.calibration["lidar_yaw_correction_deg"]
        sensor_yaw = math.radians(180.0 + yaw_correction_deg)
        cos_yaw = math.cos(sensor_yaw)
        sin_yaw = math.sin(sensor_yaw)
        offset_x = self.calibration["lidar_x_offset_m"]
        offset_y = self.calibration["lidar_y_offset_m"]
        min_range = max(float(msg.range_min), 0.05)
        max_range = min(float(msg.range_max), 6.0)
        visible_points: list[list[float]] = []
        valid_count = 0
        for index, raw_range in enumerate(msg.ranges):
            distance = float(raw_range)
            if not math.isfinite(distance) or distance < min_range or distance > max_range:
                continue
            valid_count += 1
            angle = float(msg.angle_min) + index * float(msg.angle_increment)
            sensor_x = distance * math.cos(angle)
            sensor_y = distance * math.sin(angle)
            body_x = cos_yaw * sensor_x - sin_yaw * sensor_y + offset_x
            body_y = sin_yaw * sensor_x + cos_yaw * sensor_y + offset_y
            if -0.55 <= body_x <= 1.85 and -2.55 <= body_y <= 2.55:
                visible_points.append([round(body_x, 4), round(body_y, 4)])

        max_points = 720
        if len(visible_points) > max_points:
            step = max(1, math.ceil(len(visible_points) / max_points))
            visible_points = visible_points[::step][:max_points]

        geometry: dict[str, Any]
        try:
            estimate, debug = estimate_row(msg, self.row_cfg, self.last_good_row_width)

            def point_list(values: Any, limit: int = 120) -> list[list[float]]:
                array = np.asarray(values, dtype=np.float64)
                if array.ndim != 2 or array.shape[1] < 2 or len(array) == 0:
                    return []
                if len(array) > limit:
                    stride = max(1, math.ceil(len(array) / limit))
                    array = array[::stride][:limit]
                return [
                    [round(float(point[0]), 4), round(float(point[1]), 4)]
                    for point in array
                    if math.isfinite(float(point[0])) and math.isfinite(float(point[1]))
                ]

            def line_value(value: Any) -> list[float] | None:
                if value is None or len(value) < 2:
                    return None
                slope = float(value[0])
                intercept = float(value[1])
                if not math.isfinite(slope) or not math.isfinite(intercept):
                    return None
                return [round(slope, 6), round(intercept, 6)]

            left_line = line_value(estimate.left_line)
            right_line = line_value(estimate.right_line)
            center_line = line_value(estimate.center_line)
            measured_widths: list[float] = []
            if left_line is not None and right_line is not None:
                for sample_x in (0.30, 0.60, 0.90, 1.20):
                    measured_widths.append(
                        abs(
                            (left_line[0] * sample_x + left_line[1])
                            - (right_line[0] * sample_x + right_line[1])
                        )
                    )
            measured_width = (
                sum(measured_widths) / len(measured_widths)
                if measured_widths
                else 0.0
            )
            width_valid = bool(
                measured_widths
                and all(
                    self.row_cfg.min_row_width <= width <= self.row_cfg.max_row_width
                    for width in measured_widths
                )
            )
            # A visible channel requires two independently validated, parallel boundaries.
            # A one-sided/virtual estimate remains useful to the controller, but must not be
            # presented in the UI as a measured flower-pot corridor.
            geometry_found = bool(
                estimate.left_valid
                and estimate.right_valid
                and str(estimate.effective_mode or estimate.mode) == "both_sides"
                and left_line is not None
                and right_line is not None
                and center_line is not None
                and width_valid
            )
            control_accepted = bool(geometry_found and estimate.found)
            if geometry_found:
                self.last_good_row_width = measured_width
            reject_reason = str(estimate.reject_reason or "")
            if estimate.found and not geometry_found:
                reject_reason = "not_a_complete_two_side_channel"
                if not width_valid and measured_widths:
                    reject_reason = "row_width_out_of_range"
            geometry = {
                "found": geometry_found,
                "control_accepted": control_accepted,
                "algorithm_found": bool(estimate.found),
                "mode": str(estimate.mode or estimate.effective_mode or "lost"),
                "reject_reason": reject_reason,
                "warning": str(estimate.warning or ""),
                "row_width_m": round(measured_width, 4) if measured_width > 0.0 else None,
                "target_row_width_m": self.row_cfg.row_width,
                "center_y_m": (
                    round(float(estimate.center_y if control_accepted else estimate.raw_center_y), 4)
                    if geometry_found
                    else None
                ),
                "heading_deg": (
                    round(math.degrees(math.atan(float(center_line[0]))), 3)
                    if geometry_found and center_line is not None
                    else None
                ),
                "left_valid": bool(estimate.left_valid),
                "right_valid": bool(estimate.right_valid),
                "left_bins": int(estimate.left_bins),
                "right_bins": int(estimate.right_bins),
                "candidate_bins": int(estimate.candidate_bins),
                "left_line": left_line if geometry_found else None,
                "right_line": right_line if geometry_found else None,
                "center_line": center_line if geometry_found else None,
                "left_points": point_list(debug.left_points) if geometry_found else [],
                "right_points": point_list(debug.right_points) if geometry_found else [],
                "center_points": point_list(debug.center_points) if geometry_found else [],
            }
        except Exception as exc:
            geometry = {
                "found": False,
                "algorithm_found": False,
                "mode": "fit_error",
                "reject_reason": str(exc),
                "left_line": None,
                "right_line": None,
                "center_line": None,
                "left_points": [],
                "right_points": [],
                "center_points": [],
            }
        self.sink.put((
            "lidar_scan",
            {
                "topic": self.topic,
                "frame_id": str(msg.header.frame_id or "laser"),
                "points": visible_points,
                "raw_count": len(msg.ranges),
                "valid_count": valid_count,
                "visible_count": len(visible_points),
                "scan_hz": round(self.scan_hz, 1),
                "received_monotonic": now,
                "sensor_yaw_deg": 180.0,
                "yaw_correction_deg": yaw_correction_deg,
                "offset_x_m": offset_x,
                "offset_y_m": offset_y,
                "geometry": geometry,
            },
        ))


class RosChassisChargeMonitor:
    """Read the BMS charging flag from the official chassis feedback topic."""

    def __init__(self, sink: queue.Queue[tuple[str, Any]]) -> None:
        self.sink = sink
        self.running = False
        self.subscription: Any | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.running = True
        self.thread.start()

    def stop(self) -> None:
        self.running = False

    def _run(self) -> None:
        try:
            ros_thread = ensure_ros_monitor_node()
            self.subscription = ros_thread.node.create_subscription(
                ChassisInfoFb,
                "chassis_info_fb",
                self._on_feedback,
                10,
            )
            self.sink.put((
                "log",
                f"{now_text()} Charging monitor subscribed to chassis_info_fb "
                "(BMS charge flag).",
            ))
            while self.running:
                time.sleep(0.1)
        except Exception as exc:
            self.sink.put(("log", f"{now_text()} Charging monitor failed: {exc}"))

    def _on_feedback(self, msg: ChassisInfoFb) -> None:
        if not self.running:
            return
        bms = getattr(msg, "bms_flag_fb", None)
        io_fb = getattr(msg, "io_fb", None)
        self.sink.put((
            "charge_feedback",
            {
                "charging": (
                    bool(getattr(bms, "bms_flag_fb_charge_flag"))
                    if bms is not None and hasattr(bms, "bms_flag_fb_charge_flag")
                    else None
                ),
                "charge_dock": (
                    bool(getattr(io_fb, "io_fb_charge_state"))
                    if io_fb is not None and hasattr(io_fb, "io_fb_charge_state")
                    else None
                ),
                "received_monotonic": time.monotonic(),
            },
        ))


class WebController:
    def __init__(self) -> None:
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.lock = threading.RLock()
        self.logs: deque[str] = deque(maxlen=LOG_LIMIT)
        self.settings = dict(DEFAULT_SETTINGS)
        self.task_worker: ProcessWorker | None = None
        self.record_localization_worker: ProcessWorker | None = None
        self.replay_localization_worker: ProcessWorker | None = None
        self.preview_worker: ProcessWorker | None = None
        self.preview_restart_at = 0.0
        self.lidar_worker: ProcessWorker | None = None
        self.lidar_restart_at = 0.0
        self.lidar_driver_probe_at = 0.0
        self.camera_monitor: RosImageMonitor | None = None
        self.pose_debug_monitor: RosPoseDebugMonitor | None = None
        self.lidar_monitor: RosLaserScanMonitor | None = None
        self.chassis_charge_monitor: RosChassisChargeMonitor | None = None
        self.camera_status = "Stopped"
        self.lidar_status = "Starting..."
        self.can_status = "Unknown"
        self.preview_source = "Waiting for preview stream"
        self.localization_status = "Not started"
        self.task_status = "Idle"
        self.localization_map_path: Path | None = None
        self.pending_action: str | None = None
        self.latest_preview_jpeg: bytes | None = None
        self.latest_camera_frame_at = 0.0
        self.latest_lidar_scan: dict[str, Any] = {}
        self.vehicle_status: dict[str, Any] = {}
        self.charge_feedback: dict[str, Any] = {}
        self.gamepad_control: dict[str, Any] = {
            "enabled": False,
            "owner_client_id": "",
            "browser_connected": False,
            "device_name": "",
            "deadman": False,
            "gear": "4t4d",
            "speed_mode": "low",
            "drive_axis": 0.0,
            "steer_axis": 0.0,
            "last_packet_at": 0.0,
            "status": "Disabled",
            "blocked_reason": "",
            "control_source": "idle",
            "control_label": "Idle",
            "final_vx": 0.0,
            "final_vy": 0.0,
            "final_wz_deg": 0.0,
            "final_crab_angle_deg": 0.0,
        }
        self._gamepad_output_active = False
        self._gamepad_last_published_gear = ""
        self.map_paths: dict[str, Path] = {}
        self.mission_paths: dict[str, Path] = {}
        self.library_mission_paths: dict[str, Path] = {}
        self.selected_record_map_id = ""
        self.selected_replay_map_id = ""
        self.selected_mission_id = ""
        self.selected_library_map_id = ""
        self.selected_library_mission_id = ""
        self.mapping_name = generated_name("map")
        self.mission_name = generated_name("mission")
        self.anchor_pose_debug: dict[str, float] | None = None
        self.last_raw_pose_debug: dict[str, float] | None = None
        self.pose_debug_state: dict[str, str] = {
            "monitor_status": "Starting...",
            "raw_xy": "--",
            "raw_rp": "--",
            "anchor": "Not set",
            "proj_xy": "--",
            "proj_delta": "--",
            "proj_rpy": "--",
        }
        self.closing = False
        self._load_settings()
        self._refresh_maps()
        self._refresh_missions()
        self.can_status = self._query_can_state(str(self.settings.get("can_channel") or "can0"))
        self._log("Web UI is ready.")
        self._auto_connect_can()
        self._start_pose_debug_monitor()
        self.event_thread = threading.Thread(target=self._pump_events, daemon=True)
        self.event_thread.start()
        self._start_chassis_charge_monitor()
        self._start_lidar_monitor()
        self._start_lidar_preview_driver(auto=True)
        # Subscribe first. An already-running publisher may own /dev/video0 and
        # provide the preview topic; only start our own publisher if no frames arrive.
        self._start_camera_monitor()
        self.preview_restart_at = time.monotonic() + 1.5
        self.status_thread = threading.Thread(target=self._poll_vehicle_status, daemon=True)
        self.status_thread.start()
        self.gamepad_thread = threading.Thread(target=self._gamepad_control_loop, daemon=True)
        self.gamepad_thread.start()

    @staticmethod
    def _clamp_gamepad_axis(value: Any) -> float:
        try:
            number = float(value)
        except Exception:
            return 0.0
        if not math.isfinite(number):
            return 0.0
        return max(-1.0, min(1.0, number))

    def claim_gamepad_control(self, payload: dict[str, Any]) -> dict[str, Any]:
        client_id = str(payload.get("client_id") or "").strip()
        if not client_id:
            raise RuntimeError("Missing browser gamepad client ID.")
        with self.lock:
            owner = str(self.gamepad_control.get("owner_client_id") or "")
            last_packet_at = float(self.gamepad_control.get("last_packet_at") or 0.0)
            owner_is_fresh = (time.monotonic() - last_packet_at) <= 2.0
            if bool(self.gamepad_control.get("enabled")) and owner and owner != client_id and owner_is_fresh:
                raise RuntimeError("Web gamepad control is already active in another browser.")
            if self.task_status == "Hybrid Drive":
                raise RuntimeError("Stop Hybrid Drive before enabling the web gamepad.")
            self.gamepad_control.update(
                {
                    "enabled": True,
                    "owner_client_id": client_id,
                    "browser_connected": bool(payload.get("connected", False)),
                    "device_name": str(payload.get("device_name") or "")[:160],
                    "deadman": False,
                    "drive_axis": 0.0,
                    "steer_axis": 0.0,
                    "last_packet_at": time.monotonic(),
                    "status": "Waiting for controller input",
                    "blocked_reason": "",
                }
            )
            self._log("Web gamepad control enabled. Hold RT to move.")
            return self._gamepad_state_locked()

    def update_gamepad_control(self, payload: dict[str, Any]) -> dict[str, Any]:
        client_id = str(payload.get("client_id") or "").strip()
        with self.lock:
            if not bool(self.gamepad_control.get("enabled")):
                raise RuntimeError("Web gamepad control is not enabled.")
            if client_id != str(self.gamepad_control.get("owner_client_id") or ""):
                raise RuntimeError("This browser does not own web gamepad control.")
            gear = str(payload.get("gear") or self.gamepad_control.get("gear") or "4t4d").strip().lower()
            if gear not in {"4t4d", "crab", "park", "neutral"}:
                gear = "4t4d"
            speed_mode = str(payload.get("speed_mode") or "low").strip().lower()
            if speed_mode not in GAMEPAD_SPEED_LIMITS:
                speed_mode = "low"
            connected = bool(payload.get("connected", False))
            self.gamepad_control.update(
                {
                    "browser_connected": connected,
                    "device_name": str(payload.get("device_name") or "")[:160],
                    "deadman": connected and bool(payload.get("deadman", False)),
                    "gear": gear,
                    "speed_mode": speed_mode,
                    "drive_axis": self._clamp_gamepad_axis(payload.get("drive_axis", 0.0)),
                    "steer_axis": self._clamp_gamepad_axis(payload.get("steer_axis", 0.0)),
                    "last_packet_at": time.monotonic(),
                }
            )
            return self._gamepad_state_locked()

    def release_gamepad_control(self, payload: dict[str, Any] | None = None, *, reason: str = "Released") -> None:
        client_id = str((payload or {}).get("client_id") or "").strip()
        reason = str(reason or "Released")[:160]
        with self.lock:
            owner = str(self.gamepad_control.get("owner_client_id") or "")
            if client_id and owner and client_id != owner:
                raise RuntimeError("This browser does not own web gamepad control.")
            was_enabled = bool(self.gamepad_control.get("enabled"))
            self.gamepad_control.update(
                {
                    "enabled": False,
                    "owner_client_id": "",
                    "browser_connected": False,
                    "deadman": False,
                    "drive_axis": 0.0,
                    "steer_axis": 0.0,
                    "status": reason,
                    "blocked_reason": "",
                    "final_vx": 0.0,
                    "final_vy": 0.0,
                    "final_wz_deg": 0.0,
                    "final_crab_angle_deg": 0.0,
                }
            )
            if was_enabled:
                self._publish_gamepad_stop_locked()
                self._log(f"Web gamepad control released: {reason}.")

    def _publish_gamepad_stop_locked(self) -> None:
        io_state = self.vehicle_status.get("io", {}) if isinstance(self.vehicle_status, dict) else {}
        if bool(io_state.get("remote_control", False)) or bool(io_state.get("estop", False)):
            self._gamepad_output_active = False
            return
        gear = str(self.gamepad_control.get("gear") or "4t4d")
        if gear not in {"4t4d", "crab"}:
            gear = "neutral"
        bridge = get_bridge()
        if gear == "crab":
            angle = self._clamp_gamepad_axis(
                float(self.gamepad_control.get("final_crab_angle_deg") or 0.0) / 90.0
            ) * 90.0
            bridge.publish_steering("crab", 0.0, angle)
        else:
            bridge.publish_body(gear, 0.0, 0.0, 0.0)
        bridge.publish_io(unlock=False, brake=True)
        self._gamepad_output_active = False
        self._gamepad_last_published_gear = gear

    def _gamepad_state_locked(self) -> dict[str, Any]:
        now = time.monotonic()
        packet_at = float(self.gamepad_control.get("last_packet_at") or 0.0)
        packet_age_ms = None if packet_at <= 0.0 else max(0, int((now - packet_at) * 1000.0))
        state = dict(self.gamepad_control)
        state["packet_age_ms"] = packet_age_ms
        state["command_timeout_ms"] = int(GAMEPAD_COMMAND_TIMEOUT_S * 1000.0)
        return state

    def _gamepad_control_loop(self) -> None:
        bridge = get_bridge()
        while not self.closing:
            with self.lock:
                now = time.monotonic()
                state = self.gamepad_control
                io_state = self.vehicle_status.get("io", {}) if isinstance(self.vehicle_status, dict) else {}
                physical_remote = bool(io_state.get("remote_control", False))
                estop = bool(io_state.get("estop", False))
                enabled = bool(state.get("enabled", False))
                connected = bool(state.get("browser_connected", False))
                packet_at = float(state.get("last_packet_at") or 0.0)
                fresh = packet_at > 0.0 and (now - packet_at) <= GAMEPAD_COMMAND_TIMEOUT_S
                automatic_drive = self.task_status == "Hybrid Drive"
                blocked_reason = ""
                if physical_remote:
                    source, label = "remote_controller", "Remote control"
                    blocked_reason = "Physical remote controller has priority"
                elif automatic_drive:
                    source, label = "automatic", "Automatic"
                    blocked_reason = "Hybrid Drive is running"
                elif enabled:
                    source, label = "browser_gamepad", "Gamepad"
                    if estop:
                        blocked_reason = "Emergency stop is active"
                    elif not connected:
                        blocked_reason = "Controller is not detected by the browser"
                    elif not fresh:
                        blocked_reason = "Browser command timed out"
                else:
                    source, label = "idle", "Idle"

                gear = str(state.get("gear") or "4t4d")
                deadman = bool(state.get("deadman", False))
                can_publish = enabled and connected and fresh and not physical_remote and not estop and not automatic_drive
                command_active = can_publish and deadman and gear in {"4t4d", "crab"}
                speed_mode = str(state.get("speed_mode") or "low")
                linear_limit, yaw_limit_deg = GAMEPAD_SPEED_LIMITS.get(speed_mode, GAMEPAD_SPEED_LIMITS["low"])
                drive_axis = self._clamp_gamepad_axis(state.get("drive_axis", 0.0))
                steer_axis = self._clamp_gamepad_axis(state.get("steer_axis", 0.0))
                drive_speed = drive_axis * linear_limit if command_active else 0.0
                vx = drive_speed if gear == "4t4d" else 0.0
                vy = 0.0
                wz_deg = steer_axis * yaw_limit_deg if command_active and gear == "4t4d" else 0.0
                crab_angle_deg = steer_axis * 90.0 if gear == "crab" else 0.0

                state["control_source"] = source
                state["control_label"] = label
                state["blocked_reason"] = blocked_reason
                state["final_vx"] = round(vx, 3)
                state["final_vy"] = round(vy, 3)
                state["final_wz_deg"] = round(wz_deg, 2)
                state["final_crab_angle_deg"] = round(crab_angle_deg, 2)
                if command_active:
                    active_speed = drive_speed if gear == "crab" else vx
                    state["status"] = "Driving" if abs(active_speed) > 1e-4 else "Ready"
                    if gear == "crab":
                        bridge.publish_steering("crab", drive_speed, crab_angle_deg)
                    else:
                        bridge.publish_body(gear, vx, 0.0, math.radians(wz_deg))
                    bridge.publish_io(unlock=True, brake=False)
                    self._gamepad_output_active = True
                    self._gamepad_last_published_gear = gear
                else:
                    if enabled and blocked_reason:
                        state["status"] = blocked_reason
                    elif enabled and connected and fresh:
                        state["status"] = "Ready - hold RT to move"
                    elif not enabled:
                        state["status"] = "Disabled"
                    should_stop_previous_motion = (
                        self._gamepad_output_active
                        and not physical_remote
                        and not estop
                        and not automatic_drive
                    )
                    should_send_zero = should_stop_previous_motion or (
                        can_publish
                        and gear in {"4t4d", "crab", "park", "neutral"}
                        and gear != self._gamepad_last_published_gear
                    )
                    if should_send_zero:
                        if gear == "crab":
                            bridge.publish_steering("crab", 0.0, crab_angle_deg)
                        else:
                            bridge.publish_body(gear, 0.0, 0.0, 0.0)
                        bridge.publish_io(unlock=False, brake=True)
                        self._gamepad_last_published_gear = gear
                    self._gamepad_output_active = False
            time.sleep(GAMEPAD_CONTROL_PERIOD_S)

    def _load_settings(self) -> None:
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return
        if isinstance(data, dict):
            self.settings.update(data)

    def save_settings(self, updates: dict[str, Any]) -> None:
        with self.lock:
            self.settings.update(updates)
            SETTINGS_PATH.write_text(json.dumps(self.settings, ensure_ascii=True, indent=2), encoding="utf-8")
            self.can_status = self._query_can_state(str(self.settings.get("can_channel") or "can0"))
            self._log("Settings saved.")

    def _query_can_state(self, channel: str) -> str:
        channel = str(channel or "").strip() or "can0"
        try:
            result = subprocess.run(
                ["ip", "-details", "link", "show", channel],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception:
            return "Unavailable"
        text = result.stdout
        flags_match = re.search(r"<([^>]*)>", text)
        if flags_match and "UP" in {flag.strip().upper() for flag in flags_match.group(1).split(",")}:
            return "UP"
        match = re.search(r"state\s+([A-Z]+)", text)
        if match:
            return match.group(1)
        return "UNKNOWN"

    def _auto_connect_can(self) -> None:
        channel = str(self.settings.get("can_channel") or "can0").strip() or "can0"
        bitrate = str(self.settings.get("can_bitrate") or "500000").strip() or "500000"
        current_state = self._query_can_state(channel)
        if current_state == "UP":
            self.can_status = current_state
            self._log(f"CAN {channel} is already up; automatic connection reused it.")
            return
        self._log(
            f"CAN {channel} is {current_state}. Automatically connecting at {bitrate} bps."
        )
        try:
            self.connect_can(channel, bitrate)
        except Exception as exc:
            self.can_status = self._query_can_state(channel)
            self._log(f"Automatic CAN connection failed: {exc}")

    def connect_can(self, channel: str, bitrate: str | int) -> None:
        with self.lock:
            channel = str(channel or self.settings.get("can_channel") or "can0").strip() or "can0"
            try:
                bitrate_value = int(str(bitrate or self.settings.get("can_bitrate") or "500000").strip())
            except Exception as exc:
                raise RuntimeError(f"Invalid CAN bitrate: {bitrate}") from exc

            cmds = [
                ["ip", "link", "set", channel, "down"],
                ["ip", "link", "set", channel, "type", "can", "bitrate", str(bitrate_value)],
                ["ip", "link", "set", channel, "up"],
            ]
            last_error = ""
            for cmd in cmds:
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
                if result.returncode != 0:
                    sudo_cmd = ["sudo", "-n", *cmd]
                    result = subprocess.run(sudo_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
                if result.returncode != 0:
                    last_error = (result.stderr or result.stdout or "").strip()
                    raise RuntimeError(last_error or f"Failed to run: {' '.join(cmd)}")

            self.settings["can_channel"] = channel
            self.settings["can_bitrate"] = str(bitrate_value)
            SETTINGS_PATH.write_text(json.dumps(self.settings, ensure_ascii=True, indent=2), encoding="utf-8")
            self.can_status = self._query_can_state(channel)
            self._log(f"CAN connected on {channel} at {bitrate_value} bps.")

    def _log(self, text: str) -> None:
        line = f"{now_text()} {text}"
        self.logs.append(line)

    def _odin_usb_attached(self) -> bool:
        usb_root = Path("/sys/bus/usb/devices")
        try:
            entries = list(usb_root.iterdir())
        except Exception:
            return False
        for entry in entries:
            vendor_path = entry / "idVendor"
            product_path = entry / "idProduct"
            try:
                vendor = vendor_path.read_text(encoding="utf-8").strip().lower()
                product = product_path.read_text(encoding="utf-8").strip().lower()
            except Exception:
                continue
            if vendor == ODIN_USB_VENDOR and product == ODIN_USB_PRODUCT:
                return True
        return False

    def _wait_for_odin_usb(self, timeout_sec: float = 8.0, settle_sec: float = 1.2) -> bool:
        deadline = time.monotonic() + timeout_sec
        seen = False
        last_log = 0.0
        while time.monotonic() < deadline:
            attached = self._odin_usb_attached()
            now = time.monotonic()
            if attached:
                if not seen:
                    seen = True
                    ready_at = now + settle_sec
                if now >= ready_at:
                    self._log("Odin USB device is attached and stable.")
                    return True
            else:
                seen = False
                if now - last_log >= 1.5:
                    self._log("Waiting for Odin USB device 2207:0019 to attach before localization starts.")
                    last_log = now
            time.sleep(0.2)
        self._log(
            "Odin USB device 2207:0019 did not re-attach before the wait timeout. "
            "Proceeding with localization startup anyway."
        )
        return False

    def _localization_process_groups(self) -> list[tuple[int, str]]:
        try:
            result = subprocess.run(
                ["ps", "-eo", "pid=,pgid=,args="],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception:
            return []

        patterns = (
            "main.py localization",
            "host_sdk_sample",
            "odin1_ros2.launch.py",
            "pcd2depth_ros2_node",
            "cloud_reprojection_ros2_node",
            "image_overlay_node",
        )
        current_pid = os.getpid()
        groups: dict[int, str] = {}
        for raw_line in result.stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(None, 2)
            if len(parts) < 3:
                continue
            try:
                pid = int(parts[0])
                pgid = int(parts[1])
            except ValueError:
                continue
            cmd = parts[2]
            if pid == current_pid or pgid <= 0:
                continue
            if "autorunlida" not in cmd and "shadow_ros2_ws" not in cmd:
                continue
            if any(pattern in cmd for pattern in patterns):
                groups.setdefault(pgid, cmd)
        return sorted(groups.items())

    def _cleanup_localization_processes(self, reason: str) -> None:
        groups = self._localization_process_groups()
        if not groups:
            return
        self._log(
            f"Cleaning up lingering localization processes ({reason}): "
            + ", ".join(f"pgid={pgid}" for pgid, _cmd in groups)
        )
        for pgid, _cmd in groups:
            try:
                os.killpg(pgid, signal.SIGTERM)
            except Exception:
                pass
        time.sleep(0.6)

        remaining = {pgid for pgid, _cmd in self._localization_process_groups()}
        if not remaining:
            return
        self._log(
            "Localization process groups still alive after SIGTERM; sending SIGKILL: "
            + ", ".join(str(pgid) for pgid in sorted(remaining))
        )
        for pgid in sorted(remaining):
            try:
                os.killpg(pgid, signal.SIGKILL)
            except Exception:
                pass

    def _cleanup_empty_odin_configs(self) -> None:
        removed = 0
        config_dir = PROJECT_ROOT / "runtime" / "configs"
        try:
            candidates = list(config_dir.glob("odin_mode_*.yaml"))
        except Exception:
            return
        for path in candidates:
            try:
                if path.is_file() and path.stat().st_size <= 0:
                    path.unlink()
                    removed += 1
            except Exception:
                continue
        if removed:
            self._log(f"Removed {removed} empty Odin runtime config file(s) before localization startup.")

    def _start_pose_debug_monitor(self) -> None:
        if self.pose_debug_monitor is not None:
            return
        self.pose_debug_monitor = RosPoseDebugMonitor(self.events)
        self.pose_debug_monitor.start()
        self.pose_debug_state["monitor_status"] = "Subscribed"

    def _update_pose_debug_state(self, raw_pose: dict[str, float]) -> None:
        self.last_raw_pose_debug = dict(raw_pose)
        self.pose_debug_state["monitor_status"] = "Live"
        self.pose_debug_state["raw_xy"] = f"x={raw_pose['x']:.3f}, y={raw_pose['y']:.3f}, z={raw_pose['z']:.3f}"
        self.pose_debug_state["raw_rp"] = (
            f"roll={math.degrees(raw_pose['roll']):.1f}deg, "
            f"pitch={math.degrees(raw_pose['pitch']):.1f}deg, "
            f"yaw={math.degrees(raw_pose['yaw']):.1f}deg"
        )
        if self.anchor_pose_debug is None:
            self.pose_debug_state["anchor"] = "Not set"
            self.pose_debug_state["proj_xy"] = "--"
            self.pose_debug_state["proj_delta"] = "--"
            self.pose_debug_state["proj_rpy"] = "--"
            return
        try:
            sensor_height = float(self.settings.get("sensor_height_m") or DEFAULT_SENSOR_HEIGHT_M)
            body_x = float(self.settings.get("body_x_offset_m") or DEFAULT_BODY_X_OFFSET_M)
            body_y = float(self.settings.get("body_y_offset_m") or DEFAULT_BODY_Y_OFFSET_M)
            roll_gain = float(self.settings.get("roll_gain") or DEFAULT_ROLL_GAIN)
            pitch_gain = float(self.settings.get("pitch_gain") or DEFAULT_PITCH_GAIN)
        except Exception:
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
        self.pose_debug_state["anchor"] = (
            f"x={self.anchor_pose_debug['x']:.3f}, y={self.anchor_pose_debug['y']:.3f}, "
            f"roll={math.degrees(self.anchor_pose_debug['roll']):.1f}deg, "
            f"pitch={math.degrees(self.anchor_pose_debug['pitch']):.1f}deg"
        )
        self.pose_debug_state["proj_xy"] = f"x={projected['x']:.3f}, y={projected['y']:.3f}, z={projected['z']:.3f}"
        self.pose_debug_state["proj_delta"] = f"dx={dx:+.3f}, dy={dy:+.3f}"
        self.pose_debug_state["proj_rpy"] = (
            f"roll={math.degrees(projected['roll']):.1f}deg, "
            f"pitch={math.degrees(projected['pitch']):.1f}deg, "
            f"yaw={math.degrees(projected['yaw']):.1f}deg"
        )

    def capture_pose_anchor(self) -> None:
        with self.lock:
            if self.last_raw_pose_debug is None:
                raise RuntimeError("No pose is available yet for anchor capture.")
            self.anchor_pose_debug = dict(self.last_raw_pose_debug)
            self._update_pose_debug_state(self.last_raw_pose_debug)
            self._log("Current pose captured as the ground-projection anchor.")

    def _refresh_maps(self) -> None:
        self._prune_empty_map_dirs()
        maps = sorted(MAPDATA_DIR.rglob("*.bin"), key=lambda p: p.stat().st_mtime, reverse=True)
        self.map_paths = {str(path): path for path in maps}
        if self.selected_record_map_id not in self.map_paths:
            self.selected_record_map_id = next(iter(self.map_paths), "")
        if self.selected_replay_map_id not in self.map_paths:
            self.selected_replay_map_id = self.selected_record_map_id
        if self.selected_library_map_id not in self.map_paths:
            self.selected_library_map_id = next(iter(self.map_paths), "")

    def _prune_empty_map_dirs(self) -> None:
        protected_dirs: set[Path] = set()
        if self.task_status == "Mapping" and self.mapping_name:
            protected_dirs.add((MAPDATA_DIR / self.mapping_name).resolve())
        for path in sorted(MAPDATA_DIR.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if not path.is_dir():
                continue
            try:
                resolved = path.resolve()
            except Exception:
                resolved = path
            if resolved in protected_dirs:
                continue
            try:
                next(path.iterdir())
            except StopIteration:
                try:
                    path.rmdir()
                except OSError:
                    pass
            except OSError:
                pass

    def _refresh_missions(self) -> None:
        selected_map = self.map_paths.get(self.selected_replay_map_id)
        library_map = self.map_paths.get(self.selected_library_map_id)
        library_missions: list[Path] = []
        if library_map is not None:
            for mission_path in missions_for_map(library_map):
                if mission_path not in library_missions:
                    library_missions.append(mission_path)
        self.library_mission_paths = {str(path): path for path in library_missions}
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
            filtered[str(mission_path)] = mission_path
        self.mission_paths = filtered
        if self.selected_mission_id not in self.mission_paths:
            self.selected_mission_id = next(iter(self.mission_paths), "")
        if self.selected_library_mission_id not in self.library_mission_paths:
            self.selected_library_mission_id = next(iter(self.library_mission_paths), "")

    def _mission_preview_payload(self, mission_path: Path | None) -> dict[str, Any] | None:
        if mission_path is None or not mission_path.exists():
            return None
        try:
            payload = json.loads(mission_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        raw_samples = payload.get("samples", [])
        points: list[dict[str, float]] = []
        for sample in raw_samples:
            pose = sample.get("pose") if isinstance(sample, dict) else None
            if not isinstance(pose, dict):
                continue
            try:
                x = float(pose.get("x"))
                y = float(pose.get("y"))
            except Exception:
                continue
            points.append({"x": x, "y": y})
        if len(points) < 2:
            return None
        if len(points) > 400:
            step = max(1, len(points) // 400)
            points = points[::step]
            if points[-1] != {"x": float(raw_samples[-1].get("pose", {}).get("x", points[-1]["x"])), "y": float(raw_samples[-1].get("pose", {}).get("y", points[-1]["y"]))}:
                try:
                    last_pose = raw_samples[-1].get("pose", {})
                    points.append({"x": float(last_pose.get("x")), "y": float(last_pose.get("y"))})
                except Exception:
                    pass
        min_x = min(p["x"] for p in points)
        max_x = max(p["x"] for p in points)
        min_y = min(p["y"] for p in points)
        max_y = max(p["y"] for p in points)
        return {
            "points": points,
            "sample_count": len(raw_samples),
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
        }

    def delete_map(self, map_id: str) -> None:
        with self.lock:
            map_path = self.map_paths.get(map_id)
            if map_path is None:
                raise RuntimeError("No map selected.")
            map_dir = map_path.parent
            if MAPDATA_DIR not in map_dir.parents and map_dir != MAPDATA_DIR:
                raise RuntimeError(f"Refusing to delete map outside workspace: {map_dir}")
            if map_dir.exists():
                shutil.rmtree(map_dir)
            if self.selected_record_map_id == map_id:
                self.selected_record_map_id = ""
            if self.selected_replay_map_id == map_id:
                self.selected_replay_map_id = ""
            if self.selected_library_map_id == map_id:
                self.selected_library_map_id = ""
            if self.localization_map_path == map_path:
                self.localization_map_path = None
            self._prune_empty_map_dirs()
            self._refresh_maps()
            self._refresh_missions()
            self._log(f"Deleted map folder: {map_dir}")

    def delete_mission(self, mission_id: str) -> None:
        with self.lock:
            mission_path = self.library_mission_paths.get(mission_id) or self.mission_paths.get(mission_id)
            if mission_path is None:
                raise RuntimeError("No mission selected.")
            resolved_mission_path = mission_path.resolve()
            allowed_parents = {MISSIONS_DIR.resolve()}
            allowed_parents.update(
                map_path.parent.resolve()
                for map_path in self.map_paths.values()
            )
            if resolved_mission_path.parent not in allowed_parents:
                raise RuntimeError(f"Refusing to delete mission outside workspace: {mission_path}")
            csv_path = resolved_mission_path.with_suffix(".csv")
            deleted_names = [resolved_mission_path.name]
            if resolved_mission_path.exists():
                resolved_mission_path.unlink()
            if csv_path.exists():
                csv_path.unlink()
                deleted_names.append(csv_path.name)
            if self.selected_mission_id == mission_id:
                self.selected_mission_id = ""
            if self.selected_library_mission_id == mission_id:
                self.selected_library_mission_id = ""
            self._refresh_missions()
            self._log(f"Deleted mission files: {', '.join(deleted_names)}")

    def select_map(self, role: str, map_id: str) -> None:
        with self.lock:
            if map_id and map_id not in self.map_paths:
                raise RuntimeError("Selected map is not available.")
            normalized_role = str(role or "").strip().lower()
            if normalized_role == "record":
                self.selected_record_map_id = map_id
            elif normalized_role == "drive":
                self.selected_replay_map_id = map_id
                self._refresh_missions()
            elif normalized_role == "library":
                self.selected_library_map_id = map_id
                self.selected_library_mission_id = ""
                self._refresh_missions()
            else:
                raise RuntimeError(f"Unknown map selection role: {role}")

    def select_mission(self, mission_id: str, role: str = "drive") -> None:
        with self.lock:
            normalized_role = str(role or "drive").strip().lower()
            if normalized_role == "library":
                if mission_id and mission_id not in self.library_mission_paths:
                    raise RuntimeError("Selected mission is not available.")
                self.selected_library_mission_id = mission_id
            else:
                if mission_id and mission_id not in self.mission_paths:
                    raise RuntimeError("Selected mission is not available.")
                self.selected_mission_id = mission_id

    def _active_localization_worker(self) -> ProcessWorker | None:
        return self.replay_localization_worker or self.record_localization_worker

    def _map_id_for_path(self, map_path: Path | None) -> str:
        if map_path is None:
            return ""
        try:
            resolved = map_path.resolve()
        except Exception:
            resolved = map_path
        for map_id, candidate in self.map_paths.items():
            try:
                if candidate.resolve() == resolved:
                    return map_id
            except Exception:
                if candidate == map_path:
                    return map_id
        return ""

    def _projection_args(self) -> list[str]:
        return [
            "--sensor-height-m", str(self.settings.get("sensor_height_m") or DEFAULT_SENSOR_HEIGHT_M),
            "--body-x-offset-m", str(self.settings.get("body_x_offset_m") or DEFAULT_BODY_X_OFFSET_M),
            "--body-y-offset-m", str(self.settings.get("body_y_offset_m") or DEFAULT_BODY_Y_OFFSET_M),
            "--roll-gain", str(self.settings.get("roll_gain") or DEFAULT_ROLL_GAIN),
            "--pitch-gain", str(self.settings.get("pitch_gain") or DEFAULT_PITCH_GAIN),
        ]

    def _hybrid_args(self) -> list[str]:
        args = [
            *self._projection_args(),
            "autorun",
            "--line-cruise-vx", str(self.settings.get("line_cruise_vx") or "0.20"),
            "--line-target-center-offset-px", str(self.settings.get("line_target_center_offset_px") or "0"),
            "--line-vehicle-direction-angle-deg", str(self.settings.get("line_vehicle_direction_angle_deg") or "0.0"),
            "--line-steer-sign", str(self.settings.get("line_steer_sign") or "-1.0"),
            "--line-kp-offset", str(self.settings.get("line_kp_offset") or "7.0"),
            "--line-kp-heading", str(self.settings.get("line_kp_heading") or "0.08"),
            "--line-max-wz", str(self.settings.get("line_max_wz") or "1.6"),
            "--lidar-yaw-correction-deg", DEFAULT_LIDAR_YAW_CORRECTION_DEG,
            "--lidar-x-offset-m", DEFAULT_LIDAR_X_OFFSET_M,
            "--lidar-y-offset-m", DEFAULT_LIDAR_Y_OFFSET_M,
        ]
        return args

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

    def _lidar_driver_running(self) -> bool:
        patterns = (str(LIDAR_DRIVER_BIN), "ros2 run lidar_pkg lidar_node")
        try:
            result = subprocess.run(
                ["ps", "-eo", "args="],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except Exception:
            return False
        for line in result.stdout.splitlines():
            command = line.strip()
            if command and any(pattern in command for pattern in patterns):
                return True
        return False

    def _lidar_driver_args(self) -> list[str]:
        if not LIDAR_DRIVER_BIN.exists():
            raise RuntimeError(f"Lidar driver binary not found: {LIDAR_DRIVER_BIN}")
        args = [str(LIDAR_DRIVER_BIN)]
        if LIDAR_DRIVER_PARAMS.exists():
            args.extend(["--ros-args", "--params-file", str(LIDAR_DRIVER_PARAMS)])
        else:
            port = "/dev/lidar" if Path("/dev/lidar").exists() else "/dev/ttyACM0"
            args.extend(["--ros-args", "-p", f"port_name:={port}", "-p", "frame_id:=laser"])
        return args

    def _start_lidar_monitor(self) -> None:
        if self.lidar_monitor is not None:
            return
        self.lidar_monitor = RosLaserScanMonitor(self.events)
        self.lidar_monitor.start()
        self.lidar_status = f"Waiting for {LIDAR_SCAN_TOPIC}"

    def _start_chassis_charge_monitor(self) -> None:
        if self.chassis_charge_monitor is not None:
            return
        self.chassis_charge_monitor = RosChassisChargeMonitor(self.events)
        self.chassis_charge_monitor.start()

    def _start_lidar_preview_driver(self, auto: bool = False) -> None:
        if self.lidar_worker is not None:
            return
        self.lidar_restart_at = 0.0
        if self._lidar_driver_running():
            self.lidar_status = "Waiting for existing lidar driver"
            if not self.latest_lidar_scan:
                self._log("Existing lidar driver detected; preview monitor will reuse /scan.")
            return
        try:
            args = self._lidar_driver_args()
        except Exception as exc:
            self.lidar_status = "Driver unavailable"
            self._log(f"Lidar preview driver unavailable: {exc}")
            self.lidar_restart_at = time.monotonic() + 5.0
            return
        worker = ProcessWorker(args, LIDAR_DRIVER_ROOT, "Lidar Preview", self.events)
        self.lidar_worker = worker
        self.lidar_status = "Driver starting"
        worker.start()
        self._log("Lidar preview driver auto-started." if auto else "Lidar preview driver started.")

    def _start_camera_monitor(self) -> None:
        if self.camera_monitor is not None:
            return
        self.camera_monitor = RosImageMonitor(self.events, topics=(UVC_PREVIEW_TOPIC,))
        self.camera_monitor.start()
        self.camera_status = "Starting..."
        self._log("UVC preview monitor started. This is only the camera preview stream, not lidar local guidance.")

    def _camera_stream_live(self, now: float | None = None) -> bool:
        current = time.monotonic() if now is None else float(now)
        return (
            self.latest_camera_frame_at > 0.0
            and (current - self.latest_camera_frame_at) <= 1.5
        )

    def _start_uvc_preview_publisher(self, auto: bool = False) -> None:
        self._start_camera_monitor()
        if self._camera_stream_live():
            self.preview_restart_at = 0.0
            self.camera_status = "Streaming"
            return
        if self.preview_worker is not None:
            return
        self.preview_restart_at = 0.0
        worker = ProcessWorker([ROS_PYTHON, *self._uvc_preview_args()], PROJECT_ROOT, "UVC Preview", self.events)
        self.preview_worker = worker
        worker.start()
        self._start_camera_monitor()
        self._log("UVC preview publisher started." if not auto else "UVC preview publisher auto-started.")

    def _clear_preview(self, text: str = "Waiting for preview stream") -> None:
        self.latest_preview_jpeg = None
        self.preview_source = text
        if self.camera_monitor is not None:
            self.camera_monitor.reset()

    def _start_task(self, label: str, args: list[str], *, slot: str = "task") -> None:
        if slot == "task" and self.task_worker is not None:
            raise RuntimeError("A task is already running.")
        worker = ProcessWorker([ROS_PYTHON, str(MAIN_SCRIPT), *args], PROJECT_ROOT, label, self.events)
        if slot == "record_localization":
            self.record_localization_worker = worker
            self.localization_status = "Starting..."
        elif slot == "replay_localization":
            self.replay_localization_worker = worker
            self.localization_status = "Starting..."
        else:
            self.task_worker = worker
            self.task_status = label
        self._log(f"Starting {label}: {' '.join(args)}")
        worker.start()

    def start_mapping(self, map_name: str) -> None:
        with self.lock:
            map_name = map_name.strip() or generated_name("map")
            self.mapping_name = map_name
            active = self._active_localization_worker()
            if active is not None:
                self.localization_status = "Interrupted by Mapping"
                active.stop()
            self._start_uvc_preview_publisher(auto=True)
            args = ["map", "--map-name", map_name, "--viz", "off"]
            if bool(self.settings.get("mapping_recorddata", False)):
                args.append("--recorddata")
            self._start_task("Mapping", args)

    def start_localization(self, map_id: str) -> None:
        with self.lock:
            map_path = self.map_paths.get(map_id)
            if map_path is None:
                raise RuntimeError("No map selected.")
            self.selected_record_map_id = map_id
            self.selected_replay_map_id = map_id
            self._start_shared_localization(map_path)

    def _start_shared_localization(self, map_path: Path, *, auto: bool = False) -> None:
        active = self._active_localization_worker()
        if active is not None and self.localization_map_path == map_path:
            return
        if active is not None:
            active.stop()
            self.record_localization_worker = None
            self.replay_localization_worker = None
        self._cleanup_localization_processes("before starting a new localization session")
        self._cleanup_empty_odin_configs()
        self._wait_for_odin_usb()
        self.localization_map_path = map_path
        self.localization_status = "Starting..."
        self._start_camera_monitor()
        self._clear_preview()
        self._start_uvc_preview_publisher(auto=True)
        worker = ProcessWorker(
            [ROS_PYTHON, str(MAIN_SCRIPT), "localization", "--db", str(map_path), "--base-frame", "odin1_base_link", "--localization-wait-sec", "60"],
            PROJECT_ROOT,
            "Shared Localization",
            self.events,
        )
        self.record_localization_worker = worker
        self.replay_localization_worker = worker
        self._log(f"{'Auto-starting' if auto else 'Starting'} shared localization with map: {map_path.name}")
        self._log("Waiting for relocalization result from Odin/TF.")
        worker.start()

    def stop_localization(self) -> None:
        with self.lock:
            active = self._active_localization_worker()
            if active is None:
                return
            self._log("Stop requested for shared localization.")
            active.stop()
            self._cleanup_localization_processes("after localization stop request")

    def start_recording(self, map_id: str, mission_name: str) -> None:
        with self.lock:
            map_path = self.map_paths.get(map_id)
            if map_path is None:
                raise RuntimeError("No map selected.")
            self.selected_record_map_id = map_id
            active = self._active_localization_worker()
            if active is None or self.localization_map_path != map_path:
                self.pending_action = "record"
                self.mission_name = mission_name.strip() or generated_name("mission")
                self._start_shared_localization(map_path)
                return
            if self.localization_status != "Ready":
                self.pending_action = "record"
                self.mission_name = mission_name.strip() or generated_name("mission")
                self._log("Path recording will start automatically after localization becomes ready.")
                return
            mission_name = mission_name.strip() or generated_name("mission")
            self.mission_name = mission_name
            self._start_uvc_preview_publisher(auto=True)
            self._start_task(
                "Path Recording",
                [*self._projection_args(), "record", "--db", str(map_path), "--mission-name", mission_name, "--base-frame", "odin1_base_link", "--localization-wait-sec", "60", "--reuse-localization"],
            )

    def start_drive(self, map_id: str, mission_id: str) -> None:
        with self.lock:
            active_map_id = self._map_id_for_path(self.localization_map_path)
            if active_map_id:
                map_id = active_map_id
            map_path = self.map_paths.get(map_id)
            mission_path = self.mission_paths.get(mission_id)
            if map_path is None:
                raise RuntimeError("No map selected.")
            if mission_path is None:
                raise RuntimeError("No mission selected.")
            self.selected_replay_map_id = map_id
            self.selected_mission_id = mission_id
            try:
                mission_payload = json.loads(mission_path.read_text(encoding="utf-8"))
                bound_map_db = str(mission_payload.get("bound_map_db") or "").strip()
            except Exception as exc:
                raise RuntimeError(f"Failed to read mission file: {exc}") from exc
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
                        raise RuntimeError("This mission belongs to a different map.")
                except RuntimeError:
                    raise
                except Exception:
                    pass
            if bool(self.gamepad_control.get("enabled", False)):
                self.release_gamepad_control(reason="Hybrid Drive requested")
            active = self._active_localization_worker()
            if active is None or self.localization_map_path != map_path:
                self.pending_action = "drive"
                self._start_shared_localization(map_path)
                return
            if self.localization_status != "Ready":
                self.pending_action = "drive"
                self._log("Hybrid drive will start automatically after localization becomes ready.")
                return
            self._start_uvc_preview_publisher(auto=True)
            self._start_camera_monitor()
            self._start_task(
                "Hybrid Drive",
                [*self._hybrid_args(), "--db", str(map_path), "--mission", str(mission_path), "--base-frame", "odin1_base_link", "--localization-wait-sec", "60", "--reuse-localization"],
            )

    def stop(self) -> None:
        with self.lock:
            if self.task_worker is not None:
                self._log("Stop requested for the current task.")
                self.task_worker.stop()
                return
            active = self._active_localization_worker()
            if active is not None:
                self._log("Stop requested for shared localization.")
                active.stop()
                self._cleanup_localization_processes("after global stop request")

    def _mark_localization_ready(self) -> None:
        if self._active_localization_worker() is None:
            return
        self.localization_status = "Ready"
        active_map_id = self._map_id_for_path(self.localization_map_path)
        if active_map_id:
            self.selected_record_map_id = active_map_id
            self.selected_replay_map_id = active_map_id
        if self.pending_action == "record":
            self.pending_action = None
            self.start_recording(self.selected_record_map_id, self.mission_name)
        elif self.pending_action == "drive":
            self.pending_action = None
            self.start_drive(self.selected_replay_map_id, self.selected_mission_id)

    def _pump_events(self) -> None:
        while not self.closing:
            try:
                event, payload = self.events.get(timeout=0.2)
            except queue.Empty:
                continue
            with self.lock:
                if event == "log":
                    line = str(payload)
                    self.logs.append(line)
                    lower_line = line.lower()
                    if (
                        ("Localization succeeded." in line or "relocalization success!" in lower_line)
                        and self._active_localization_worker() is not None
                    ):
                        self._mark_localization_ready()
                elif event == "task_finished":
                    self._on_task_finished(payload)
                elif event == "camera_frame":
                    if isinstance(payload, dict):
                        ppm = payload.get("image")
                        topic = str(payload.get("topic", ""))
                        if isinstance(ppm, (bytes, bytearray)):
                            self.latest_preview_jpeg = self._ppm_to_jpeg(bytes(ppm))
                            self.latest_camera_frame_at = time.monotonic()
                            self.camera_status = "Streaming"
                            self.preview_restart_at = 0.0
                        if topic == UVC_PREVIEW_TOPIC:
                            self.preview_source = "Raw UVC Preview"
                        elif topic:
                            self.preview_source = topic
                elif event == "camera_status":
                    status = str(payload)
                    if status == "Streaming" or not self._camera_stream_live():
                        self.camera_status = status
                elif event == "lidar_scan" and isinstance(payload, dict):
                    first_scan = not bool(self.latest_lidar_scan)
                    self.latest_lidar_scan = dict(payload)
                    self.lidar_status = "Live"
                    if first_scan:
                        self._log(
                            f"Live lidar scans are arriving from {payload.get('topic', LIDAR_SCAN_TOPIC)} "
                            f"({payload.get('visible_count', 0)} visible points)."
                        )
                elif event == "lidar_status":
                    if self.lidar_status != "Live" or not self.latest_lidar_scan:
                        self.lidar_status = str(payload)
                elif event == "charge_feedback" and isinstance(payload, dict):
                    self.charge_feedback = dict(payload)
                elif event == "pose_debug" and isinstance(payload, dict):
                    self._update_pose_debug_state(payload)

    def _on_task_finished(self, payload: dict[str, Any]) -> None:
        code = int(payload["code"])
        stopped = bool(payload["stopped"])
        label = str(payload["label"])
        worker_id = int(payload.get("worker_id", -1))
        if label == "UVC Preview":
            if self.preview_worker is not None and self.preview_worker.worker_id == worker_id:
                self.preview_worker = None
                if not self.closing and not stopped:
                    if self._camera_stream_live():
                        self.camera_status = "Streaming"
                        self.preview_source = "Raw UVC Preview"
                        self.preview_restart_at = 0.0
                    else:
                        self.camera_status = "Restarting..."
                        self.preview_source = "UVC preview restarting"
                        self.preview_restart_at = time.monotonic() + 3.0
            self._log(f"{label} {'stopped' if stopped else 'finished'} with exit code {code}")
            return
        if label == "Lidar Preview":
            if self.lidar_worker is not None and self.lidar_worker.worker_id == worker_id:
                self.lidar_worker = None
                if not self.closing:
                    self.lidar_status = "Driver restarting"
                    self.lidar_restart_at = time.monotonic() + 3.0
            self._log(f"{label} {'stopped' if stopped else 'finished'} with exit code {code}")
            return
        if label in {"Record Localization", "Replay Localization", "Shared Localization"}:
            active = self._active_localization_worker()
            if active is not None and active.worker_id == worker_id:
                self.record_localization_worker = None
                self.replay_localization_worker = None
                self.localization_status = "Stopped" if stopped else ("Ready" if code == 0 else "Failed")
            if code != 0 or stopped:
                self._cleanup_localization_processes(f"after {label} finished with code {code}")
        else:
            if self.task_worker is not None and self.task_worker.worker_id == worker_id:
                self.task_worker = None
                self.task_status = "Idle"
        if label in {"Mapping", "Path Recording", "Hybrid Drive", "Record Localization", "Replay Localization", "Shared Localization"}:
            self._refresh_maps()
            self._refresh_missions()
        if label == "Mapping":
            self.mapping_name = generated_name("map")
        if label == "Path Recording":
            self.mission_name = generated_name("mission")
        self._log(f"{label} {'stopped' if stopped else 'finished'} with exit code {code}")

    def _poll_vehicle_status(self) -> None:
        bridge = get_bridge()
        while not self.closing:
            with self.lock:
                now = time.monotonic()
                snapshot = bridge.snapshot()
                charge_received_at = float(
                    self.charge_feedback.get("received_monotonic", 0.0) or 0.0
                )
                if charge_received_at > 0.0 and (now - charge_received_at) <= 2.0:
                    battery = dict(snapshot.get("battery", {}))
                    io_state = dict(snapshot.get("io", {}))
                    battery["charging"] = self.charge_feedback.get("charging")
                    io_state["charge_dock"] = self.charge_feedback.get("charge_dock")
                    snapshot["battery"] = battery
                    snapshot["io"] = io_state
                self.vehicle_status = snapshot
                if (
                    self.preview_worker is None
                    and self.preview_restart_at > 0.0
                    and now >= self.preview_restart_at
                ):
                    if self._camera_stream_live(now):
                        self.preview_restart_at = 0.0
                        self.camera_status = "Streaming"
                    else:
                        self._start_uvc_preview_publisher(auto=True)
                if now >= self.lidar_driver_probe_at:
                    self.lidar_driver_probe_at = now + 2.0
                    last_scan_at = float(self.latest_lidar_scan.get("received_monotonic", 0.0) or 0.0)
                    scan_stale = last_scan_at <= 0.0 or (now - last_scan_at) > 2.0
                    restart_due = self.lidar_restart_at <= 0.0 or now >= self.lidar_restart_at
                    if (
                        self.lidar_worker is None
                        and scan_stale
                        and restart_due
                        and not self._lidar_driver_running()
                    ):
                        self._start_lidar_preview_driver(auto=True)
            time.sleep(0.4)

    def _ppm_to_jpeg(self, ppm: bytes) -> bytes | None:
        try:
            frame = cv2.imdecode(np.frombuffer(ppm, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                return None
            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                return None
            return encoded.tobytes()
        except Exception:
            return None

    def _lidar_preview_snapshot_locked(self) -> dict[str, Any]:
        snapshot = dict(self.latest_lidar_scan)
        received_at = float(snapshot.pop("received_monotonic", 0.0) or 0.0)
        age_ms: float | None = None
        if received_at > 0.0:
            age_ms = max(0.0, (time.monotonic() - received_at) * 1000.0)
        live = age_ms is not None and age_ms <= 1600.0
        snapshot["live"] = live
        snapshot["age_ms"] = round(age_ms, 1) if age_ms is not None else None
        snapshot["status"] = "Live" if live else self.lidar_status
        snapshot.setdefault("topic", LIDAR_SCAN_TOPIC)
        snapshot.setdefault("points", [])
        snapshot.setdefault("raw_count", 0)
        snapshot.setdefault("valid_count", 0)
        snapshot.setdefault("visible_count", 0)
        snapshot.setdefault("scan_hz", 0.0)
        return snapshot

    def state_snapshot(self) -> dict[str, Any]:
        with self.lock:
            self._refresh_maps()
            self._refresh_missions()
            selected_library_mission_preview = self._mission_preview_payload(
                self.library_mission_paths.get(self.selected_library_mission_id)
            )
            return {
                "task_status": self.task_status,
                "localization_status": self.localization_status,
                "camera_status": self.camera_status,
                "can_status": self.can_status,
                "preview_source": self.preview_source,
                "logs": list(self.logs),
                "maps": [
                    {"id": key, "label": f"{path.name}  |  {path.parent.name}"}
                    for key, path in self.map_paths.items()
                ],
                "missions": [
                    {"id": key, "label": path.stem}
                    for key, path in self.mission_paths.items()
                ],
                "library_missions": [
                    {"id": key, "label": path.stem}
                    for key, path in self.library_mission_paths.items()
                ],
                "selected_record_map_id": self.selected_record_map_id,
                "selected_replay_map_id": self.selected_replay_map_id,
                "active_localization_map_id": self._map_id_for_path(self.localization_map_path),
                "selected_mission_id": self.selected_mission_id,
                "selected_library_map_id": self.selected_library_map_id,
                "selected_library_mission_id": self.selected_library_mission_id,
                "selected_library_mission_preview": selected_library_mission_preview,
                "mapping_name": self.mapping_name,
                "mission_name": self.mission_name,
                "settings": dict(self.settings),
                "vehicle_status": getattr(self, "vehicle_status", {}),
                "gamepad_control": self._gamepad_state_locked(),
                "pose_debug": dict(self.pose_debug_state),
            }

    def preview_jpeg(self) -> bytes | None:
        with self.lock:
            return self.latest_preview_jpeg

    def lidar_preview_snapshot(self) -> dict[str, Any]:
        with self.lock:
            return self._lidar_preview_snapshot_locked()

    def close(self) -> None:
        with self.lock:
            if bool(self.gamepad_control.get("enabled", False)) or self._gamepad_output_active:
                self.release_gamepad_control(reason="Web server stopped")
            self.closing = True
        if self.task_worker is not None:
            self.task_worker.stop()
        active = self._active_localization_worker()
        if active is not None:
            active.stop()
        if self.preview_worker is not None:
            self.preview_worker.stop()
        if self.lidar_worker is not None:
            self.lidar_worker.stop()
        if self.camera_monitor is not None:
            self.camera_monitor.stop()
        if self.lidar_monitor is not None:
            self.lidar_monitor.stop()
        if self.chassis_charge_monitor is not None:
            self.chassis_charge_monitor.stop()
        if self.pose_debug_monitor is not None:
            self.pose_debug_monitor.stop()


APP = WebController()


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "autorun-final-web/0.1"

    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def _safe_write(self, body: bytes) -> bool:
        try:
            self.wfile.write(body)
            return True
        except (BrokenPipeError, ConnectionResetError):
            return False

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        data = json.loads(raw.decode("utf-8"))
        return data if isinstance(data, dict) else {}

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Permissions-Policy", "gamepad=(self)")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self._safe_write(body)

    def _send_text(self, text: str, status: int = 200, content_type: str = "text/plain; charset=utf-8") -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Permissions-Policy", "gamepad=(self)")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self._safe_write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_text(HTML_PAGE, content_type="text/html; charset=utf-8")
            return
        if parsed.path == "/api/state":
            self._send_json(APP.state_snapshot())
            return
        if parsed.path == "/api/lidar":
            self._send_json(APP.lidar_preview_snapshot())
            return
        if parsed.path == "/api/preview.jpg":
            preview = APP.preview_jpeg()
            if preview is None:
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(preview)))
            self.end_headers()
            self._safe_write(preview)
            return
        self._send_text("Not found", status=404)

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            if self.path == "/api/settings":
                APP.save_settings(payload)
                self._send_json({"ok": True})
                return
            if self.path == "/api/start_mapping":
                APP.start_mapping(str(payload.get("map_name") or ""))
                self._send_json({"ok": True})
                return
            if self.path == "/api/start_localization":
                APP.start_localization(str(payload.get("map_id") or ""))
                self._send_json({"ok": True})
                return
            if self.path == "/api/stop_localization":
                APP.stop_localization()
                self._send_json({"ok": True})
                return
            if self.path == "/api/start_recording":
                APP.start_recording(str(payload.get("map_id") or ""), str(payload.get("mission_name") or ""))
                self._send_json({"ok": True})
                return
            if self.path == "/api/start_drive":
                APP.start_drive(str(payload.get("map_id") or ""), str(payload.get("mission_id") or ""))
                self._send_json({"ok": True})
                return
            if self.path == "/api/capture_anchor":
                APP.capture_pose_anchor()
                self._send_json({"ok": True})
                return
            if self.path == "/api/connect_can":
                APP.connect_can(str(payload.get("channel") or ""), str(payload.get("bitrate") or ""))
                self._send_json({"ok": True})
                return
            if self.path == "/api/gamepad/claim":
                state = APP.claim_gamepad_control(payload)
                self._send_json({"ok": True, "gamepad_control": state})
                return
            if self.path == "/api/gamepad/command":
                state = APP.update_gamepad_control(payload)
                self._send_json({"ok": True, "gamepad_control": state})
                return
            if self.path == "/api/gamepad/release":
                APP.release_gamepad_control(payload, reason=str(payload.get("reason") or "Released"))
                self._send_json({"ok": True})
                return
            if self.path == "/api/select_map":
                APP.select_map(str(payload.get("role") or ""), str(payload.get("map_id") or ""))
                self._send_json({"ok": True})
                return
            if self.path == "/api/select_mission":
                APP.select_mission(str(payload.get("mission_id") or ""), str(payload.get("role") or "drive"))
                self._send_json({"ok": True})
                return
            if self.path == "/api/delete_map":
                APP.delete_map(str(payload.get("map_id") or ""))
                self._send_json({"ok": True})
                return
            if self.path == "/api/delete_mission":
                APP.delete_mission(str(payload.get("mission_id") or ""))
                self._send_json({"ok": True})
                return
            if self.path == "/api/stop":
                APP.stop()
                self._send_json({"ok": True})
                return
            self._send_text("Not found", status=404)
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=400)


def main() -> int:
    server = ThreadingHTTPServer((HOST, PORT), RequestHandler)

    def _detect_lan_ip() -> str:
        try:
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                probe.connect(("8.8.8.8", 80))
                return str(probe.getsockname()[0])
            finally:
                probe.close()
        except Exception:
            return "127.0.0.1"

    def _handle_signal(signum, frame) -> None:
        del signum, frame
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    lan_ip = _detect_lan_ip()
    print(f"autorun_final web UI listening on http://127.0.0.1:{PORT}", flush=True)
    print(f"autorun_final web UI LAN URL: http://{lan_ip}:{PORT}", flush=True)
    try:
        server.serve_forever()
    finally:
        APP.close()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
