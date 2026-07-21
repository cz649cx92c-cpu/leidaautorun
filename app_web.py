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
    generated_name,
    missions_for_map,
    normalize_device_path,
    now_text,
    project_ground_pose,
)
from control.official_fwmini_compat import get_bridge


HOST = "0.0.0.0"
PORT = 8765
LOG_LIMIT = 250
ODIN_USB_VENDOR = "2207"
ODIN_USB_PRODUCT = "0019"
DEFAULT_LIDAR_YAW_CORRECTION_DEG = "-3.0"
DEFAULT_LIDAR_X_OFFSET_M = "0.035"
DEFAULT_LIDAR_Y_OFFSET_M = "0.0"


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


  </style>
</head>
<body data-theme="dark">
  <div id="toastStack" class="toast-stack"></div>
  <div class="page control-root">
    <header class="topbar">
      <div class="brand-block">
        <div class="brand-row">
          <h1>autorun</h1>
          <div class="top-status-cluster">
            <div class="status-pill"><span class="status-dot"></span><span id="cameraStatus">Camera stopped</span></div>
            <div class="status-pill compact"><strong id="canStateSummary">CAN --</strong></div>
          </div>
        </div>
        <div class="summary-strip">
          <div class="summary-cell">
            <span class="summary-label">Task</span>
            <strong id="taskStatus">Idle</strong>
          </div>
          <div class="summary-cell">
            <span class="summary-label">Localization</span>
            <strong id="localizationStatus">Not started</strong>
          </div>
          <div class="summary-cell stretch">
            <span class="summary-label">Map</span>
            <strong id="selectedMapSummary">--</strong>
          </div>
          <div class="summary-cell stretch">
            <span class="summary-label">Mission</span>
            <strong id="selectedMissionSummary">--</strong>
          </div>
        </div>
      </div>
      <div class="topbar-actions">
        <button class="secondary slim" style="width:auto" onclick="toggleTheme()">Theme</button>
        <button id="canConnectTopBtn" class="secondary slim" style="width:auto" onclick="connectCan()">Connect CAN</button>
      </div>
    </header>

    <section class="tab-dock">
      <div class="segmented nav-tabs">
        <button id="tabBtn-dashboard" class="active" onclick="selectTab('dashboard')">Overview</button>
        <button id="tabBtn-tasks" onclick="selectTab('tasks')">Workflow</button>
        <button id="tabBtn-library" onclick="selectTab('library')">Library</button>
        <button id="tabBtn-settings" onclick="selectTab('settings')">Tuning</button>
      </div>
    </section>

    <section class="control-grid">
      <aside class="ops-rail">
        <div class="rail-panel panel">
          <div class="rail-title-row">
            <h2>Session</h2>
            <div class="status-badge" id="previewModeBadge">Preview</div>
          </div>
          <div class="rail-metrics">
            <div class="rail-metric">
              <span>Preview Source</span>
              <strong id="previewSource">Waiting for preview stream</strong>
            </div>
            <div class="rail-metric">
              <span>Guidance Mode</span>
              <strong>Lidar in-row / staged global</strong>
            </div>
            <div class="rail-metric">
              <span>Projection</span>
              <strong id="projectionSummary">h=0.00, x=0.00, y=0.00</strong>
            </div>
            <div class="rail-metric">
              <span>Workflow Gate</span>
              <strong id="workflowGateSummary">Waiting for map lock</strong>
            </div>
          </div>
        </div>

        <div class="rail-panel panel gate-panel">
          <div class="rail-title-row">
            <h2>Workflow Gate</h2>
          </div>
          <div class="gate-stack">
            <div class="status-line compact">
              <div>
                <strong>Relocalization</strong>
                <div class="panel-sub">Required before recording or drive.</div>
              </div>
              <div class="status-badge" id="localizationGateBadge">Waiting</div>
            </div>
            <div class="status-line compact">
              <div>
                <strong>Mission Recording</strong>
                <div class="panel-sub">Enabled after map lock.</div>
              </div>
              <div class="status-badge" id="recordingGateBadge">Locked</div>
            </div>
            <div class="status-line compact">
              <div>
                <strong>Hybrid Drive</strong>
                <div class="panel-sub">Mission + local guidance blend.</div>
              </div>
              <div class="status-badge" id="driveGateBadge">Locked</div>
            </div>
          </div>
        </div>
      </aside>

      <main class="workspace">
        <section id="tab-dashboard" class="tab-panel active">
          <div class="overview-grid">
            <div class="panel preview-panel">
              <div class="panel-head tight">
                <div>
                  <h2>Live Preview</h2>
                  <div class="panel-sub">Live UVC preview feed.</div>
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

            <div class="panel status-panel">
              <div class="panel-head tight">
                <div>
                  <h2>Vehicle Status</h2>
                  <div class="panel-sub">Live chassis feedback.</div>
                </div>
              </div>
              <div class="properties" id="vehicleStatus"></div>
            </div>

            <div class="panel console-panel">
              <div class="panel-head tight">
                <div>
                  <h2>Console</h2>
                  <div class="panel-sub">Backend events and task transitions.</div>
                </div>
              </div>
              <textarea id="console" readonly></textarea>
            </div>
          </div>
        </section>

        <section id="tab-tasks" class="tab-panel">
          <div class="workflow-shell fresh-workflow">
            <div class="workflow-board primary-workflow">
              <article class="workflow-step map">
                <div class="step-number">0</div>
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
                <div class="step-number">1</div>
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
                  <div class="step-number">2</div>
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

        <section id="tab-library" class="tab-panel">
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

        <section id="tab-settings" class="tab-panel">
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
      </main>
    </section>
  </div>
<script>
    let stateCache = null;
    let activeTab = 'dashboard';
    const dirtyFields = new Set();
    let consoleAutoFollow = true;
    const defaultTheme = window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
    const editableFieldIds = [
      'mapName', 'missionName', 'lineCruiseVx',
      'mappingRecorddata', 'sensorHeight', 'bodyXOffset',
      'bodyYOffset', 'rollGain', 'pitchGain'
    ];

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

    function formatNumber(value, digits=2) {
      const num = Number(value);
      if (!Number.isFinite(num)) return value ?? '--';
      return num.toFixed(digits).replace(/\.?0+$/, '');
    }

    function updateVehicleStatus(status) {
      const root = document.getElementById('vehicleStatus');
      const entries = [
        ['Gear', status.motion?.gear ?? '--'],
        ['Linear X', formatNumber(status.motion?.vx_mps ?? status.motion?.vx ?? '--', 2)],
        ['Linear Y', formatNumber(status.motion?.vy_mps ?? status.motion?.vy ?? '--', 2)],
        ['Yaw Rate', formatNumber(status.motion?.wz_dps ?? status.motion?.wz ?? '--', 2)],
        ['Battery', status.battery?.soc_pct ?? '--'],
        ['Voltage', formatNumber(status.battery?.voltage_v ?? '--', 2)],
        ['Current', formatNumber(status.battery?.current_a ?? '--', 2)],
        ['Capacity', formatNumber(status.battery?.capacity_ah ?? '--', 2)],
        ['Unlock', status.io?.unlock_ok ?? '--'],
        ['Remote', status.io?.remote_control ?? '--'],
        ['E-Stop', status.io?.estop ?? '--'],
        ['Error', (status.error?.level ?? '--') + '/' + (status.error?.type ?? '--')],
      ];
      root.innerHTML = entries.map(([k, v]) => `<div>${k}</div><div>${v}</div>`).join('');
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
      activeTab = tabName;
      for (const name of ['dashboard', 'tasks', 'library', 'settings']) {
        document.getElementById('tab-' + name).classList.toggle('active', name === tabName);
        document.getElementById('tabBtn-' + name).classList.toggle('active', name === tabName);
      }
    }

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
      document.getElementById('workflowGateSummary').textContent = ready ? 'Ready for record or drive' : 'Waiting for map lock';
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

    function applyState(data) {
      stateCache = data;
      const activeLocalizationMapId = data.active_localization_map_id || '';
      const localizationText = data.localization_status || 'Not started';
      document.getElementById('taskStatus').textContent = data.task_status || 'Idle';
      document.getElementById('localizationStatus').textContent = localizationText;
      document.getElementById('cameraStatus').textContent = data.camera_status || 'Stopped';
      document.getElementById('cameraStatusDashboard').textContent = data.camera_status || 'Stopped';
      document.getElementById('previewSource').textContent = data.preview_source || 'Waiting';
      document.getElementById('previewSourceDashboard').textContent = data.preview_source || 'Waiting';
      document.getElementById('previewModeBadge').textContent = String(data.preview_source || 'Preview');
      const previewSourceTuning = document.getElementById('previewSourceTuning');
      if (previewSourceTuning) previewSourceTuning.textContent = data.preview_source || 'Waiting for preview stream';
      const previewModeBadgeTuning = document.getElementById('previewModeBadgeTuning');
      if (previewModeBadgeTuning) previewModeBadgeTuning.textContent = String(data.preview_source || 'Preview');
      document.getElementById('canStateSummary').textContent = data.can_status || 'Unknown';

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
      document.getElementById('selectedMapSummary').textContent = selectedMap ? selectedMap.label : '--';
      document.getElementById('selectedMissionSummary').textContent = selectedMission ? selectedMission.label : '--';
      document.getElementById('mapDeleteSummary').textContent = libraryMap ? libraryMap.label : 'No map selected.';
      document.getElementById('missionDeleteSummary').textContent = libraryMission ? libraryMission.label : 'No mission selected.';
      renderMissionPreview(data.selected_library_mission_preview || null);

      document.getElementById('projectionSummary').textContent =
        'h=' + (data.settings.sensor_height_m || '--') +
        ', x=' + (data.settings.body_x_offset_m || '--') +
        ', y=' + (data.settings.body_y_offset_m || '--');
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

    for (const id of editableFieldIds) markFieldDirty(id);
    const consoleBox = document.getElementById('console');
    if (consoleBox) {
      consoleBox.addEventListener('scroll', () => {
        const nearBottom = (consoleBox.scrollHeight - consoleBox.scrollTop - consoleBox.clientHeight) < 24;
        consoleAutoFollow = nearBottom;
      });
    }
    setTheme(localStorage.getItem('autorun_final_theme') || defaultTheme);
    setInterval(refreshState, 1000);
    setInterval(refreshPreview, 250);
    refreshState();
    refreshPreview();
  </script>
</body>
</html>
"""


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
        self.camera_monitor: RosImageMonitor | None = None
        self.pose_debug_monitor: RosPoseDebugMonitor | None = None
        self.camera_status = "Stopped"
        self.can_status = "Unknown"
        self.preview_source = "Waiting for preview stream"
        self.localization_status = "Not started"
        self.task_status = "Idle"
        self.localization_map_path: Path | None = None
        self.pending_action: str | None = None
        self.latest_preview_jpeg: bytes | None = None
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
        self._start_pose_debug_monitor()
        self.event_thread = threading.Thread(target=self._pump_events, daemon=True)
        self.event_thread.start()
        self.status_thread = threading.Thread(target=self._poll_vehicle_status, daemon=True)
        self.status_thread.start()

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
        match = re.search(r"state\\s+([A-Z]+)", text)
        if match:
            return match.group(1)
        if "<NOARP,UP" in text or ",UP" in text:
            return "UP"
        return "UNKNOWN"

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

    def _start_camera_monitor(self) -> None:
        if self.camera_monitor is not None:
            return
        self.camera_monitor = RosImageMonitor(self.events, topics=(UVC_PREVIEW_TOPIC,))
        self.camera_monitor.start()
        self.camera_status = "Starting..."
        self._log("UVC preview monitor started. This is only the camera preview stream, not lidar local guidance.")

    def _start_uvc_preview_publisher(self, auto: bool = False) -> None:
        if self.preview_worker is not None:
            return
        worker = ProcessWorker([ROS_PYTHON, *self._uvc_preview_args()], PROJECT_ROOT, "UVC Preview", self.events)
        self.preview_worker = worker
        worker.start()
        self._start_camera_monitor()
        self._log("UVC preview publisher started." if not auto else "UVC preview publisher auto-started.")

    def _stop_uvc_preview_publisher(self) -> None:
        if self.preview_worker is None:
            return
        self.preview_worker.stop()

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
                self._stop_uvc_preview_publisher()
                return
            active = self._active_localization_worker()
            if active is not None:
                self._log("Stop requested for shared localization.")
                active.stop()
                self._cleanup_localization_processes("after global stop request")
                self._stop_uvc_preview_publisher()

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
                        if topic == UVC_PREVIEW_TOPIC:
                            self.preview_source = "Raw UVC Preview"
                        elif topic:
                            self.preview_source = topic
                elif event == "camera_status":
                    self.camera_status = str(payload)
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
                self.vehicle_status = bridge.snapshot()
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
                "pose_debug": dict(self.pose_debug_state),
            }

    def preview_jpeg(self) -> bytes | None:
        with self.lock:
            return self.latest_preview_jpeg

    def close(self) -> None:
        self.closing = True
        if self.task_worker is not None:
            self.task_worker.stop()
        active = self._active_localization_worker()
        if active is not None:
            active.stop()
        if self.preview_worker is not None:
            self.preview_worker.stop()
        if self.camera_monitor is not None:
            self.camera_monitor.stop()
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
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self._safe_write(body)

    def _send_text(self, text: str, status: int = 200, content_type: str = "text/plain; charset=utf-8") -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
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
