#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import signal
import socket
import subprocess
import threading
import time
from collections import deque
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from row_geometry import RowFollowerConfig, estimate_row


ROOT = Path(__file__).resolve().parent
ROS_SETUP = Path("/opt/ros/humble/setup.bash")
WORKSPACE_SETUP = Path("/home/orangepi/ugv/install/setup.bash")
FOLLOWER = ROOT / "plant_lidar_centerline_follower.py"
CALIBRATION = ROOT / "config" / "lidar_calibration.json"


class LidarMonitor(Node):
    def __init__(self) -> None:
        super().__init__("dual_lidar_test_web_monitor")
        self.lock = threading.Lock()
        self.scans: dict[str, tuple[LaserScan, float]] = {}
        self.status: dict[str, Any] = {}
        self.create_subscription(LaserScan, "/front/scan", lambda msg: self._scan("front", msg), 10)
        self.create_subscription(LaserScan, "/rear/scan", lambda msg: self._scan("rear", msg), 10)
        self.create_subscription(String, "/plant_row/status", self._status, 10)

    def _scan(self, role: str, msg: LaserScan) -> None:
        with self.lock:
            self.scans[role] = (msg, time.monotonic())

    def _status(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        with self.lock:
            self.status = payload

    def snapshot(self, role: str) -> tuple[LaserScan | None, float, dict[str, Any]]:
        with self.lock:
            item = self.scans.get(role)
            status = dict(self.status)
        if item is None:
            return None, float("inf"), status
        return item[0], time.monotonic() - item[1], status


class App:
    def __init__(self, speed: float) -> None:
        self.lock = threading.RLock()
        self.mode = "front"
        self.speed = abs(float(speed))
        self.process: subprocess.Popen[str] | None = None
        self.intentional_stop = False
        self.logs: deque[str] = deque(maxlen=120)
        self.last_width = {"front": 0.60, "rear": 0.60}
        self.calibration = self._load_calibration()
        self.monitor = LidarMonitor()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.monitor)
        threading.Thread(target=self.executor.spin, daemon=True).start()

    def _load_calibration(self) -> dict[str, float]:
        defaults = {
            "front_lidar_yaw_correction_deg": -3.0,
            "front_lidar_x_offset_m": 0.0,
            "front_lidar_y_offset_m": 0.035,
            "rear_lidar_yaw_correction_deg": -3.0,
            "rear_lidar_x_offset_m": 0.0,
            "rear_lidar_y_offset_m": 0.035,
        }
        try:
            raw = json.loads(CALIBRATION.read_text(encoding="utf-8"))
        except Exception:
            raw = {}
        defaults["front_lidar_yaw_correction_deg"] = float(
            raw.get("front_lidar_yaw_correction_deg", raw.get("lidar_yaw_correction_deg", -3.0))
        )
        defaults["front_lidar_x_offset_m"] = float(
            raw.get("front_lidar_x_offset_m", raw.get("lidar_x_offset_m", 0.0))
        )
        defaults["front_lidar_y_offset_m"] = float(
            raw.get("front_lidar_y_offset_m", raw.get("lidar_y_offset_m", 0.035))
        )
        defaults["rear_lidar_yaw_correction_deg"] = float(
            raw.get("rear_lidar_yaw_correction_deg", raw.get("lidar_yaw_correction_deg", -3.0))
        )
        defaults["rear_lidar_x_offset_m"] = float(
            raw.get("rear_lidar_x_offset_m", raw.get("lidar_x_offset_m", 0.0))
        )
        defaults["rear_lidar_y_offset_m"] = float(
            raw.get("rear_lidar_y_offset_m", raw.get("lidar_y_offset_m", 0.035))
        )
        return defaults

    def _cfg(self, role: str) -> RowFollowerConfig:
        cfg = RowFollowerConfig(
            row_width=0.60, min_row_width=0.48, max_row_width=0.78,
            lookahead_x=0.60, forward_min=0.25, forward_max=1.20,
            lateral_limit=0.60, range_min=0.05, range_max=6.0,
            bin_size=0.20, min_points=8, min_bins=2, min_line_bins=4,
            min_side_points_per_bin=2, center_deadband=0.03,
            left_percentile=30.0, right_percentile=70.0,
            sensor_yaw_deg=180.0, vehicle_half_width=0.20, safety_margin=0.04,
            center_jump_reject=0.25, one_side_center_jump_reject=0.30,
        )
        if role == "rear":
            return replace(
                cfg,
                # Rear scan is normalized to the direction of reverse travel.
                # The physical lidar 0-degree ray must point toward the tail.
                sensor_yaw_deg=180.0,
                lidar_yaw_correction_deg=self.calibration["rear_lidar_yaw_correction_deg"],
                lidar_x_offset_m=self.calibration["rear_lidar_x_offset_m"],
                lidar_y_offset_m=self.calibration["rear_lidar_y_offset_m"],
                scan_view_center_deg=0.0,
                scan_view_angle_deg=180.0,
                reflect_x_axis=True,
            )
        return replace(
            cfg,
            lidar_yaw_correction_deg=self.calibration["front_lidar_yaw_correction_deg"],
            lidar_x_offset_m=self.calibration["front_lidar_x_offset_m"],
            lidar_y_offset_m=self.calibration["front_lidar_y_offset_m"],
        )

    @staticmethod
    def _points(value: Any, limit: int = 180) -> list[list[float]]:
        if value is None:
            return []
        arr = np.asarray(value)
        if arr.size == 0:
            return []
        if len(arr) > limit:
            arr = arr[:: max(1, math.ceil(len(arr) / limit))]
        return [[round(float(x), 3), round(float(y), 3)] for x, y in arr]

    @staticmethod
    def _line(value: Any) -> list[float] | None:
        if value is None:
            return None
        return [round(float(value[0]), 5), round(float(value[1]), 5)]

    def state(self) -> dict[str, Any]:
        with self.lock:
            role = self.mode
            proc = self.process
            running = proc is not None and proc.poll() is None
            exit_code = None if proc is None or running else proc.returncode
            calibration = dict(self.calibration)
        scan, age, follower_status = self.monitor.snapshot(role)
        payload: dict[str, Any] = {
            "mode": role,
            "topic": f"/{role}/scan",
            "scan_age": None if not math.isfinite(age) else round(age, 3),
            "scan_live": scan is not None and age < 0.6,
            "running": running,
            "exit_code": exit_code,
            "speed": self.speed,
            "calibration": calibration,
            "logs": list(self.logs)[-30:],
            "follower": follower_status,
            "view_angle_deg": 180 if role == "rear" else 360,
        }
        if scan is None or age > 0.6:
            payload.update(found=False, points=[], left=[], right=[], center=[], lines={})
            return payload
        estimate, debug = estimate_row(scan, self._cfg(role), self.last_width[role])
        if estimate.found and estimate.row_width > 0:
            self.last_width[role] = float(estimate.row_width)
        payload.update(
            found=bool(estimate.found), mode_name=estimate.mode,
            input_points=int(len(debug.raw_points)),
            center_y=round(float(estimate.center_y), 4),
            heading_deg=round(math.degrees(float(estimate.heading_rad)), 2),
            row_width=round(float(estimate.row_width), 3),
            points=self._points(debug.web_points),
            left=self._points(debug.left_points, 80),
            right=self._points(debug.right_points, 80),
            center=self._points(debug.center_points, 80),
            lines={
                "left": self._line(debug.left_line),
                "right": self._line(debug.right_line),
                "center": self._line(debug.center_line),
            },
        )
        return payload

    def _command(self, reverse: bool) -> list[str]:
        cal = self.calibration
        command = (
            f"source {ROS_SETUP} && source {WORKSPACE_SETUP} && /usr/bin/python3 {FOLLOWER}"
            " --front-scan-topic /front/scan --rear-scan-topic /rear/scan"
            " --status-topic /plant_row/status --interface socketcan --channel can0"
            f" --gear 4t4d --speed {self.speed} --row-width 0.60 --vehicle-width 0.40"
            " --boundary-width-tolerance-m 0.0 --center-y-target 0.0"
            " --front-sensor-yaw-deg 180"
            f" --front-lidar-yaw-correction-deg {cal['front_lidar_yaw_correction_deg']}"
            f" --front-lidar-x-offset-m {cal['front_lidar_x_offset_m']}"
            f" --front-lidar-y-offset-m {cal['front_lidar_y_offset_m']}"
            " --rear-sensor-yaw-deg 180"
            f" --rear-lidar-yaw-correction-deg {cal['rear_lidar_yaw_correction_deg']}"
            f" --rear-lidar-x-offset-m {cal['rear_lidar_x_offset_m']}"
            f" --rear-lidar-y-offset-m {cal['rear_lidar_y_offset_m']}"
            " --front-extrinsics-confirmed --rear-extrinsics-confirmed"
            " --forward-lookahead-x 0.6 --control-deadband-y 0.001"
            " --forward-lost-hold-sec 0.35 --forward-lost-stop-sec 0.50"
            " --forward-lost-hold-wz-scale 0.5 --forward-lost-hold-max-wz-deg 0.6"
            " --k-heading 0.05 --reverse-steer-sign -1.0"
            + (" --reverse" if reverse else "")
        )
        return ["bash", "-lc", command]

    def start(self, reverse: bool) -> None:
        self.stop()
        with self.lock:
            self.mode = "rear" if reverse else "front"
            self.intentional_stop = False
            self.process = subprocess.Popen(
                self._command(reverse), cwd=str(ROOT), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
            )
            proc = self.process
            self.logs.append(f"{time.strftime('%H:%M:%S')} {'reverse' if reverse else 'forward'} started")
        threading.Thread(target=self._read_logs, args=(proc,), daemon=True).start()

    def _read_logs(self, proc: subprocess.Popen[str]) -> None:
        if proc.stdout is not None:
            for line in proc.stdout:
                with self.lock:
                    self.logs.append(line.rstrip())
        code = proc.wait()
        with self.lock:
            if not self.intentional_stop and self.process is proc:
                self.logs.append(f"process exited code={code}")

    def stop(self) -> None:
        with self.lock:
            proc = self.process
            self.intentional_stop = True
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
        with self.lock:
            self.process = None

    def save_calibration(self, payload: dict[str, Any]) -> None:
        values = {
            "rear_lidar_yaw_correction_deg": float(payload["yaw"]),
            "rear_lidar_x_offset_m": float(payload["x"]),
            "rear_lidar_y_offset_m": float(payload["y"]),
        }
        if abs(values["rear_lidar_yaw_correction_deg"]) > 180:
            raise ValueError("yaw must be within -180..180")
        with self.lock:
            self.calibration.update(values)
            CALIBRATION.parent.mkdir(parents=True, exist_ok=True)
            CALIBRATION.write_text(json.dumps(self.calibration, indent=2) + "\n", encoding="utf-8")
            self.logs.append("rear calibration saved; restart reverse to apply control parameters")


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>双雷达中线测试</title><style>
:root{color-scheme:dark;--bg:#0c1013;--panel:#151b1f;--line:#2a343a;--text:#e6ecef;--muted:#8d9ba3;--green:#50c878;--red:#ef6461;--amber:#e8b44f}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px system-ui,"Microsoft YaHei",sans-serif;letter-spacing:0}
header{height:58px;display:flex;align-items:center;gap:20px;padding:0 20px;border-bottom:1px solid var(--line)}h1{font-size:17px;margin:0;font-weight:650}#live{margin-left:auto;color:var(--muted)}
.toolbar{display:flex;align-items:end;gap:10px;padding:12px 20px;border-bottom:1px solid var(--line);flex-wrap:wrap}button,input{height:36px;border:1px solid var(--line);background:#11171a;color:var(--text);border-radius:5px;padding:0 12px}button{cursor:pointer;font-weight:600}button.primary{background:#176b3a;border-color:#258d50}button.danger{color:#ffb4b2}button:hover{border-color:#60717a}label{display:grid;gap:5px;color:var(--muted);font-size:12px}input{width:92px}
main{height:calc(100vh - 119px);display:grid;grid-template-columns:minmax(520px,1fr) 330px}.plot{min-width:0;position:relative;border-right:1px solid var(--line)}canvas{width:100%;height:100%;display:block}.plot-meta{position:absolute;left:16px;top:14px;display:flex;gap:8px}.tag{padding:5px 8px;background:#11171add;border:1px solid var(--line);border-radius:4px;color:var(--muted)}
aside{overflow:auto;padding:18px}.section{padding:0 0 18px;margin:0 0 18px;border-bottom:1px solid var(--line)}h2{font-size:12px;text-transform:uppercase;color:var(--muted);margin:0 0 12px}.metric{display:flex;justify-content:space-between;padding:6px 0}.metric b{font-variant-numeric:tabular-nums}.ok{color:var(--green)}.bad{color:var(--red)}pre{white-space:pre-wrap;word-break:break-all;color:#aab6bc;font:12px/1.5 ui-monospace,monospace;margin:0}.legend{display:flex;gap:14px;color:var(--muted);font-size:12px}.dot:before{content:"";display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px;background:currentColor}
@media(max-width:800px){main{grid-template-columns:1fr;height:auto}.plot{height:62vh;border-right:0;border-bottom:1px solid var(--line)}aside{height:auto}.toolbar{padding:10px}.toolbar label{flex:1}header{padding:0 12px}}
</style></head><body>
<header><h1>双雷达中线测试</h1><span id="mode">前雷达</span><span id="live">连接中</span></header>
<div class="toolbar"><button onclick="act('preview_front')">只看前雷达</button><button onclick="act('preview_rear')">只看后雷达</button><button class="primary" onclick="act('forward')">前进测试</button><button class="primary" onclick="act('reverse')">倒车测试</button><button class="danger" onclick="act('stop')">停止</button><label>后雷达角度修正<input id="yaw" type="number" step="0.5"></label><label>纵向偏移 m<input id="offx" type="number" step="0.005"></label><label>横向偏移 m<input id="offy" type="number" step="0.005"></label><button onclick="saveCal()">保存后雷达外参</button></div>
<main><section class="plot"><canvas id="plot"></canvas><div class="plot-meta"><span class="tag" id="topic">-</span><span class="tag" id="view">-</span></div></section><aside>
<div class="section"><h2>识别状态</h2><div class="metric"><span>LaserScan</span><b id="scan">-</b></div><div class="metric"><span>输入点数</span><b id="count">-</b></div><div class="metric"><span>中线</span><b id="found">-</b></div><div class="metric"><span>横向误差</span><b id="center">-</b></div><div class="metric"><span>航向角</span><b id="heading">-</b></div><div class="metric"><span>通道宽度</span><b id="width">-</b></div></div>
<div class="section"><h2>独立安装外参</h2><div class="metric"><span>前雷达</span><b id="frontcal">-</b></div><div class="metric"><span>后雷达</span><b id="rearcal">-</b></div></div>
<div class="section legend"><span class="dot" style="color:#71818a">点云</span><span class="dot" style="color:#50c878">左边界</span><span class="dot" style="color:#ef8b45">右边界</span><span class="dot" style="color:#ef6461">中线</span></div>
<div><h2>运行日志</h2><pre id="logs"></pre></div></aside></main>
<script>
const c=document.getElementById('plot'),ctx=c.getContext('2d');let initialized=false,last={};
function resize(){const d=devicePixelRatio||1,r=c.getBoundingClientRect();c.width=r.width*d;c.height=r.height*d;ctx.setTransform(d,0,0,d,0,0)}addEventListener('resize',resize);resize();
function line(v,x0,x1,color,w=2){if(!v)return;ctx.strokeStyle=color;ctx.lineWidth=w;ctx.beginPath();for(let i=0;i<2;i++){let x=i?x1:x0,y=v[0]*x+v[1],p=map(x,y);i?ctx.lineTo(...p):ctx.moveTo(...p)}ctx.stroke()}
function map(x,y){const r=c.getBoundingClientRect(),s=Math.min(r.width/3.2,r.height/3.4),rear=last.mode==='rear';return[r.width/2-y*s,(rear?r.height*.18:r.height*.82)+(rear?x*s:-x*s)]}
function dots(a,color,r=2){ctx.fillStyle=color;for(const [x,y] of a||[]){const p=map(x,y);ctx.beginPath();ctx.arc(p[0],p[1],r,0,Math.PI*2);ctx.fill()}}
function draw(s){const r=c.getBoundingClientRect();ctx.clearRect(0,0,r.width,r.height);ctx.fillStyle='#0c1013';ctx.fillRect(0,0,r.width,r.height);ctx.strokeStyle='#253038';ctx.lineWidth=1;let o=map(0,0);ctx.beginPath();ctx.moveTo(0,o[1]);ctx.lineTo(r.width,o[1]);ctx.moveTo(o[0],0);ctx.lineTo(o[0],r.height);ctx.stroke();ctx.strokeStyle='#dce5e9';ctx.strokeRect(o[0]-14,o[1]-30,28,42);dots(s.points,'#71818a',1.7);dots(s.left,'#50c878',2.6);dots(s.right,'#ef8b45',2.6);dots(s.center,'#ef6461',2.8);line(s.lines?.left,.15,1.5,'#50c878');line(s.lines?.right,.15,1.5,'#ef8b45');line(s.lines?.center,0,1.3,'#ef6461',3)}
async function refresh(){try{let s=await(await fetch('/api/state',{cache:'no-store'})).json();last=s;draw(s);mode.textContent=s.mode==='rear'?'后雷达 / 倒车方向':'前雷达 / 前进方向';topic.textContent=s.topic;view.textContent=`输入视角 ${s.view_angle_deg}°`;live.textContent=s.running?'自动控制运行中':'仅预览 / 已停止';scan.textContent=s.scan_live?`${s.scan_age.toFixed(2)} s`:'无数据';scan.className=s.scan_live?'ok':'bad';count.textContent=s.input_points??0;found.textContent=s.found?'已识别':'未识别';found.className=s.found?'ok':'bad';center.textContent=s.found?`${s.center_y.toFixed(3)} m`:'-';heading.textContent=s.found?`${s.heading_deg.toFixed(1)}°`:'-';width.textContent=s.found?`${s.row_width.toFixed(3)} m`:'-';let q=s.calibration;frontcal.textContent=`${q.front_lidar_yaw_correction_deg}° / ${q.front_lidar_x_offset_m} / ${q.front_lidar_y_offset_m} m`;rearcal.textContent=`${q.rear_lidar_yaw_correction_deg}° / ${q.rear_lidar_x_offset_m} / ${q.rear_lidar_y_offset_m} m`;logs.textContent=(s.logs||[]).slice(-18).join('\n');if(!initialized){yaw.value=q.rear_lidar_yaw_correction_deg;offx.value=q.rear_lidar_x_offset_m;offy.value=q.rear_lidar_y_offset_m;initialized=true}}catch(e){live.textContent='网页连接中断'}}
async function act(action){await fetch('/api/action',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action})});refresh()}
async function saveCal(){await fetch('/api/calibration',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({yaw:+yaw.value,x:+offx.value,y:+offy.value})});initialized=false;refresh()}
setInterval(refresh,180);refresh();
</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    app: App

    def _json(self, payload: Any, code: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/api/state":
            self._json(self.app.state())
        elif self.path in {"/", "/index.html"}:
            data = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        try:
            size = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(size) or b"{}")
            if self.path == "/api/action":
                action = payload.get("action")
                if action == "forward": self.app.start(False)
                elif action == "reverse": self.app.start(True)
                elif action == "stop": self.app.stop()
                elif action == "preview_front":
                    self.app.stop(); self.app.mode = "front"
                elif action == "preview_rear":
                    self.app.stop(); self.app.mode = "rear"
                else: raise ValueError("unknown action")
            elif self.path == "/api/calibration":
                self.app.save_calibration(payload)
            else:
                self.send_error(404); return
            self._json({"ok": True})
        except Exception as exc:
            self._json({"ok": False, "error": str(exc)}, 400)

    def log_message(self, _format: str, *_args: Any) -> None:
        return


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--speed", type=float, default=0.12)
    args = parser.parse_args()
    rclpy.init()
    app = App(args.speed)
    Handler.app = app
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    def _terminate(*_args: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _terminate)
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("8.8.8.8", 80))
        lan_ip = probe.getsockname()[0]
        probe.close()
    except OSError:
        lan_ip = args.host
    print(f"Dual lidar test web: http://{lan_ip}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.stop(); server.server_close(); app.executor.shutdown(); app.monitor.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
