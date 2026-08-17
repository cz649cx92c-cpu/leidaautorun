from __future__ import annotations

import json
import fcntl
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CONFIG_PATH = Path(__file__).resolve().parent / "config" / "dual_lidar.json"
START_LOCK_PATH = Path("/tmp/autorunlida-dual-lidar.lock")


@dataclass(frozen=True)
class LidarConfig:
    role: str
    node_name: str
    device: str
    scan_topic: str
    frame_id: str
    sensor_yaw_deg: float
    lidar_yaw_correction_deg: float
    lidar_x_offset_m: float
    lidar_y_offset_m: float
    extrinsics_confirmed: bool

    def driver_args(self, driver_bin: Path) -> list[str]:
        return [
            str(driver_bin),
            "--ros-args",
            "-r",
            f"__node:={self.node_name}",
            "-r",
            f"scan:={self.scan_topic}",
            "-p",
            f"port_name:={self.device}",
            "-p",
            f"frame_id:={self.frame_id}",
        ]


@dataclass(frozen=True)
class DualLidarConfig:
    front: LidarConfig
    rear: LidarConfig

    def sensors(self) -> tuple[LidarConfig, LidarConfig]:
        return self.front, self.rear


@contextmanager
def lidar_driver_start_lock():
    with START_LOCK_PATH.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _parse_sensor(role: str, payload: Any) -> LidarConfig:
    if not isinstance(payload, dict):
        raise RuntimeError(f"dual lidar config '{role}' must be an object")
    return LidarConfig(
        role=role,
        node_name=str(payload.get("node_name") or f"{role}_lidar_node").strip(),
        device=str(payload.get("device") or "").strip(),
        scan_topic=str(payload.get("scan_topic") or "").strip(),
        frame_id=str(payload.get("frame_id") or "").strip(),
        sensor_yaw_deg=float(payload.get("sensor_yaw_deg", 0.0)),
        lidar_yaw_correction_deg=float(payload.get("lidar_yaw_correction_deg", 0.0)),
        lidar_x_offset_m=float(payload.get("lidar_x_offset_m", 0.0)),
        lidar_y_offset_m=float(payload.get("lidar_y_offset_m", 0.0)),
        extrinsics_confirmed=bool(payload.get("extrinsics_confirmed", False)),
    )


def load_dual_lidar_config(path: Path = CONFIG_PATH) -> DualLidarConfig:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"failed to load dual lidar config {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"dual lidar config {path} must contain an object")
    config = DualLidarConfig(
        front=_parse_sensor("front", payload.get("front")),
        rear=_parse_sensor("rear", payload.get("rear")),
    )
    for sensor in config.sensors():
        if not sensor.node_name or not sensor.device or not sensor.scan_topic or not sensor.frame_id:
            raise RuntimeError(f"dual lidar config '{sensor.role}' has an empty required field")
        if not sensor.device.startswith("/dev/"):
            raise RuntimeError(f"dual lidar device must be under /dev: {sensor.device}")
        if not sensor.scan_topic.startswith("/"):
            raise RuntimeError(f"dual lidar scan topic must be absolute: {sensor.scan_topic}")
    for field in ("node_name", "device", "scan_topic", "frame_id"):
        if getattr(config.front, field) == getattr(config.rear, field):
            raise RuntimeError(f"front and rear lidar must use different {field}")
    return config


def lidar_driver_running(driver_bin: Path, sensor: LidarConfig) -> bool:
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
    required = (
        str(driver_bin),
        f"__node:={sensor.node_name}",
        f"scan:={sensor.scan_topic}",
        f"port_name:={sensor.device}",
    )
    return any(all(token in line for token in required) for line in result.stdout.splitlines())


def legacy_lidar_driver_running(driver_bin: Path) -> bool:
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
    return any(
        str(driver_bin) in line and "port_name:=" not in line
        for line in result.stdout.splitlines()
    )
