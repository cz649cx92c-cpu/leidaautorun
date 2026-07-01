#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

import cv2
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage


def ensure_ros_python_available() -> None:
    for path in (
        Path("/opt/ros/humble/lib/python3.10/site-packages"),
        Path("/usr/lib/python3/dist-packages"),
    ):
        text = str(path)
        if path.exists() and text not in sys.path:
            sys.path.append(text)


def normalize_device_path(source: str) -> str:
    text = str(source).strip()
    if text.isdigit():
        return f"/dev/video{text}"
    return text


class PreviewPublisher(Node):
    def __init__(self, topic: str) -> None:
        super().__init__("autorun_final_uvc_preview")
        self.publisher = self.create_publisher(CompressedImage, topic, 1)


def open_capture(source: str, width: int, height: int, fps: float, fourcc: str) -> cv2.VideoCapture:
    source = normalize_device_path(source)
    if source.isdigit():
        capture = cv2.VideoCapture(int(source))
    elif source.startswith("/dev/video"):
        capture = cv2.VideoCapture(source, cv2.CAP_V4L2)
    else:
        capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"failed to open UVC source: {source}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, max(1, int(width)))
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, max(1, int(height)))
    capture.set(cv2.CAP_PROP_FPS, max(1.0, float(fps)))
    if fourcc:
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc[:4]))
    return capture


def prepare_preview(frame, width: int, height: int):
    target_width = max(1, int(width))
    target_height = max(1, int(height))
    h, w = frame.shape[:2]
    scale = min(target_width / float(w), target_height / float(h))
    scale = max(scale, 1e-6)
    resized_w = max(1, int(w * scale))
    resized_h = max(1, int(h * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(frame, (resized_w, resized_h), interpolation=interpolation)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish lightweight UVC preview to a ROS compressed topic")
    parser.add_argument("--source", default="/dev/video0")
    parser.add_argument("--camera-width", type=int, default=1024)
    parser.add_argument("--camera-height", type=int, default=768)
    parser.add_argument("--camera-fps", type=float, default=10.0)
    parser.add_argument("--camera-fourcc", default="MJPG")
    parser.add_argument("--image-topic", default="/autorun_final/uvc_preview/compressed")
    parser.add_argument("--preview-width", type=int, default=640)
    parser.add_argument("--preview-height", type=int, default=360)
    parser.add_argument("--jpeg-quality", type=int, default=35)
    parser.add_argument("--publish-fps", type=float, default=6.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stop_requested = False
    node: PreviewPublisher | None = None
    executor: SingleThreadedExecutor | None = None

    def _handle_stop(signum, frame):
        del signum, frame
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    capture = open_capture(
        args.source,
        args.camera_width,
        args.camera_height,
        args.camera_fps,
        args.camera_fourcc,
    )
    try:
        os.environ.setdefault("ROS_HOME", "/tmp/autorun_final_ros2_home")
        rclpy.init()
        node = PreviewPublisher(args.image_topic)
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        min_interval = 0.0 if args.publish_fps <= 0 else (1.0 / float(args.publish_fps))
        last_publish_at = 0.0
        jpeg_quality = max(10, min(80, int(args.jpeg_quality)))
        while not stop_requested and rclpy.ok():
            executor.spin_once(timeout_sec=0.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            now = time.monotonic()
            if min_interval > 0.0 and (now - last_publish_at) < min_interval:
                continue
            preview = prepare_preview(frame, args.preview_width, args.preview_height)
            ok_encoded, encoded = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if not ok_encoded:
                time.sleep(0.02)
                continue
            msg = CompressedImage()
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.format = "jpeg"
            msg.data = encoded.tobytes()
            try:
                if not stop_requested and rclpy.ok():
                    node.publisher.publish(msg)
                else:
                    break
            except Exception:
                # ROS is already shutting down; exit quietly instead of logging a traceback.
                break
            last_publish_at = now
    finally:
        capture.release()
        try:
            if executor is not None and node is not None:
                executor.remove_node(node)
            if executor is not None:
                executor.shutdown()
            if node is not None:
                node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
