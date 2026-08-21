#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/humble/setup.bash
source /home/orangepi/ugv/install/setup.bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LIDAR_NODE=/home/orangepi/ugv/install/lidar_pkg/lib/lidar_pkg/lidar_node
FRONT_PORT=/dev/serial/by-path/platform-fc8c0000.usb-usb-0:1:1.0
REAR_PORT=/dev/serial/by-path/platform-fc800000.usb-usb-0:1.2.4:1.0

topic_has_scan() {
  timeout 5 ros2 topic echo "$1" --once >/dev/null 2>&1
}

ensure_lidar() {
  local role="$1" topic="$2" port="$3" frame="$4"
  if topic_has_scan "$topic" || topic_has_scan "$topic"; then
    echo "$role lidar healthy on $topic"
    return
  fi

  echo "$role lidar has no scan; restarting its driver..."
  while read -r pid; do
    [[ -n "$pid" ]] && kill -TERM "$pid" 2>/dev/null || true
  done < <(pgrep -f "__node:=${role}_lidar_node" || true)
  sleep 1
  nohup "$LIDAR_NODE" --ros-args \
    -r "__node:=${role}_lidar_node" \
    -r "scan:=${topic}" \
    -p "port_name:=${port}" \
    -p "frame_id:=${frame}" \
    >>"/tmp/${role}_lidar_test.log" 2>&1 &

  for _ in 1 2 3 4; do
    topic_has_scan "$topic" && return
  done
  echo "ERROR: $role lidar still has no data on $topic" >&2
  tail -20 "/tmp/${role}_lidar_test.log" >&2 || true
  return 1
}

ensure_lidar front /front/scan "$FRONT_PORT" front_laser
ensure_lidar rear /rear/scan "$REAR_PORT" rear_laser

exec /usr/bin/python3 "$ROOT/plant_lidar_centerline_web.py" --host 0.0.0.0 --port 8788
