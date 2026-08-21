#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/humble/setup.bash
source /home/orangepi/ugv/install/setup.bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec /usr/bin/python3 "$ROOT/plant_lidar_centerline_web.py" --host 0.0.0.0 --port 8788
