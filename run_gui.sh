#!/usr/bin/env bash
set -eo pipefail

cd "$(dirname "$0")"
source /opt/ros/humble/setup.bash
source /home/orangepi/ugv/install/setup.bash
exec /usr/bin/python3 ./plant_lidar_centerline_gui.py "$@"
