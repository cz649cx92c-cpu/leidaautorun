#!/usr/bin/env bash
set -eo pipefail

cd "$(dirname "$0")"
export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"
source /opt/ros/humble/setup.bash
source /home/orangepi/ugv/install/setup.bash
exec /usr/bin/python3 ./app_web.py "$@"
