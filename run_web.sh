#!/usr/bin/env bash
set -eo pipefail

cd "$(dirname "$0")"
export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"
source /opt/ros/humble/setup.bash
source /home/orangepi/ugv/install/setup.bash

# Clear any stale process still holding the web UI port.
if command -v fuser >/dev/null 2>&1; then
    fuser -k 8765/tcp >/dev/null 2>&1 || true
fi

exec /usr/bin/python3 ./app_web.py "$@"
