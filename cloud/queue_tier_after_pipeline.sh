#!/usr/bin/env bash
# Wait for current cloud_job.pid to finish, then run tier experiments.
set -euo pipefail
cd "$(dirname "$0")/.."
PID_FILE=outputs/cloud_job.pid
LOG=outputs/tier_queue.log
echo "[$(date -u +%H:%M:%S)] waiting for pipeline pid" >>"$LOG"
while [[ -f "$PID_FILE" ]]; do
  pid=$(cat "$PID_FILE")
  if ! ps -p "$pid" >/dev/null 2>&1; then
    break
  fi
  sleep 120
done
echo "[$(date -u +%H:%M:%S)] starting tier experiments" >>"$LOG"
nohup bash cloud/run_tier_experiments.sh >>outputs/cloud_job.log 2>&1 &
echo $! >"$PID_FILE"
echo "[$(date -u +%H:%M:%S)] tier pid=$!" >>"$LOG"
