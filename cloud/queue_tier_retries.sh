#!/usr/bin/env bash
# Wait for main tier job, then auto-retry any missing outputs.
set -euo pipefail
cd "$(dirname "$0")/.."
PID_FILE=outputs/cloud_job.pid
log() { echo "[$(date -u +%H:%M:%S)] [tier-watcher] $*" | tee -a outputs/cloud_job.log; }

if [[ -f "$PID_FILE" ]]; then
  pid=$(cat "$PID_FILE")
  log "waiting for tier pid=$pid"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 45
  done
  log "tier pid=$pid exited"
fi
sleep 5
log "starting retry_tier_failures"
bash cloud/retry_tier_failures.sh
echo $? >outputs/tier_retry_exit.code
log "retry exit=$(cat outputs/tier_retry_exit.code)"
