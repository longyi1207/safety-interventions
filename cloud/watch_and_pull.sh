#!/usr/bin/env bash
# Poll cloud job until finished, then pull_results + teardown.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ACTIVE="${JOB_ENV:-$ROOT/cloud/.active/latest.env}"
# shellcheck source=/dev/null
source "$ACTIVE"
# shellcheck source=/dev/null
source "$ROOT/cloud/_ssh_env.sh"
KEY="${AWS_KEY_FILE:-$ROOT/cloud/.ssh/safety-interventions.pem}"
SSH="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15"
[[ -f "$KEY" ]] && SSH="ssh -i $KEY -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15"
REMOTE_DIR="${REMOTE_REPO_DIR:-~/ai_lab/code/safety_interventions}"
REMOTE="${SSH_USER}@${PUBLIC_IP}"
INTERVAL="${POLL_SEC:-120}"
LOG="$ROOT/outputs/ablations/watch_and_pull.log"

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "Watching ${JOB_ID:-?} @ ${PUBLIC_IP} every ${INTERVAL}s"

while true; do
  if ! $SSH "$REMOTE" "test -f $REMOTE_DIR/outputs/cloud_job.pid" 2>/dev/null; then
    log "No pid file — assuming done or never started"
    break
  fi
  alive=$($SSH "$REMOTE" "pid=\$(cat $REMOTE_DIR/outputs/cloud_job.pid 2>/dev/null); ps -p \$pid >/dev/null 2>&1 && echo yes || echo no" || echo no)
  if [[ "$alive" == "no" ]]; then
    log "Process finished"
    break
  fi
  tail=$($SSH "$REMOTE" "tail -1 $REMOTE_DIR/outputs/cloud_job.log 2>/dev/null" || true)
  log "Still running… $tail"
  sleep "$INTERVAL"
done

log "Pulling results…"
bash "$ROOT/cloud/pull_results.sh" 2>&1 | tee -a "$LOG"
log "Done. Artifacts under outputs/cloud_pull/${JOB_ID:-latest}/"
