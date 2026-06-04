#!/usr/bin/env bash
# Poll until gpu finish pipeline exits, then start eval_only (uses .env keys via start_job.sh).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/cloud/_ssh_env.sh" && _cloud_ssh "$ROOT"

POLL_SEC="${POLL_SEC:-120}"
MAX_HOURS="${MAX_HOURS:-8}"
deadline=$((SECONDS + MAX_HOURS * 3600))

echo "[watch] Waiting for cloud/run_phase_b_finish.sh to finish (poll ${POLL_SEC}s)..."

while (( SECONDS < deadline )); do
  if ! $SSH "${SSH_USER}@${PUBLIC_IP}" "test -f $REMOTE_DIR/outputs/cloud_job.pid" 2>/dev/null; then
    echo "[watch] No pid file — assuming idle."
    break
  fi
  running=$($SSH "${SSH_USER}@${PUBLIC_IP}" "pid=\$(cat $REMOTE_DIR/outputs/cloud_job.pid 2>/dev/null); ps -p \$pid >/dev/null 2>&1 && echo yes || echo no" 2>/dev/null || echo no)
  if [[ "$running" != "yes" ]]; then
    echo "[watch] Finish job process ended."
    break
  fi
  $SSH "${SSH_USER}@${PUBLIC_IP}" "tail -c 2000 $REMOTE_DIR/outputs/cloud_job.log 2>/dev/null" | tr '\r' '\n' | tail -1 || true
  sleep "$POLL_SEC"
done

if (( SECONDS >= deadline )); then
  echo "[watch] Timeout after ${MAX_HOURS}h — not starting eval." >&2
  exit 1
fi

if $SSH "${SSH_USER}@${PUBLIC_IP}" "grep -q 'finish done' $REMOTE_DIR/outputs/cloud_job.log 2>/dev/null"; then
  echo "[watch] Log shows 'finish done'."
else
  echo "[watch] WARN: finish may have failed — check cloud_job.log. Starting eval anyway if vectors exist."
fi

echo "[watch] Starting PIPELINE=eval_only..."
PIPELINE=eval_only bash "$ROOT/cloud/start_job.sh"
echo "[watch] eval_only started. Monitor: bash cloud/status.sh"
