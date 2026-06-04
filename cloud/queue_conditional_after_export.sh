#!/usr/bin/env bash
# Queue judge-only conditional after export JSONL is complete (no duplicate generate).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/cloud/.active/latest.env"
KEY="${AWS_KEY_FILE:-$ROOT/cloud/.ssh/safety-interventions.pem}"
SSH="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"
[[ -f "$KEY" ]] && SSH="ssh -i $KEY -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"

# Kill old watcher that would run full conditional_main (duplicate GPU).
$SSH "${SSH_USER}@${PUBLIC_IP}" bash -s <<'REMOTE'
pkill -f "conditional_after_export|run_conditional_main.sh" 2>/dev/null || true
pgrep -f run_export_c1_c4_review || true
REMOTE

$SSH "${SSH_USER}@${PUBLIC_IP}" bash -s <<'REMOTE'
set -euo pipefail
cd ~/ai_lab/code/safety_interventions
OUT=outputs/ablations/c1_c4_review_main.jsonl
LOG=outputs/logs/conditional_from_review.log
mkdir -p outputs/logs
if pgrep -f conditional_evil_from_review.py >/dev/null; then
  echo "conditional_from_review already running"
  exit 0
fi
nohup bash -c '
  set -euo pipefail
  cd ~/ai_lab/code/safety_interventions
  OUT=outputs/ablations/c1_c4_review_main.jsonl
  echo "waiting for $OUT (200 lines)..."
  while true; do
    if [ -f "$OUT" ]; then
      n=$(wc -l < "$OUT" | tr -d " ")
      if [ "$n" -ge 200 ]; then break; fi
    fi
    sleep 60
  done
  echo "export done ($n lines); starting conditional_from_review (API only)"
  set -a && source ~/.si_eval_env && set +a
  bash cloud/run_conditional_from_review.sh
' >> "$LOG" 2>&1 &
echo "queued conditional_from_review pid=$! log=$LOG"
REMOTE

echo "Queued judge-only conditional on $PUBLIC_IP (cancelled duplicate GPU conditional)."
