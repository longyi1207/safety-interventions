#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ACTIVE="${1:-$ROOT/cloud/.active/latest.env}"
# shellcheck source=/dev/null
source "$ACTIVE"
# shellcheck source=/dev/null
source "$ROOT/cloud/_ssh_env.sh"
KEY="${AWS_KEY_FILE:-$ROOT/cloud/.ssh/safety-interventions.pem}"
SSH="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"
[[ -f "$KEY" ]] && SSH="ssh -i $KEY -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"
REMOTE_DIR="${REMOTE_REPO_DIR:-~/ai_lab/code/safety_interventions}"

REMOTE="${SSH_USER}@${PUBLIC_IP}"

echo "=== Job ${JOB_ID:-?} @ ${PUBLIC_IP} ==="
if ! $SSH "$REMOTE" "test -d $REMOTE_DIR" 2>/dev/null; then
  echo "SSH failed or code not synced. Run: cloud/sync_code.sh"
  exit 1
fi

echo ""
echo "--- job_status.json ---"
$SSH "$REMOTE" "cat $REMOTE_DIR/outputs/job_status.json 2>/dev/null" || echo "(not started)"

echo ""
echo "--- process ---"
$SSH "$REMOTE" "if [ -f $REMOTE_DIR/outputs/cloud_job.pid ]; then pid=\$(cat $REMOTE_DIR/outputs/cloud_job.pid); ps -p \$pid -o pid,etime,command 2>/dev/null || echo finished; else echo no pid; fi"

echo ""
echo "--- log tail ---"
$SSH "$REMOTE" "tail -n 20 $REMOTE_DIR/outputs/cloud_job.log 2>/dev/null" || echo "(no log)"
