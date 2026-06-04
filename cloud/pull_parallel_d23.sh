#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PAR="$ROOT/cloud/.active/parallel_d23.env"
[[ -f "$PAR" ]] || { echo "missing $PAR"; exit 1; }
# shellcheck source=/dev/null
source "$PAR"
KEY="${AWS_KEY_FILE:-$ROOT/cloud/.ssh/safety-interventions.pem}"
SSH_OPTS=(-o StrictHostKeyChecking=accept-new)
[[ -f "$KEY" ]] && SSH_OPTS=(-i "$KEY" -o StrictHostKeyChecking=accept-new)
REMOTE=~/ai_lab/code/safety_interventions

pull_one() {
  local JOB="$1" IP="$2"
  local DEST="$ROOT/outputs/cloud_pull/${JOB}"
  mkdir -p "$DEST/adapters" "$DEST/data"
  echo "Pull $JOB from $IP -> $DEST"
  scp "${SSH_OPTS[@]}" -r "ubuntu@${IP}:${REMOTE}/outputs/adapters/d2_er" "$DEST/adapters/" 2>/dev/null || true
  scp "${SSH_OPTS[@]}" -r "ubuntu@${IP}:${REMOTE}/outputs/adapters/d3a_ent" "$DEST/adapters/" 2>/dev/null || true
  scp "${SSH_OPTS[@]}" -r "ubuntu@${IP}:${REMOTE}/outputs/adapters/d3c_fuse" "$DEST/adapters/" 2>/dev/null || true
  scp "${SSH_OPTS[@]}" "${REMOTE}/outputs/cloud_job.log" "$DEST/" 2>/dev/null || true
  scp "${SSH_OPTS[@]}" "${REMOTE}/outputs/adapters/*/eval_tamper.json" "$DEST/adapters/" 2>/dev/null || true
}

pull_one "$D2_JOB" "$D2_IP"
pull_one "$D3_JOB" "$D3_IP"
echo "Done."
