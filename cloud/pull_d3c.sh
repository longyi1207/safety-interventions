#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PAR="$ROOT/cloud/.active/parallel_d3c.env"
[[ -f "$PAR" ]] || { echo "missing $PAR"; exit 1; }
# shellcheck source=/dev/null
source "$PAR"
KEY="${AWS_KEY_FILE:-$ROOT/cloud/.ssh/safety-interventions.pem}"
SSH_OPTS=(-o StrictHostKeyChecking=accept-new)
[[ -f "$KEY" ]] && SSH_OPTS=(-i "$KEY" -o StrictHostKeyChecking=accept-new)
REMOTE=~/ai_lab/code/safety_interventions
DEST="$ROOT/outputs/cloud_pull/${D3C_JOB}"
mkdir -p "$DEST/adapters"
echo "Pull $D3C_JOB from $D3C_IP"
scp "${SSH_OPTS[@]}" -r "ubuntu@${D3C_IP}:${REMOTE}/outputs/adapters/d3c_fuse" "$DEST/adapters/" 2>/dev/null || true
scp "${SSH_OPTS[@]}" "${REMOTE}/outputs/cloud_job.log" "$DEST/" 2>/dev/null || true
scp "${SSH_OPTS[@]}" "${REMOTE}/outputs/adapters/d3c_fuse/eval_tamper.json" "$DEST/adapters/" 2>/dev/null || true
echo "Done -> $DEST"
