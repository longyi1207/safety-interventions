#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JOB="${1:-si-20260602-194000}"
IP="${2:-35.175.110.106}"
KEY="${AWS_KEY_FILE:-$ROOT/cloud/.ssh/safety-interventions.pem}"
SSH_OPTS=(-o StrictHostKeyChecking=accept-new)
[[ -f "$KEY" ]] && SSH_OPTS=(-i "$KEY" -o StrictHostKeyChecking=accept-new)
REMOTE=~/ai_lab/code/safety_interventions
DEST="$ROOT/outputs/cloud_pull/${JOB}"

mkdir -p "$DEST/adapters" "$DEST/ablations"
for ad in d2_er d3a_ent d3c_fuse d3c_fuse_v3c; do
  scp "${SSH_OPTS[@]}" -r "ubuntu@${IP}:${REMOTE}/outputs/adapters/${ad}" "$DEST/adapters/" 2>/dev/null || true
done
scp "${SSH_OPTS[@]}" "${REMOTE}/outputs/ablations/d3c_fuse_zero_review_main.jsonl" "$DEST/ablations/" 2>/dev/null || true
scp "${SSH_OPTS[@]}" "${REMOTE}/outputs/ablations/d3c_fuse_zero_review_main_summary.json" "$DEST/ablations/" 2>/dev/null || true
scp "${SSH_OPTS[@]}" "${REMOTE}/outputs/cloud_job.log" "$DEST/" 2>/dev/null || true
# v3c from D3 host if second arg omitted and parallel env exists
if [[ -f "$ROOT/cloud/.active/parallel_d23.env" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT/cloud/.active/parallel_d23.env"
  scp "${SSH_OPTS[@]}" -r "ubuntu@${D3_IP}:${REMOTE}/outputs/adapters/d3c_fuse_v3c" "$DEST/adapters/" 2>/dev/null || true
fi
echo "Pulled -> $DEST"
