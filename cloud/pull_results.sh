#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ACTIVE="${JOB_ENV:-${1:-$ROOT/cloud/.active/latest.env}}"
# shellcheck source=/dev/null
source "$ACTIVE"
# shellcheck source=/dev/null
source "$ROOT/cloud/_ssh_env.sh"
KEY="${AWS_KEY_FILE:-$ROOT/cloud/.ssh/safety-interventions.pem}"
SSH="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"
[[ -f "$KEY" ]] && SSH="ssh -i $KEY -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"
REMOTE_DIR="${REMOTE_REPO_DIR:-~/ai_lab/code/safety_interventions}"

DEST="$ROOT/outputs/cloud_pull/${JOB_ID}"
mkdir -p "$DEST/prompts" "$DEST/prompts/quality_contrast" "$DEST/ablations" "$DEST/vectors"
R="${SSH_USER}@${PUBLIC_IP}:${REMOTE_DIR}"

$SSH "${SSH_USER}@${PUBLIC_IP}" "test -d $REMOTE_DIR" || { echo "sync first"; exit 1; }

scp -i "${AWS_KEY_FILE:-$ROOT/cloud/.ssh/safety-interventions.pem}" -o StrictHostKeyChecking=accept-new \
  "${R}/prompts/evil_contrast_full.jsonl" "$DEST/prompts/" 2>/dev/null || true
scp -i "${AWS_KEY_FILE:-$ROOT/cloud/.ssh/safety-interventions.pem}" -o StrictHostKeyChecking=accept-new \
  "${R}/outputs/vectors/qwen7b_vectors.pt" "$DEST/vectors/" 2>/dev/null || true
scp -i "${AWS_KEY_FILE:-$ROOT/cloud/.ssh/safety-interventions.pem}" -o StrictHostKeyChecking=accept-new \
  "${R}/outputs/ablations/main_v2_conditions.json" "${R}/outputs/ablations/all_traits_benchmark_main.json" "$DEST/ablations/" 2>/dev/null || true
scp -i "${AWS_KEY_FILE:-$ROOT/cloud/.ssh/safety-interventions.pem}" -o StrictHostKeyChecking=accept-new \
  "${R}/outputs/ablations/main_evil_system_conditions.json" \
  "${R}/outputs/ablations/conditional_evil_C1ref_main.json" \
  "${R}/outputs/ablations/c1_c4_review_main.jsonl" \
  "${R}/outputs/ablations/handpick_probe_main.jsonl" \
  "${R}/outputs/ablations/handpick_probe_main_summary.json" \
  "${R}/outputs/ablations/handpick_probe_subspace.jsonl" \
  "${R}/outputs/ablations/handpick_probe_subspace_summary.json" \
  "$DEST/ablations/" 2>/dev/null || true
scp -i "${AWS_KEY_FILE:-$ROOT/cloud/.ssh/safety-interventions.pem}" -o StrictHostKeyChecking=accept-new \
  "${R}/outputs/vectors/refusal_subspace_L18.pt" "$DEST/vectors/" 2>/dev/null || true
scp -i "${AWS_KEY_FILE:-$ROOT/cloud/.ssh/safety-interventions.pem}" -o StrictHostKeyChecking=accept-new \
  "${R}/outputs/ablations/dev_v2_conditions.json" "${R}/outputs/ablations/dev_c2_variants.json" "$DEST/ablations/" 2>/dev/null || true
scp -i "${AWS_KEY_FILE:-$ROOT/cloud/.ssh/safety-interventions.pem}" -o StrictHostKeyChecking=accept-new \
  "${R}/outputs/job_status.json" "${R}/outputs/cloud_job.log" "$DEST/" 2>/dev/null || true
scp -r -i "${AWS_KEY_FILE:-$ROOT/cloud/.ssh/safety-interventions.pem}" -o StrictHostKeyChecking=accept-new \
  "${R}/outputs/ablations/tier_experiments/" "$DEST/ablations/tier_experiments/" 2>/dev/null || true
scp -r -i "${AWS_KEY_FILE:-$ROOT/cloud/.ssh/safety-interventions.pem}" -o StrictHostKeyChecking=accept-new \
  "${R}/prompts/quality_contrast/" "$DEST/prompts/quality_contrast/" 2>/dev/null || true
scp -i "${AWS_KEY_FILE:-$ROOT/cloud/.ssh/safety-interventions.pem}" -o StrictHostKeyChecking=accept-new \
  "${R}/outputs/vectors/qwen7b_qualities.pt" "${R}/outputs/vectors/qwen7b_qualities_ortho.pt" "$DEST/vectors/" 2>/dev/null || true
scp -i "${AWS_KEY_FILE:-$ROOT/cloud/.ssh/safety-interventions.pem}" -o StrictHostKeyChecking=accept-new \
  "${R}/outputs/ablations/phase_b_"*.json "${R}/outputs/ablations/quality_cosine_before_ortho.json" "$DEST/ablations/" 2>/dev/null || true

echo "Pulled to $DEST"
cat "$DEST/ablations/dev_v2_conditions.json" 2>/dev/null || true
