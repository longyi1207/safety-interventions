#!/usr/bin/env bash
# Export C1 vs C4 full responses on main manifest for human review.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY="${PY:-python3}"
CONFIG="${CONFIG:-configs/qwen7b_harmbench.cloud.yaml}"

bash cloud/preflight_openai.sh

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a outputs/cloud_job.log; }
log "[export] C1/C4 review JSONL (main n=200, with judge)"
"$PY" scripts/export_c1_c4_review.py \
  --config "$CONFIG" \
  --manifest prompts/harmbench_manifest_main.jsonl \
  --output outputs/ablations/c1_c4_review_main.jsonl \
  --max-new 256 \
  --judge \
  2>&1 | tee -a outputs/cloud_job.log
log "[export] done"
