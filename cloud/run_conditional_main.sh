#!/usr/bin/env bash
# C1 vs C4 paired evil-persona eval on HarmBench main (n=200).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY="${PY:-python3}"
CONFIG="${CONFIG:-configs/qwen7b_harmbench.cloud.yaml}"

bash cloud/preflight_openai.sh

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a outputs/cloud_job.log; }
log "[conditional] C1-ref paired evil on main manifest"
"$PY" scripts/eval_conditional_evil.py \
  --config "$CONFIG" \
  --manifest prompts/harmbench_manifest_main.jsonl \
  --conditions C0,C1,C4 \
  --pair-ref C1 \
  --max-new 256 \
  --output outputs/ablations/conditional_evil_C1ref_main.json \
  2>&1 | tee -a outputs/cloud_job.log
log "[conditional] done"
