#!/usr/bin/env bash
# Main n=200: C1/C4 default + EVIL_SYSTEM variants (LLM judge).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY="${PY:-python3}"
CONFIG="${CONFIG:-configs/qwen7b_harmbench.cloud.yaml}"
MANIFEST="${MANIFEST:-prompts/harmbench_manifest_main.jsonl}"

bash cloud/preflight_openai.sh
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a outputs/cloud_job.log; }

log "[main_evil_system] n=200 C1,C4,C1_evil_system,C4_evil_system,evil_system_only"
"$PY" scripts/eval_evil_system_main.py \
  --config "$CONFIG" \
  --manifest "$MANIFEST" \
  --max-new "${MAX_NEW:-256}" \
  --out outputs/ablations/main_evil_system_conditions.json \
  2>&1 | tee -a outputs/cloud_job.log
log "[main_evil_system] done"
