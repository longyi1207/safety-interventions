#!/usr/bin/env bash
# HarmBench main split: C0/C1/C4 + all-traits RFA (LLM judge).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY="${PY:-python3}"
HB_CONFIG="${CONFIG:-configs/qwen7b_harmbench.cloud.yaml}"
QCONFIG="${QCONFIG:-configs/evil_qualities.yaml}"
MANIFEST="${MANIFEST:-prompts/harmbench_manifest_main.jsonl}"
MAX_NEW="${MAX_NEW:-256}"

bash cloud/preflight_openai.sh || {
  echo "Fix OPENAI_API_KEY in ai_notes/.env then retry." >&2
  exit 1
}

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a outputs/cloud_job.log; }

N=$("$PY" -c "print(sum(1 for _ in open('$MANIFEST')))")
log "[main] manifest=$MANIFEST n=$N max_new=$MAX_NEW"

log "[main] Phase A conditions C0,C1,C4"
"$PY" scripts/sweep_evil_v2.py \
  --config "$HB_CONFIG" \
  --manifest "$MANIFEST" \
  --conditions C0,C1,C4 \
  --max-new "$MAX_NEW" \
  --skip-sweep \
  --output outputs/ablations/main_v2_conditions.json \
  2>&1 | tee -a outputs/cloud_job.log

log "[main] all-traits benchmark C0,C1,C4,C_all_traits_rfa"
"$PY" scripts/eval_all_traits_benchmark.py \
  --config "$HB_CONFIG" \
  --qconfig "$QCONFIG" \
  --manifest "$MANIFEST" \
  --conditions C0,C1,C4,C_all_traits_rfa \
  --max-new "$MAX_NEW" \
  --output outputs/ablations/all_traits_benchmark_main.json \
  2>&1 | tee -a outputs/cloud_job.log

log "[main] eval complete"
