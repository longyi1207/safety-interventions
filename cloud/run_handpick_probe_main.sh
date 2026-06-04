#!/usr/bin/env bash
# Main intervention matrix on 3 handpick prompts (no subspace dependency).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY="${PY:-python3}"
CONFIG="${CONFIG:-configs/qwen7b_harmbench.cloud.yaml}"

bash cloud/preflight_openai.sh
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a outputs/cloud_job.log; }

log "[handpick-main] probe matrix (3 prompts, single load)"
"$PY" scripts/probe_evil_handpick.py \
  --config "$CONFIG" \
  --manifest prompts/handpick_c0_probe.jsonl \
  --out-jsonl outputs/ablations/handpick_probe_main.jsonl \
  --out-summary outputs/ablations/handpick_probe_main_summary.json \
  --max-new 512 \
  --judge \
  --conditions "C0,C1,C2,C4,C1_evil_a20,C1_evil_a40,C4_evil_a20,C1_multilayer,C1_ablate_d1,C1_evil_system,C4_evil_system,C2_evil_system,evil_system_only" \
  2>&1 | tee -a outputs/cloud_job.log
log "[handpick-main] done"
