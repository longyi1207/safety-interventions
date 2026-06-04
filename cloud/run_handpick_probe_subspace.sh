#!/usr/bin/env bash
# Extract refusal d2/d3 then probe subspace conditions on 3 handpick prompts.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY="${PY:-python3}"
CONFIG="${CONFIG:-configs/qwen7b_harmbench.cloud.yaml}"
SUB=outputs/vectors/refusal_subspace_L18.pt

bash cloud/preflight_openai.sh
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a outputs/cloud_job.log; }

log "[handpick-sub] extract refusal subspace L18"
"$PY" scripts/extract_refusal_subspace.py \
  --config "$CONFIG" \
  --n-pairs 50 \
  --k 3 \
  --out "$SUB" \
  2>&1 | tee -a outputs/cloud_job.log

log "[handpick-sub] probe subspace conditions"
"$PY" scripts/probe_evil_handpick.py \
  --config "$CONFIG" \
  --manifest prompts/handpick_c0_probe.jsonl \
  --vectors outputs/vectors/qwen7b_vectors.pt \
  --subspace "$SUB" \
  --out-jsonl outputs/ablations/handpick_probe_subspace.jsonl \
  --out-summary outputs/ablations/handpick_probe_subspace_summary.json \
  --max-new 512 \
  --judge \
  --conditions "C1,C1_rfa_d1_d2,C1_rfa_d2_only,C1_rfa_d1_d2_evil,C1_rfa_d1_d2_evil_sys,C4,C4_evil_system" \
  2>&1 | tee -a outputs/cloud_job.log
log "[handpick-sub] done"
