#!/usr/bin/env bash
# Track A: D3a RFA scale sweep on HarmBench main n=200.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f ~/.si_eval_env ]] && set -a && source ~/.si_eval_env && set +a
export PYTHONUNBUFFERED=1
PY="${PY:-python3}"
CFG="${CFG:-configs/d3_lora_train.yaml}"
MAIN="${MAIN:-prompts/harmbench_manifest_main.jsonl}"
OUT="${OUT:-outputs/arxiv_mva}"

log() { echo "[$(date -u +%H:%M:%S)] [arxiv-A] $*" | tee -a outputs/cloud_job.log; }
mkdir -p "$OUT"
bash cloud/preflight_openai.sh

log "RFA scale sweep d3a_ent (main n=200)"
"$PY" scripts/eval_d3a_rfa_scale.py \
  --adapter outputs/adapters/d3a_ent \
  --config "$CFG" \
  --harm-manifest "$MAIN" \
  --scales "0,0.5,1,1.5,2" \
  --out "$OUT/d3a_rfa_scale_main.json" \
  2>&1 | tee -a outputs/cloud_job.log

log "done track A"
