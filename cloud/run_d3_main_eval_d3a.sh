#!/usr/bin/env bash
# D3a host: d3a_ent main eval only.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f ~/.si_eval_env ]] && set -a && source ~/.si_eval_env && set +a
export PYTHONUNBUFFERED=1
PY="${PY:-python3}"
CFG="${CFG:-configs/d3_lora_train.yaml}"
MAIN="${MAIN:-prompts/harmbench_manifest_main.jsonl}"

log() { echo "[$(date -u +%H:%M:%S)] [d3-main-d3a] $*" | tee -a outputs/cloud_job.log; }
bash cloud/preflight_openai.sh
log "eval d3a_ent"
"$PY" scripts/eval_d3_checkpoint.py \
  --adapter outputs/adapters/d3a_ent --config "$CFG" \
  --harm-manifest "$MAIN" --judge \
  --out outputs/adapters/d3a_ent/eval_tamper_main.json \
  2>&1 | tee -a outputs/cloud_job.log
log "done d3a main"
