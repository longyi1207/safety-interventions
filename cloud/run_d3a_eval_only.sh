#!/usr/bin/env bash
# D3a adapter already trained — tamper eval only.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f ~/.si_eval_env ]] && set -a && source ~/.si_eval_env && set +a
export PYTHONUNBUFFERED=1
PY="${PY:-python3}"
CFG="${CFG:-configs/d3_lora_train.yaml}"
log() { echo "[$(date -u +%H:%M:%S)] [d3a-eval] $*" | tee -a outputs/cloud_job.log; }

log "eval D3a"
"$PY" scripts/eval_d3_checkpoint.py --adapter outputs/adapters/d3a_ent --config "$CFG" --judge
log "done D3a eval"
