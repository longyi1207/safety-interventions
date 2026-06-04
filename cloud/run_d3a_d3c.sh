#!/usr/bin/env bash
# Instance B: D3a entangle then D3c fuse (sequential on one GPU).
# Set SKIP_D3C=1 when D3c runs on a dedicated instance (launch_d3c.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f ~/.si_eval_env ]] && set -a && source ~/.si_eval_env && set +a
export PYTHONUNBUFFERED=1
PY="${PY:-python3}"
CFG="${CFG:-configs/d3_lora_train.yaml}"
log() { echo "[$(date -u +%H:%M:%S)] [d3a/c] $*" | tee -a outputs/cloud_job.log; }

pip install -q peft 2>/dev/null || true
log "build datasets"
"$PY" scripts/build_d3_datasets.py
log "train D3a"
"$PY" scripts/train_lora_track.py --track d3a_ent --config "$CFG"
log "eval D3a"
"$PY" scripts/eval_d3_checkpoint.py --adapter outputs/adapters/d3a_ent --config "$CFG" --judge
if [[ "${SKIP_D3C:-0}" == "1" ]]; then
  log "SKIP_D3C=1 — D3c on dedicated instance"
  log "done D3a only"
  exit 0
fi
log "train D3c fuse"
"$PY" scripts/train_lora_track.py --track d3c_fuse --config "$CFG"
log "eval D3c"
"$PY" scripts/eval_d3_checkpoint.py --adapter outputs/adapters/d3c_fuse --config "$CFG" --judge --fuse-eval
log "done D3a+D3c"
