#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY="${PY:-python3}"
CFG="${CFG:-configs/d3_lora_train.yaml}"
log() { echo "[$(date -u +%H:%M:%S)] [d2] $*" | tee -a outputs/cloud_job.log; }

pip install -q peft 2>/dev/null || true
log "build datasets"
"$PY" scripts/build_d3_datasets.py
log "train D2-ER"
"$PY" scripts/train_lora_track.py --track d2_er --config "$CFG"
log "eval D2"
"$PY" scripts/eval_d3_checkpoint.py --adapter outputs/adapters/d2_er --config "$CFG" --judge
log "done D2"
