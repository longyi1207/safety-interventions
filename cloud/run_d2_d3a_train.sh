#!/usr/bin/env bash
# One GPU: build data → D2-ER LoRA → D3a LoRA → eval both (sequential, ~6–12h).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY="${PY:-python3}"
CFG="${CFG:-configs/d3_lora_train.yaml}"

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a outputs/cloud_job.log; }

pip install -q peft 2>/dev/null || true

log "[d2_d3a] build datasets"
"$PY" scripts/build_d3_datasets.py 2>&1 | tee -a outputs/cloud_job.log

log "[d2_d3a] train D2-ER LoRA"
"$PY" scripts/train_lora_track.py --track d2_er --config "$CFG" 2>&1 | tee -a outputs/cloud_job.log

log "[d2_d3a] train D3a entangle LoRA"
"$PY" scripts/train_lora_track.py --track d3a_ent --config "$CFG" 2>&1 | tee -a outputs/cloud_job.log

log "[d2_d3a] eval D2"
"$PY" scripts/eval_d3_checkpoint.py --adapter outputs/adapters/d2_er --config "$CFG" --judge 2>&1 | tee -a outputs/cloud_job.log

log "[d2_d3a] eval D3a"
"$PY" scripts/eval_d3_checkpoint.py --adapter outputs/adapters/d3a_ent --config "$CFG" --judge 2>&1 | tee -a outputs/cloud_job.log

log "[d2_d3a] done"
