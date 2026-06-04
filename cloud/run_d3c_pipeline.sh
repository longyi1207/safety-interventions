#!/usr/bin/env bash
# Cloud wrapper: full D3c fuse pipeline (v3d default).
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f ~/.si_eval_env ]] && set -a && source ~/.si_eval_env && set +a
export PYTHONUNBUFFERED=1

TRACK="${TRACK:-d3c_fuse_v3d}"
CFG="${CFG:-configs/d3_lora_train_v3d.yaml}"
SMOKE="${SMOKE:-0}"

log() { echo "[$(date -u +%H:%M:%S)] [d3c-pipeline] $*" | tee -a outputs/cloud_job.log; }

pip install -q peft 2>/dev/null || true
log "start TRACK=$TRACK SMOKE=$SMOKE"
TRACK="$TRACK" CFG="$CFG" SMOKE="$SMOKE" bash scripts/run_d3c_fuse_pipeline.sh 2>&1 | tee -a outputs/cloud_job.log
log "pipeline complete"
