#!/usr/bin/env bash
# D3c v3c: fuse-on warmup then train-time kill hinge (fuse_zero_prob > 0).
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f ~/.si_eval_env ]] && set -a && source ~/.si_eval_env && set +a
export PYTHONUNBUFFERED=1
PY="${PY:-python3}"
CFG="${CFG:-configs/d3_lora_train_v3c_kill.yaml}"
MAIN="${MAIN:-prompts/harmbench_manifest_main.jsonl}"

log() { echo "[$(date -u +%H:%M:%S)] [d3c-v3c] $*" | tee -a outputs/cloud_job.log; }

pip install -q peft 2>/dev/null || true
if [[ -d outputs/adapters/d3a_ent ]]; then
  log "d3a_ent main eval (pre-train)"
  "$PY" scripts/eval_d3_checkpoint.py \
    --adapter outputs/adapters/d3a_ent --config configs/d3_lora_train.yaml \
    --harm-manifest "$MAIN" --judge \
    --out outputs/adapters/d3a_ent/eval_tamper_main.json \
    2>&1 | tee -a outputs/cloud_job.log
fi
log "build datasets"
"$PY" scripts/build_d3_datasets.py
log "train D3c v3c (kill hinge)"
"$PY" scripts/train_lora_track.py --track d3c_fuse_v3c --config "$CFG"
log "eval D3c v3c dev"
"$PY" scripts/eval_d3_checkpoint.py \
  --adapter outputs/adapters/d3c_fuse_v3c \
  --config "$CFG" \
  --judge --fuse-eval \
  --out outputs/adapters/d3c_fuse_v3c/eval_tamper.json
log "eval D3c v3c main"
"$PY" scripts/eval_d3_checkpoint.py \
  --adapter outputs/adapters/d3c_fuse_v3c \
  --config "$CFG" \
  --harm-manifest "$MAIN" \
  --judge --fuse-eval \
  --out outputs/adapters/d3c_fuse_v3c/eval_tamper_main.json
log "done d3c v3c"
