#!/usr/bin/env bash
# D2 host: d2_er + d3c_fuse main eval + fuse_zero export.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f ~/.si_eval_env ]] && set -a && source ~/.si_eval_env && set +a
export PYTHONUNBUFFERED=1
PY="${PY:-python3}"
CFG="${CFG:-configs/d3_lora_train.yaml}"
MAIN="${MAIN:-prompts/harmbench_manifest_main.jsonl}"
ADAPTERS="${ADAPTERS:-d2_er,d3c_fuse}"

log() { echo "[$(date -u +%H:%M:%S)] [d3-main-d2] $*" | tee -a outputs/cloud_job.log; }
bash cloud/preflight_openai.sh

IFS=',' read -ra ADS <<< "$ADAPTERS"
for ad in "${ADS[@]}"; do
  [[ -d "outputs/adapters/$ad" ]] || { log "SKIP $ad"; continue; }
  extra=()
  [[ "$ad" == "d3c_fuse" ]] && extra=(--fuse-eval)
  log "eval $ad"
  "$PY" scripts/eval_d3_checkpoint.py \
    --adapter "outputs/adapters/$ad" --config "$CFG" \
    --harm-manifest "$MAIN" --judge "${extra[@]}" \
    --out "outputs/adapters/$ad/eval_tamper_main.json" \
    2>&1 | tee -a outputs/cloud_job.log
done

if [[ -d outputs/adapters/d3c_fuse ]]; then
  log "fuse_zero export"
  "$PY" scripts/export_d3_fuse_zero_review.py \
    --adapter outputs/adapters/d3c_fuse --config "$CFG" \
    --manifest "$MAIN" \
    --output outputs/ablations/d3c_fuse_zero_review_main.jsonl --judge
  "$PY" scripts/analyze_d3_fuse_zero_review.py \
    --review outputs/ablations/d3c_fuse_zero_review_main.jsonl \
    --out outputs/ablations/d3c_fuse_zero_review_main_summary.json
fi
log "done d2 main"
