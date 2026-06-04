#!/usr/bin/env bash
# Main n=200 eval for D2 / D3a / D3c adapters + fuse_zero export for review.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f ~/.si_eval_env ]] && set -a && source ~/.si_eval_env && set +a
export PYTHONUNBUFFERED=1
PY="${PY:-python3}"
CFG="${CFG:-configs/d3_lora_train.yaml}"
MAIN="${MAIN:-prompts/harmbench_manifest_main.jsonl}"

log() { echo "[$(date -u +%H:%M:%S)] [d3-main] $*" | tee -a outputs/cloud_job.log; }

bash cloud/preflight_openai.sh

for ad in d2_er d3a_ent d3c_fuse; do
  if [[ ! -d "outputs/adapters/$ad" ]]; then
    log "SKIP missing adapter $ad"
    continue
  fi
  extra=()
  [[ "$ad" == "d3c_fuse" ]] && extra=(--fuse-eval)
  log "eval $ad on $MAIN"
  "$PY" scripts/eval_d3_checkpoint.py \
    --adapter "outputs/adapters/$ad" \
    --config "$CFG" \
    --harm-manifest "$MAIN" \
    --judge \
    "${extra[@]}" \
    --out "outputs/adapters/$ad/eval_tamper_main.json" \
    2>&1 | tee -a outputs/cloud_job.log
done

if [[ -d outputs/adapters/d3c_fuse ]]; then
  log "export fuse_zero review (main)"
  "$PY" scripts/export_d3_fuse_zero_review.py \
    --adapter outputs/adapters/d3c_fuse \
    --config "$CFG" \
    --manifest "$MAIN" \
    --output outputs/ablations/d3c_fuse_zero_review_main.jsonl \
    --judge \
    2>&1 | tee -a outputs/cloud_job.log
  "$PY" scripts/analyze_d3_fuse_zero_review.py \
    --review outputs/ablations/d3c_fuse_zero_review_main.jsonl \
    --out outputs/ablations/d3c_fuse_zero_review_main_summary.json
fi

log "done d3 main eval"
