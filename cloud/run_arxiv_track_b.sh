#!/usr/bin/env bash
# Track B: unified D2 vs D3a table — same harness, main n=200.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f ~/.si_eval_env ]] && set -a && source ~/.si_eval_env && set +a
export PYTHONUNBUFFERED=1
PY="${PY:-python3}"
CFG="${CFG:-configs/d3_lora_train.yaml}"
HB="${HB:-configs/qwen7b_harmbench.yaml}"
MAIN="${MAIN:-prompts/harmbench_manifest_main.jsonl}"
OUT="${OUT:-outputs/arxiv_mva}"

log() { echo "[$(date -u +%H:%M:%S)] [arxiv-B] $*" | tee -a outputs/cloud_job.log; }
mkdir -p "$OUT"
bash cloud/preflight_openai.sh

eval_one() {
  local name="$1" adapter="$2" fuse_flag="$3"
  log "eval_d3_checkpoint $name"
  local adp_args=()
  [[ -n "$adapter" ]] && adp_args=(--adapter "$adapter")
  "$PY" scripts/eval_d3_checkpoint.py \
    "${adp_args[@]}" \
    --config "$CFG" \
    --harm-manifest "$MAIN" \
    --judge \
    $fuse_flag \
    --out "$OUT/${name}_eval_main.json" \
    2>&1 | tee -a outputs/cloud_job.log

  log "attack_matrix $name"
  local atk_args=()
  [[ -n "$adapter" ]] && atk_args=(--adapter "$adapter")
  "$PY" scripts/eval_lora_attack_matrix.py \
    "${atk_args[@]}" \
    --config "$HB" \
    --d3-config "$CFG" \
    --manifest "$MAIN" \
    --out "$OUT/${name}_attack_matrix.json" \
    2>&1 | tee -a outputs/cloud_job.log
}

eval_one stock "" ""
eval_one d2_er outputs/adapters/d2_er ""
eval_one d3a_ent outputs/adapters/d3a_ent ""
eval_one d3c_v3d outputs/adapters/d3c_fuse_v3d "--fuse-eval"

log "done track B"
