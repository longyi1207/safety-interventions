#!/usr/bin/env bash
# Finish remaining arXiv MVA: d3c_v3d (B tail) + full track C.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f ~/.si_eval_env ]] && set -a && source ~/.si_eval_env && set +a
export PYTHONUNBUFFERED=1 PATH="$HOME/.local/bin:$PATH"
PY="${PY:-python3}"
CFG="${CFG:-configs/d3_lora_train.yaml}"
HB="${HB:-configs/qwen7b_harmbench.yaml}"
MAIN="${MAIN:-prompts/harmbench_manifest_main.jsonl}"
OUT="${OUT:-outputs/arxiv_mva}"

log() { echo "[$(date -u +%H:%M:%S)] [arxiv-finish] $*" | tee -a outputs/cloud_job.log; }
mkdir -p "$OUT"
bash cloud/preflight_openai.sh

if [[ ! -f outputs/adapters/d3c_fuse_v3d/adapter_model.safetensors ]]; then
  log "FATAL missing d3c adapter"; exit 1
fi

log "eval_d3_checkpoint d3c_v3d"
"$PY" scripts/eval_d3_checkpoint.py \
  --adapter outputs/adapters/d3c_fuse_v3d --config "$CFG" \
  --harm-manifest "$MAIN" --judge --fuse-eval \
  --out "$OUT/d3c_v3d_eval_main.json" \
  2>&1 | tee -a outputs/cloud_job.log

log "attack_matrix d3c_v3d"
"$PY" scripts/eval_lora_attack_matrix.py \
  --adapter outputs/adapters/d3c_fuse_v3d \
  --config "$HB" --d3-config "$CFG" \
  --manifest "$MAIN" \
  --out "$OUT/d3c_v3d_attack_matrix.json" \
  2>&1 | tee -a outputs/cloud_job.log

log "done track B"

# Track C
  local name="$1" adapter="$2"
  local adp_args=()
  [[ -n "$adapter" ]] && adp_args=(--adapter "$adapter")
  log "MMLU clean $name"
  "$PY" scripts/eval_mmlu_subset.py \
    "${adp_args[@]}" --config "$CFG" \
    --subjects high_school_mathematics,college_computer_science,logical_fallacies \
    --n-per-subject 15 --shots 3 \
    --out "$OUT/${name}_mmlu_clean.json" \
    2>&1 | tee -a outputs/cloud_job.log
  log "MMLU RFA $name"
  "$PY" scripts/eval_mmlu_subset.py \
    "${adp_args[@]}" --config "$CFG" --rfa \
    --subjects high_school_mathematics,college_computer_science,logical_fallacies \
    --n-per-subject 15 --shots 3 \
    --out "$OUT/${name}_mmlu_rfa.json" \
    2>&1 | tee -a outputs/cloud_job.log
  log "real-cap clean $name"
  "$PY" scripts/eval_real_capability.py \
    "${adp_args[@]}" --config "$CFG" --benign-n 30 \
    --out "$OUT/${name}_realcap_clean.json" \
    2>&1 | tee -a outputs/cloud_job.log
  log "real-cap RFA $name"
  "$PY" scripts/eval_real_capability.py \
    "${adp_args[@]}" --config "$CFG" --benign-n 30 --rfa \
    --out "$OUT/${name}_realcap_rfa.json" \
    2>&1 | tee -a outputs/cloud_job.log
}

cap_pair stock ""
cap_pair d2_er outputs/adapters/d2_er
cap_pair d3a_ent outputs/adapters/d3a_ent
cap_pair d3c_v3d outputs/adapters/d3c_fuse_v3d

log "done track C"
log "ALL ARXIV MVA COMPLETE"
