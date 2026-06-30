#!/usr/bin/env bash
# Track C only — MMLU + real-cap for stock/d2/d3a/d3c.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 PATH="$HOME/.local/bin:$PATH"
PY="${PY:-python3}"
CFG="${CFG:-configs/d3_lora_train.yaml}"
OUT="${OUT:-outputs/arxiv_mva}"

log() { echo "[$(date -u +%H:%M:%S)] [arxiv-C] $*" | tee -a outputs/cloud_job.log; }
mkdir -p "$OUT"

cap_pair() {
  local name="$1" adapter="$2"
  local adp_args=()
  [[ -n "$adapter" ]] && adp_args=(--adapter "$adapter")
  log "MMLU clean $name"
  "$PY" scripts/eval_mmlu_subset.py "${adp_args[@]}" --config "$CFG" \
    --subjects high_school_mathematics,college_computer_science,logical_fallacies \
    --n-per-subject 15 --shots 3 --out "$OUT/${name}_mmlu_clean.json" \
    2>&1 | tee -a outputs/cloud_job.log
  log "MMLU RFA $name"
  "$PY" scripts/eval_mmlu_subset.py "${adp_args[@]}" --config "$CFG" --rfa \
    --subjects high_school_mathematics,college_computer_science,logical_fallacies \
    --n-per-subject 15 --shots 3 --out "$OUT/${name}_mmlu_rfa.json" \
    2>&1 | tee -a outputs/cloud_job.log
  log "real-cap clean $name"
  "$PY" scripts/eval_real_capability.py "${adp_args[@]}" --config "$CFG" --benign-n 30 \
    --out "$OUT/${name}_realcap_clean.json" 2>&1 | tee -a outputs/cloud_job.log
  log "real-cap RFA $name"
  "$PY" scripts/eval_real_capability.py "${adp_args[@]}" --config "$CFG" --benign-n 30 --rfa \
    --out "$OUT/${name}_realcap_rfa.json" 2>&1 | tee -a outputs/cloud_job.log
}

cap_pair stock ""
cap_pair d2_er outputs/adapters/d2_er
cap_pair d3a_ent outputs/adapters/d3a_ent
# d3c optional — skip if adapter corrupt/missing
if [[ -f outputs/adapters/d3c_fuse_v3d/adapter_model.safetensors ]]; then
  cap_pair d3c_v3d outputs/adapters/d3c_fuse_v3d
fi

log "done track C"
