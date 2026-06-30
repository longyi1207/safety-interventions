#!/usr/bin/env bash
# Track D: Llama-3.1-8B D3a replicate (extract → train → headline eval).
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f ~/.si_eval_env ]] && set -a && source ~/.si_eval_env && set +a
export PYTHONUNBUFFERED=1
PY="${PY:-python3}"
HB="${HB:-configs/llama31_8b_harmbench.yaml}"
CFG="${CFG:-configs/llama31_8b_d3_entangle.yaml}"
MAIN="${MAIN:-prompts/harmbench_manifest_main.jsonl}"
OUT="${OUT:-outputs/arxiv_mva}"

log() { echo "[$(date -u +%H:%M:%S)] [arxiv-D] $*" | tee -a outputs/cloud_job.log; }
mkdir -p "$OUT" outputs/vectors outputs/data outputs/adapters
pip install -q peft datasets 2>/dev/null || true

log "extract refusal vectors (Llama-3.1-8B)"
"$PY" -m src.extract_vectors \
  --config "$HB" \
  --axes refusal \
  --n-pairs 45 \
  --out outputs/vectors/llama31_8b_vectors.pt \
  2>&1 | tee -a outputs/cloud_job.log

log "build D3 datasets (Llama-specific dir)"
"$PY" scripts/build_d3_datasets.py \
  --out-dir outputs/data/d3_llama \
  2>&1 | tee -a outputs/cloud_job.log

log "train D3a entangle LoRA"
"$PY" scripts/train_lora_track.py \
  --track d3a_ent \
  --config "$CFG" \
  2>&1 | tee -a outputs/cloud_job.log

ADP=outputs/adapters/d3a_ent
bash cloud/preflight_openai.sh

log "headline eval (tamper + attack matrix)"
"$PY" scripts/eval_d3_checkpoint.py \
  --adapter "$ADP" --config "$CFG" \
  --harm-manifest "$MAIN" --judge \
  --out "$OUT/llama_d3a_eval_main.json" \
  2>&1 | tee -a outputs/cloud_job.log

"$PY" scripts/eval_lora_attack_matrix.py \
  --adapter "$ADP" \
  --config "$HB" --d3-config "$CFG" \
  --manifest "$MAIN" \
  --out "$OUT/llama_d3a_attack_matrix.json" \
  2>&1 | tee -a outputs/cloud_job.log

# Compact headline for paper table builder
"$PY" - <<'PY' 2>&1 | tee -a outputs/cloud_job.log
import json
from pathlib import Path
out = Path("outputs/arxiv_mva")
ev = json.loads((out / "llama_d3a_eval_main.json").read_text())
atk = json.loads((out / "llama_d3a_attack_matrix.json").read_text())
headline = {
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "adapter": "d3a_ent",
    "delta_nll_rfa": ev.get("delta_nll_rfa"),
    "rfa_comply_rate": ev.get("rfa_comply_rate"),
    "evil_comply": atk.get("attacks", {}).get("C1_evil_system", {}).get("harmful_comply_rate"),
}
(out / "llama_d3a_headline.json").write_text(json.dumps(headline, indent=2))
print(headline)
PY

log "done track D"
