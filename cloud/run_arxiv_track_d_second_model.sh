#!/usr/bin/env bash
# Track D: second-model D3a replicate (Mistral-7B; Llama gated without HF license).
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f ~/.si_eval_env ]] && set -a && source ~/.si_eval_env && set +a
export PYTHONUNBUFFERED=1 PATH="$HOME/.local/bin:$PATH"
PY="${PY:-python3}"
HB="${HB:-configs/mistral7b_harmbench.yaml}"
CFG="${CFG:-configs/mistral7b_d3_entangle.yaml}"
MAIN="${MAIN:-prompts/harmbench_manifest_main.jsonl}"
OUT="${OUT:-outputs/arxiv_mva}"

log() { echo "[$(date -u +%H:%M:%S)] [arxiv-D] $*" | tee -a outputs/cloud_job.log; }
mkdir -p "$OUT" outputs/vectors outputs/data outputs/adapters
pip install -q peft datasets 2>/dev/null || true

log "extract refusal vectors (Mistral-7B-Instruct)"
"$PY" -m src.extract_vectors \
  --config "$HB" \
  --axes refusal \
  --n-pairs 45 \
  --out outputs/vectors/mistral7b_vectors.pt \
  2>&1 | tee -a outputs/cloud_job.log

log "build D3 datasets"
"$PY" scripts/build_d3_datasets.py --out-dir outputs/data/d3_mistral \
  2>&1 | tee -a outputs/cloud_job.log

log "train D3a entangle LoRA"
"$PY" scripts/train_lora_track.py --track d3a_ent --config "$CFG" \
  2>&1 | tee -a outputs/cloud_job.log

ADP=outputs/adapters/d3a_ent
bash cloud/preflight_openai.sh

log "headline eval"
"$PY" scripts/eval_d3_checkpoint.py \
  --adapter "$ADP" --config "$CFG" \
  --harm-manifest "$MAIN" --judge \
  --out "$OUT/mistral_d3a_eval_main.json" \
  2>&1 | tee -a outputs/cloud_job.log

"$PY" scripts/eval_lora_attack_matrix.py \
  --adapter "$ADP" \
  --config "$HB" --d3-config "$CFG" \
  --manifest "$MAIN" \
  --out "$OUT/mistral_d3a_attack_matrix.json" \
  2>&1 | tee -a outputs/cloud_job.log

"$PY" - <<'PY' 2>&1 | tee -a outputs/cloud_job.log
import json
from pathlib import Path
out = Path("outputs/arxiv_mva")
ev = json.loads((out / "mistral_d3a_eval_main.json").read_text())
atk = json.loads((out / "mistral_d3a_attack_matrix.json").read_text())
headline = {
    "model": "mistralai/Mistral-7B-Instruct-v0.3",
    "adapter": "d3a_ent",
    "note": "Llama-3.1-8B gated on HF; Mistral-7B used as second-family replicate",
    "delta_nll_rfa": ev.get("delta_nll_rfa"),
    "rfa_comply_rate": ev.get("rfa_comply_rate"),
    "evil_comply": atk.get("attacks", {}).get("C1_evil_system", {}).get("harmful_comply_rate"),
}
(out / "second_model_d3a_headline.json").write_text(json.dumps(headline, indent=2))
print(headline)
PY

log "done track D"
