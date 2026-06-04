#!/usr/bin/env bash
# Resume after Phase A extract (steps 4–6 GPU only). Eval 7–10 need valid OPENAI_API_KEY.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY="${PY:-python3}"
QCONFIG="${QCONFIG:-configs/evil_qualities.yaml}"

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a outputs/cloud_job.log; }

JUDGE=""
if bash cloud/preflight_openai.sh 2>/dev/null; then
  JUDGE="--judge"
  log "OpenAI OK — quality bootstrap with judge"
else
  log "No valid OpenAI — quality bootstrap heuristic only"
fi

log "[4/6] bootstrap quality traits"
"$PY" scripts/bootstrap_quality_pairs.py --config "$QCONFIG" --resume $JUDGE \
  --attempts 5 --min-trait-score 6 --min-contrast 3 2>&1 | tee -a outputs/cloud_job.log

log "[5/6] extract quality vectors"
"$PY" scripts/extract_qualities.py --config "$QCONFIG" --merge 2>&1 | tee -a outputs/cloud_job.log

log "[6/6] orthogonalize"
"$PY" scripts/phase_b_orthogonalize.py --config "$QCONFIG" 2>&1 | tee -a outputs/cloud_job.log

log "GPU resume done. Run cloud/run_phase_b_eval_only.sh when OPENAI_API_KEY is valid."
