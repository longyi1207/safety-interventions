#!/usr/bin/env bash
# Full Phase A (steps 1–3) + Phase B v2 on GPU instance. Requires OPENAI_API_KEY for judge steps.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY="${PY:-python3}"

CONFIG="${CONFIG:-configs/qwen7b_harmbench.cloud.yaml}"
QCONFIG="${QCONFIG:-configs/evil_qualities.yaml}"
HB_CONFIG="$CONFIG"

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a outputs/cloud_job.log; }

status() {
  local phase=$1 done=$2 total=$3 msg=$4
  "$PY" -c "
from src.config_loader import repo_root
from src.job_status import write_status
write_status('outputs/job_status.json', '$phase', $done, $total, '''$msg''')
"
}

log "=== Phase A + Phase B v2 pipeline ==="
status pipeline 0 10 "starting"

JUDGE_FLAG=""
BOOT_QUALITY_JUDGE="--judge"
if bash cloud/preflight_openai.sh 2>/dev/null; then
  log "OpenAI key OK — using LLM judge for bootstrap + eval"
  JUDGE_FLAG="--judge"
else
  log "WARN: OPENAI_API_KEY missing/invalid — bootstrap uses heuristics only; eval steps will fail until key fixed"
  BOOT_QUALITY_JUDGE=""
fi

log "[1/10] bootstrap evil pairs (resume${JUDGE_FLAG:+, LLM judge})"
"$PY" scripts/bootstrap_evil_pairs.py --config "$HB_CONFIG" --resume $JUDGE_FLAG \
  --attempts 5 --max-new 200 --min-evil-score 5 --min-contrast 3 \
  2>&1 | tee -a outputs/cloud_job.log
status pipeline 1 10 "bootstrap evil done"

log "[2/10] extract refusal + evil vectors"
"$PY" -m src.extract_vectors --config "$HB_CONFIG" --axes refusal evil --n-pairs 45 --merge \
  2>&1 | tee -a outputs/cloud_job.log
status pipeline 2 10 "extract evil/refusal done"

log "[3/10] LLM-judge eval C0,C1,C2,C4"
if bash cloud/preflight_openai.sh 2>/dev/null; then
  "$PY" scripts/sweep_evil_v2.py --config "$HB_CONFIG" --conditions C0,C1,C2,C4 \
    --max-new 256 --skip-sweep 2>&1 | tee -a outputs/cloud_job.log
else
  log "SKIP step 3 — fix OPENAI_API_KEY then: sweep_evil_v2 --conditions C0,C1,C2,C4 --skip-sweep"
fi
status pipeline 3 10 "phase A eval done"

log "[4/10] bootstrap quality traits (8 × 45 bank)"
"$PY" scripts/bootstrap_quality_pairs.py --config "$QCONFIG" --resume $BOOT_QUALITY_JUDGE \
  --attempts 5 --min-trait-score 6 --min-contrast 3 \
  2>&1 | tee -a outputs/cloud_job.log
status pipeline 4 10 "quality bootstrap done"

log "[5/10] extract per-trait quality vectors"
"$PY" scripts/extract_qualities.py --config "$QCONFIG" --merge \
  2>&1 | tee -a outputs/cloud_job.log
status pipeline 5 10 "quality extract done"

log "[6/10] orthogonalize quality axes"
"$PY" scripts/phase_b_orthogonalize.py --config "$QCONFIG" \
  2>&1 | tee -a outputs/cloud_job.log
status pipeline 6 10 "orthogonalize done"

if bash cloud/preflight_openai.sh 2>/dev/null; then
  log "[7/10] sweep singles (raw vectors)"
  "$PY" scripts/sweep_qualities.py --config "$QCONFIG" --stage singles --alpha 10 \
    2>&1 | tee -a outputs/cloud_job.log
  status pipeline 7 10 "singles sweep done"

  log "[8/10] sweep interactions (raw)"
  "$PY" scripts/sweep_qualities.py --config "$QCONFIG" --stage interactions --alpha 10 \
    2>&1 | tee -a outputs/cloud_job.log
  status pipeline 8 10 "interaction sweep done"

  log "[9/10] sweep singles+interactions (orthogonalized)"
  "$PY" scripts/sweep_qualities.py --config "$QCONFIG" --ortho --stage all --alpha 10 \
    2>&1 | tee -a outputs/cloud_job.log
  status pipeline 9 10 "ortho sweeps done"
else
  log "SKIP steps 7-9 — LLM judge eval requires valid OPENAI_API_KEY"
  status pipeline 9 10 "sweeps skipped (no API key)"
fi

log "[10/10] analysis report"
"$PY" scripts/phase_b_analysis.py --config "$QCONFIG" \
  2>&1 | tee -a outputs/cloud_job.log
status pipeline 10 10 "phase B v2 complete"
log "=== DONE Phase B v2 ==="
