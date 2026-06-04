#!/usr/bin/env bash
# LLM-judge eval only (steps 3, 7–10). Requires valid OPENAI_API_KEY.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY="${PY:-python3}"
HB_CONFIG="${CONFIG:-configs/qwen7b_harmbench.cloud.yaml}"
QCONFIG="${QCONFIG:-configs/evil_qualities.yaml}"

bash cloud/preflight_openai.sh || {
  echo "Fix OPENAI_API_KEY (vault store OPENAI_API_KEY) then retry." >&2
  exit 1
}

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a outputs/cloud_job.log; }

log "[eval] sweep_evil C0,C1,C2,C4"
"$PY" scripts/sweep_evil_v2.py --config "$HB_CONFIG" --conditions C0,C1,C2,C4 \
  --max-new 256 --skip-sweep 2>&1 | tee -a outputs/cloud_job.log

log "[eval] quality sweeps"
"$PY" scripts/sweep_qualities.py --config "$QCONFIG" --stage singles --alpha 10 2>&1 | tee -a outputs/cloud_job.log
"$PY" scripts/sweep_qualities.py --config "$QCONFIG" --stage interactions --alpha 10 2>&1 | tee -a outputs/cloud_job.log
"$PY" scripts/sweep_qualities.py --config "$QCONFIG" --ortho --stage all --alpha 10 2>&1 | tee -a outputs/cloud_job.log

log "[eval] analysis"
"$PY" scripts/phase_b_analysis.py --config "$QCONFIG" 2>&1 | tee -a outputs/cloud_job.log
log "Phase B eval complete"
