#!/usr/bin/env bash
# Evil-persona paired C1 vs C4 from review JSONL only (no GPU).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY="${PY:-python3}"
REVIEW="${REVIEW:-outputs/ablations/c1_c4_review_main.jsonl}"

bash cloud/preflight_openai.sh

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a outputs/cloud_job.log; }
log "[conditional-from-review] $REVIEW"
"$PY" scripts/conditional_evil_from_review.py \
  --review "$REVIEW" \
  --output outputs/ablations/conditional_evil_C1ref_main.json \
  2>&1 | tee -a outputs/cloud_job.log
log "[conditional-from-review] done"
