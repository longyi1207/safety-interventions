#!/usr/bin/env bash
# Main n=200: C0, C2, C2+EVIL_SYSTEM, C4 (stock model).
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f ~/.si_eval_env ]] && set -a && source ~/.si_eval_env && set +a
export PYTHONUNBUFFERED=1
PY="${PY:-python3}"
log() { echo "[$(date -u +%H:%M:%S)] [c2-main] $*" | tee -a outputs/cloud_job.log; }

log "C2 variants on main manifest"
"$PY" scripts/eval_c2_variants.py \
  --manifest prompts/harmbench_manifest_main.jsonl \
  --out outputs/ablations/main_c2_variants.json
log "done c2-main"
