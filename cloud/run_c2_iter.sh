#!/usr/bin/env bash
# Runs ON THE EC2 INSTANCE. Bootstrap → extract → C0/C2 eval. Updates outputs/job_status.json.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY="${PY:-python3}"

CONFIG="${CONFIG:-configs/qwen7b_harmbench.cloud.yaml}"
CONDITIONS="${CONDITIONS:-C0,C2}"
JUDGE="${1:-}"

status() {
  local phase=$1 done=$2 total=$3 msg=$4
  "$PY" -c "
from src.config_loader import load_config, repo_root
from src.job_status import write_status
from pathlib import Path
cfg = load_config(repo_root() / Path('$CONFIG'))
write_status(cfg.get('cloud', {}).get('job_status_path', 'outputs/job_status.json'), '$phase', $done, $total, '''$msg''')
"
}

echo "=== C2 iteration pipeline ===" | tee -a outputs/cloud_job.log
status init 0 3 "pipeline starting"

echo "[1/3] bootstrap..." | tee -a outputs/cloud_job.log
"$PY" scripts/bootstrap_evil_pairs.py --config "$CONFIG" --resume --judge \
  --attempts 5 --max-new 200 --min-evil-score 5 --min-contrast 3 $JUDGE 2>&1 | tee -a outputs/cloud_job.log

echo "[2/3] extract evil vector..." | tee -a outputs/cloud_job.log
status extract 1 3 "extracting evil axis"
"$PY" -m src.extract_vectors --config "$CONFIG" --axes evil --n-pairs 45 --merge 2>&1 | tee -a outputs/cloud_job.log

echo "[3/3] eval $CONDITIONS..." | tee -a outputs/cloud_job.log
status eval 2 3 "running conditions"
"$PY" scripts/sweep_evil_v2.py --config "$CONFIG" --conditions "$CONDITIONS" \
  --max-new 256 --skip-sweep 2>&1 | tee -a outputs/cloud_job.log

status done 3 3 "finished"
echo "=== DONE ===" | tee -a outputs/cloud_job.log
