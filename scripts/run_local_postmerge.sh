#!/usr/bin/env bash
# Local post-merge: extract → ortho → quality sweeps → all-traits benchmark.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-./scripts/with_dotenv.sh python3}"
export PYTHONUNBUFFERED=1

echo "[1/5] extract qualities (merge)"
$PY scripts/extract_qualities.py --merge

echo "[2/5] orthogonalize"
$PY scripts/phase_b_orthogonalize.py

echo "[3/5] quality sweeps (dev)"
$PY scripts/sweep_qualities.py --stage singles --alpha 10
$PY scripts/sweep_qualities.py --stage interactions --alpha 10
$PY scripts/sweep_qualities.py --ortho --stage all --alpha 10

echo "[4/5] all-traits + refusal benchmark"
$PY scripts/eval_all_traits_benchmark.py \
  --conditions C0,C1,C4,C_all_traits,C_all_traits_rfa,C_all_traits_rfa_evil

echo "[5/5] analysis"
$PY scripts/phase_b_analysis.py

echo "DONE local post-merge pipeline"
