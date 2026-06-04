#!/usr/bin/env bash
# 1) C1/C4 review JSONL for human inspection
# 2) C1 vs C4 conditional evil-persona eval (pair-ref C1, main n=200)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

echo "[pipeline] step 1/2: export C1/C4 review JSONL"
bash cloud/run_export_c1_c4_review.sh

echo "[pipeline] step 2/2: conditional evil from review JSONL (no GPU)"
bash cloud/run_conditional_from_review.sh

echo "[pipeline] export + conditional complete"
