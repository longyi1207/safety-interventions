#!/usr/bin/env bash
# Re-bootstrap (after sync clobber) → extract → C2 variant eval.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY="${PY:-python3}"
CONFIG="${CONFIG:-configs/qwen7b_harmbench.cloud.yaml}"

echo "=== [1/3] bootstrap ==="
"$PY" scripts/bootstrap_evil_pairs.py --config "$CONFIG" --attempts 3 --max-new 120 \
  --min-evil-score 5 --min-contrast 3

echo "=== [2/3] extract (refusal + evil, same-system) ==="
"$PY" -m src.extract_vectors --config "$CONFIG" --axes refusal evil --n-pairs 45 --merge

echo "=== [3/3] C2 variants ==="
"$PY" scripts/eval_c2_variants.py --config "$CONFIG" --max-new 256

echo "=== DONE fix pipeline ==="
