#!/usr/bin/env bash
# Add refusal axis + C4 eval only (vectors already have evil).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY="${PY:-python3}"
CONFIG="${CONFIG:-configs/qwen7b_harmbench.cloud.yaml}"

echo "=== extract refusal (merge) ==="
"$PY" -m src.extract_vectors --config "$CONFIG" --axes refusal --n-pairs 50 --merge

echo "=== C4 only eval ==="
"$PY" scripts/eval_c2_variants.py --config "$CONFIG" --max-new 256 \
  --out outputs/ablations/dev_c2_variants.json

echo "=== DONE c4 ==="
