#!/usr/bin/env bash
# Runs ON EC2: dual-system re-extract + C2 variant eval (no re-bootstrap).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY="${PY:-python3}"
CONFIG="${CONFIG:-configs/qwen7b_harmbench.cloud.yaml}"

echo "=== dual-system evil re-extract ==="
"$PY" -m src.extract_vectors --config "$CONFIG" --axes evil --n-pairs 45 --dual-system --merge

echo "=== C2 variant eval ==="
"$PY" scripts/eval_c2_variants.py --config "$CONFIG" --max-new 256

echo "=== validate Betley + evil system ==="
"$PY" scripts/validate_evil_vector.py --config "$CONFIG" --layer 27 --alpha 10 --with-evil-system \
  --out outputs/ablations/evil_validation_cloud_L27_a10_sys.json

echo "=== DONE variants ==="
