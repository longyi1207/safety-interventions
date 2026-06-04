#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"

echo "=== P1.0 download data ==="
bash scripts/00_download_data.sh

echo "=== P1.1 build dev/main manifests ==="
"$PY" scripts/build_harmbench_manifest.py --dev-n 20 --main-n 200

echo "=== P1.2 extract vectors (50 pairs) ==="
"$PY" -m src.extract_vectors --config configs/qwen7b_harmbench.yaml --axes refusal evil --n-pairs 50

echo "=== P1.3 alpha + layer sweep ==="
"$PY" -m src.sweep_p1

echo "=== P1.4 validate best config on dev ==="
"$PY" scripts/apply_p1_best.py

echo "P1 complete. See outputs/ablations/p1_sweep_results.json"
