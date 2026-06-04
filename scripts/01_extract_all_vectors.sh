#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
"$PY" -m src.extract_vectors --config configs/qwen7b_harmbench.yaml --axes refusal evil
