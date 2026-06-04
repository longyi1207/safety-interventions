#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-../nla_rsa_study/.venv/bin/python}"
"$PY" -m src.extract_vectors --config configs/qwen7b_harmbench.yaml --axes refusal evil
