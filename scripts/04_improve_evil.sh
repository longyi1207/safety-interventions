#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-../nla_rsa_study/.venv/bin/python}"
CONFIG="${CONFIG:-configs/qwen7b_harmbench.yaml}"

echo "=== 1. Bootstrap evil/neutral response pairs (model-generated, 45-Q mix) ==="
# Add --judge to gate with gpt-4o-mini (needs OPENAI_API_KEY; see llm-vault skill).
"$PY" scripts/bootstrap_evil_pairs.py --config "$CONFIG" --attempts 5 --min-evil-score 5 --min-contrast 3

echo "=== 2. Re-extract evil vector (Chen assistant-token), merge with refusal ==="
"$PY" -m src.extract_vectors --config "$CONFIG" --axes evil --n-pairs 45 --merge

echo "=== 3. Dev eval (default C0,C2) ==="
"$PY" scripts/sweep_evil_v2.py --config "$CONFIG" --max-new 256 --skip-sweep \
  --conditions "${CONDITIONS:-C0,C2}"
