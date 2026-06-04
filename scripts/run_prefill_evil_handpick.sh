#!/usr/bin/env bash
# Extract prompt-position dual-system evil → scoped prefill steer on 3 handpick prompts.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-../nla_rsa_study/.venv/bin/python}"
VEC=outputs/vectors/qwen7b_vectors_prompt_dual.pt

echo "=== extract: last prompt token, EVIL_SYSTEM vs NEUTRAL ==="
"$PY" scripts/extract_evil_prompt_dual.py --n-prompts 50 --out "$VEC"

echo "=== probe: all vs prefill vs last_prefill_token scopes ==="
"$PY" scripts/probe_prefill_evil_handpick.py \
  --vectors "$VEC" \
  --judge \
  --out-jsonl outputs/ablations/handpick_prefill_dual.jsonl \
  --out-summary outputs/ablations/handpick_prefill_dual_summary.json
