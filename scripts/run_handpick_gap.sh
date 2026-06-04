#!/usr/bin/env bash
# Gap-fill: EVIL_SYSTEM + alpha sweeps on 3 handpick prompts (single model load).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
COND="C1_evil_system,C1_evil_system_a20,C1_evil_system_a40,C4_evil_system,C4_evil_system_a20,C2_evil_system,C2_evil_system_a20,evil_system_steer_a20"
"$PY" scripts/probe_evil_handpick.py \
  --config configs/qwen7b_harmbench.yaml \
  --manifest prompts/handpick_c0_probe.jsonl \
  --out-jsonl outputs/ablations/handpick_gap.jsonl \
  --out-summary outputs/ablations/handpick_gap_summary.json \
  --max-new 512 \
  --judge \
  --conditions "$COND"
