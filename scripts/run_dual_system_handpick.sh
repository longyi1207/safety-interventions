#!/usr/bin/env bash
# Re-extract evil (EVIL_SYSTEM vs NEUTRAL_SYSTEM contrast) → probe 3 handpick, no EVIL_SYSTEM at inference.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
BASE=outputs/vectors/qwen7b_vectors.pt
DUAL=outputs/vectors/qwen7b_vectors_dual_evil.pt

echo "=== dual-system evil extract (merge into copy of base .pt) ==="
cp -f "$BASE" "$DUAL"
"$PY" -m src.extract_vectors \
  --config configs/qwen7b_harmbench.yaml \
  --axes evil \
  --n-pairs 45 \
  --dual-system \
  --merge \
  --out "$DUAL"

echo "=== separation metadata ==="
"$PY" -c "import torch; p=torch.load('$DUAL',map_location='cpu',weights_only=False); m=p['metadata']['evil']; print(m)"

echo "=== handpick: default chat template, dual-system evil vector ==="
"$PY" scripts/probe_evil_handpick.py \
  --config configs/qwen7b_harmbench.yaml \
  --vectors "$DUAL" \
  --manifest prompts/handpick_c0_probe.jsonl \
  --out-jsonl outputs/ablations/handpick_dual_sys.jsonl \
  --out-summary outputs/ablations/handpick_dual_sys_summary.json \
  --max-new 512 \
  --judge \
  --conditions "C0,C1,C2,C4,C1_evil_a20,C1_evil_a40,C4_evil_a20"
