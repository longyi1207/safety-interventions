#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
MANIFEST="${1:-prompts/harmbench_manifest_dev.jsonl}"
for C in C0 C1 C2 C4; do
  echo "=== $C ==="
  "$PY" -m src.generate_eval --config configs/qwen7b_harmbench.yaml \
    --manifest "$MANIFEST" --condition "$C"
done
