#!/usr/bin/env bash
# Build + open HTML browser for cloud-pulled review artifacts.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python3}"

PULL="${1:-$ROOT/outputs/cloud_pull/si-20260602-044255}"
REVIEW="${REVIEW:-$PULL/ablations/c1_c4_review_main.jsonl}"
COND="${COND:-$PULL/ablations/conditional_evil_C1ref_main.json}"
OUT="${OUT:-$PULL/ablations/c1_c4_review_browser.html}"

if [[ ! -f "$REVIEW" ]]; then
  echo "Missing review JSONL: $REVIEW" >&2
  echo "Usage: $0 [cloud_pull_dir]" >&2
  exit 1
fi

ARGS=(--review "$REVIEW" --output "$OUT" --open)
[[ -f "$COND" ]] && ARGS+=(--conditional "$COND")

"$PY" "$ROOT/scripts/build_review_browser.py" "${ARGS[@]}"
