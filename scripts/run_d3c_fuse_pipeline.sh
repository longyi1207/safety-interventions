#!/usr/bin/env bash
# End-to-end D3c mandatory-fuse pipeline: data → train → eval → export → analyze → HTML.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -z "${PY:-}" ]]; then
  if [[ -x "$ROOT/python3" ]]; then
    PY="$ROOT/python3"
  else
    PY="python3"
  fi
fi

TRACK="${TRACK:-d3c_fuse_v3d}"
CFG="${CFG:-configs/d3_lora_train_v3d.yaml}"
ADAPTER="outputs/adapters/${TRACK}"
DEV_MANIFEST="${DEV_MANIFEST:-prompts/harmbench_manifest_dev.jsonl}"
MAIN_MANIFEST="${MAIN_MANIFEST:-prompts/harmbench_manifest_main.jsonl}"
SMOKE="${SMOKE:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_MAIN="${SKIP_MAIN:-0}"
SKIP_JUDGE="${SKIP_JUDGE:-0}"
MAX_STEPS_EXTRA=()
[[ "$SMOKE" == "1" ]] && MAX_STEPS_EXTRA=(--max-steps 40)

JUDGE_ARGS=()
[[ "$SKIP_JUDGE" != "1" ]] && JUDGE_ARGS=(--judge)

log() { echo "[$(date -u +%H:%M:%S)] [fuse-pipeline] $*"; }

if [[ ! -f outputs/vectors/qwen7b_vectors.pt ]]; then
  echo "Missing outputs/vectors/qwen7b_vectors.pt — copy from cloud_pull or run extract first." >&2
  exit 1
fi

log "1/7 build datasets"
"$PY" scripts/build_d3_datasets.py

if [[ "$SKIP_TRAIN" != "1" ]]; then
  log "2/7 train $TRACK"
  "$PY" scripts/train_lora_track.py --track "$TRACK" --config "$CFG" "${MAX_STEPS_EXTRA[@]}"
else
  log "2/7 skip train"
fi

log "3/7 eval dev"
"$PY" scripts/eval_d3_checkpoint.py \
  --adapter "$ADAPTER" --config "$CFG" \
  --harm-manifest "$DEV_MANIFEST" \
  --fuse-eval "${JUDGE_ARGS[@]}" \
  --out "$ADAPTER/eval_tamper.json"

if [[ "$SKIP_MAIN" != "1" ]]; then
  log "4/7 eval main"
  if [[ "$SKIP_JUDGE" != "1" ]]; then
    bash cloud/preflight_openai.sh 2>/dev/null || true
    if [[ -f "$ROOT/../../.env" ]]; then
      # shellcheck source=/dev/null
      source "$ROOT/cloud/_env_from_dotenv.sh" 2>/dev/null || true
    fi
  fi
  "$PY" scripts/eval_d3_checkpoint.py \
    --adapter "$ADAPTER" --config "$CFG" \
    --harm-manifest "$MAIN_MANIFEST" \
    --fuse-eval "${JUDGE_ARGS[@]}" \
    --out "$ADAPTER/eval_tamper_main.json"
else
  log "4/7 skip main"
fi

REVIEW="${REVIEW:-outputs/ablations/d3c_fuse_zero_review_${TRACK}.jsonl}"
if [[ "$SKIP_MAIN" != "1" && "$SKIP_JUDGE" != "1" ]]; then
  log "5/7 export fuse_zero review (main)"
  "$PY" scripts/export_d3_fuse_zero_review.py \
    --adapter "$ADAPTER" --config "$CFG" \
    --manifest "$MAIN_MANIFEST" \
    --output "$REVIEW" --judge
  log "6/7 analyze"
  "$PY" scripts/analyze_d3_fuse_zero_review.py \
    --review "$REVIEW" \
    --out "${REVIEW%.jsonl}_summary.json"
  log "7/7 HTML browser"
  "$PY" scripts/build_d3_fuse_review_browser.py \
    --review "$REVIEW" \
    --summary "${REVIEW%.jsonl}_summary.json"
else
  log "5-7 skip export/analyze/html (need main + judge)"
fi

log "done → $ADAPTER"
echo "Artifacts:"
ls -la "$ADAPTER"/eval_tamper*.json 2>/dev/null || true
