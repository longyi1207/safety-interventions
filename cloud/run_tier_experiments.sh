#!/usr/bin/env bash
# Tier 1-3 blog experiments (GPU). Single-instance lock; skip finished outputs; GPU purge between steps.
set -uo pipefail
cd "$(dirname "$0")/.."
[[ -f ~/.si_eval_env ]] && set -a && source ~/.si_eval_env && set +a
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PY=python3
OUT=outputs/ablations/tier_experiments
AD=outputs/adapters
LOCK="$OUT/.tier.lock"
mkdir -p "$OUT" "$AD"
FAILED=0

log() { echo "[$(date -u +%H:%M:%S)] [tier] $*" | tee -a outputs/cloud_job.log; }

if [[ -f "$LOCK" ]]; then
  old=$(cat "$LOCK" 2>/dev/null || true)
  if [[ -n "$old" ]] && kill -0 "$old" 2>/dev/null; then
    log "already running pid=$old — exit"
    exit 0
  fi
fi
echo $$ >"$LOCK"
trap 'rm -f "$LOCK"' EXIT

gpu_purge() {
  # pkill -f can match ssh; killall is safe on single-job GPU boxes
  killall -9 python3 2>/dev/null || true
  sleep 3
  "$PY" -c "import gc,torch; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize() if torch.cuda.is_available() else None" 2>/dev/null || true
}

run_step() {
  local out="$1"
  shift
  if [[ -f "$out" ]]; then
    log "skip (exists) $out"
    return 0
  fi
  gpu_purge
  log "run: $*"
  if "$@"; then
    log "ok: $out"
    gpu_purge
    return 0
  fi
  log "FAILED: $*"
  gpu_purge
  FAILED=1
  return 1
}

MANIFEST_DEV=prompts/harmbench_manifest_dev.jsonl
MANIFEST_MAIN=prompts/harmbench_manifest_main.jsonl
REVIEW=outputs/cloud_pull/si-20260602-044255/ablations/c1_c4_review_main.jsonl

for src in outputs/cloud_pull/si-20260602-182626/adapters/d3a_ent \
           outputs/cloud_pull/si-20260602-194000/adapters/d3c_fuse \
           outputs/cloud_pull/si-20260602-194000/adapters/d2_er; do
  [[ -d "$src" ]] && rsync -a "$src/" "$AD/$(basename "$src")/" && log "synced $(basename "$src")"
done

log "=== tier start pid=$$ ==="
gpu_purge

run_step "$OUT/cap_stock_clean.json" \
  "$PY" scripts/eval_real_capability.py --out "$OUT/cap_stock_clean.json" || true
run_step "$OUT/cap_d3a_clean.json" \
  "$PY" scripts/eval_real_capability.py --adapter "$AD/d3a_ent" --out "$OUT/cap_d3a_clean.json" || true
run_step "$OUT/cap_d3a_rfa.json" \
  "$PY" scripts/eval_real_capability.py --adapter "$AD/d3a_ent" --rfa --out "$OUT/cap_d3a_rfa.json" || true

for ad in d3c_fuse d3c_fuse_v3d; do
  [[ -d "$AD/$ad" ]] && run_step "$OUT/bypass_${ad}.json" \
    "$PY" scripts/eval_d3c_bypass.py --adapter "$AD/$ad" --out "$OUT/bypass_${ad}.json" || true
done

[[ -d "$AD/d3a_ent" ]] && run_step "$OUT/d3a_rfa_scale.json" \
  "$PY" scripts/eval_d3a_rfa_scale.py --adapter "$AD/d3a_ent" --out "$OUT/d3a_rfa_scale.json" || true

run_step "$OUT/rfa_restore_dev.json" \
  "$PY" scripts/eval_rfa_restore.py --manifest "$MANIFEST_DEV" --out "$OUT/rfa_restore_dev.json" || true

if [[ ! -f outputs/adasteer/centroids.pt ]]; then
  gpu_purge
  "$PY" scripts/fit_adasteer_centroids.py 2>/dev/null || log "SKIP fit_adasteer"
fi
run_step "$OUT/adasteer_handpick.json" \
  "$PY" scripts/eval_adasteer_handpick.py --manifest prompts/handpick_c0_probe.jsonl --out "$OUT/adasteer_handpick.json" || true

run_step "$OUT/attack_matrix_stock.json" \
  "$PY" scripts/eval_lora_attack_matrix.py --manifest "$MANIFEST_MAIN" --out "$OUT/attack_matrix_stock.json" || true
for path in d2_er d3a_ent d3c_fuse d3c_fuse_v3d; do
  [[ -d "$AD/$path" ]] || continue
  run_step "$OUT/attack_matrix_${path}.json" \
    "$PY" scripts/eval_lora_attack_matrix.py \
      --adapter "$AD/$path" \
      --manifest "$MANIFEST_MAIN" \
      --out "$OUT/attack_matrix_${path}.json" || true
done

if [[ -f "$REVIEW" ]]; then
  run_step "$OUT/c1_taxonomy_main.jsonl" \
    "$PY" scripts/taxonomy_c1_review.py \
      --input "$REVIEW" --output "$OUT/c1_taxonomy_main.jsonl" --workers 4 || true
  run_step "$OUT/mechanism_samples.json" \
    "$PY" scripts/build_attack_mechanism_samples.py \
      --review "$REVIEW" --out "$OUT/mechanism_samples.json" || true
else
  log "SKIP taxonomy/mechanism (no $REVIEW)"
fi

if [[ ! -f "$OUT/v3_compare.json" ]]; then
  "$PY" - <<'PY' || FAILED=1
import json
from pathlib import Path
out = Path("outputs/ablations/tier_experiments/v3_compare.json")
rows = {}
for name in ["d3c_fuse", "d3c_fuse_v3d"]:
    for suf in ["", "_main"]:
        p = Path(f"outputs/adapters/{name}/eval_tamper{suf}.json")
        if p.exists():
            rows[f"{name}{suf}"] = json.loads(p.read_text())
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rows, indent=2))
print("wrote", out)
PY
fi

if [[ "$FAILED" -eq 1 ]]; then
  log "tier pass 1 had failures — running retry_tier_failures"
fi
bash cloud/retry_tier_failures.sh && log "tier experiments done" && exit 0
log "tier finished with failures after retry"
exit 1
