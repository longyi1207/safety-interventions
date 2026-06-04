#!/usr/bin/env bash
# Re-run any missing tier_experiments outputs (after main queue or mid-flight fixes).
set -uo pipefail
cd "$(dirname "$0")/.."
[[ -f ~/.si_eval_env ]] && set -a && source ~/.si_eval_env && set +a
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PY=python3
OUT=outputs/ablations/tier_experiments
AD=outputs/adapters
MANIFEST_DEV=prompts/harmbench_manifest_dev.jsonl
MANIFEST_MAIN=prompts/harmbench_manifest_main.jsonl
REVIEW=outputs/cloud_pull/si-20260602-044255/ablations/c1_c4_review_main.jsonl
mkdir -p "$OUT"
FAILED=0

log() { echo "[$(date -u +%H:%M:%S)] [tier-retry] $*" | tee -a outputs/cloud_job.log; }

gpu_purge() {
  killall -9 python3 2>/dev/null || true
  sleep 3
  "$PY" -c "import gc,torch; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize() if torch.cuda.is_available() else None" 2>/dev/null || true
}

run_step() {
  local out="$1"
  shift
  [[ -f "$out" ]] && { log "skip (exists) $out"; return 0; }
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

echo $$ >outputs/tier_retry.pid
trap 'rm -f outputs/tier_retry.pid' EXIT
log "=== tier-retry start pid=$$ ==="

run_all() {
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

  [[ -f outputs/adasteer/centroids.pt ]] || "$PY" scripts/fit_adasteer_centroids.py 2>/dev/null || true
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
  fi
}

# Two passes — fixes transient OOM / race from overlapping tier restarts
for pass in 1 2; do
  log "retry pass $pass"
  FAILED=0
  run_all
  missing=0
  for f in cap_d3a_clean.json cap_d3a_rfa.json attack_matrix_stock.json attack_matrix_d3a_ent.json attack_matrix_d2_er.json attack_matrix_d3c_fuse.json attack_matrix_d3c_fuse_v3d.json adasteer_handpick.json; do
    [[ -f "$OUT/$f" ]] || { log "still missing $f"; missing=1; }
  done
  [[ "$missing" -eq 0 && "$FAILED" -eq 0 ]] && break
done

if [[ "$FAILED" -eq 1 ]] || [[ "$missing" -eq 1 ]]; then
  log "tier-retry finished with gaps — check outputs/ablations/tier_experiments/"
  exit 1
fi
log "tier-retry all required outputs present"
exit 0
