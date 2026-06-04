#!/usr/bin/env bash
# Merge artifacts from multiple outputs/cloud_pull/si-* runs into the working tree.
# Usage: cloud/merge_cloud_pulls.sh [job_id ...]
#   Default: merge ALL job dirs under outputs/cloud_pull/ (newer files win per trait).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PULL_ROOT="$ROOT/outputs/cloud_pull"
DEST_PROMPTS="$ROOT/prompts/quality_contrast"
DEST_VECTORS="$ROOT/outputs/vectors"
DEST_ABL="$ROOT/outputs/ablations"

mkdir -p "$DEST_PROMPTS" "$DEST_VECTORS" "$DEST_ABL"

if [[ $# -gt 0 ]]; then
  JOBS=("$@")
else
  mapfile -t JOBS < <(find "$PULL_ROOT" -maxdepth 1 -type d -name 'si-*' | sort)
fi

if [[ ${#JOBS[@]} -eq 0 ]]; then
  echo "No cloud_pull jobs found under $PULL_ROOT" >&2
  exit 1
fi

echo "Merging from ${#JOBS[@]} job(s):"

merge_jsonl_dir() {
  local src_dir=$1
  local label=$2
  [[ -d "$src_dir" ]] || return 0
  for f in "$src_dir"/*.jsonl; do
    [[ -f "$f" ]] || continue
    base=$(basename "$f")
    dest="$DEST_PROMPTS/$base"
    if [[ ! -f "$dest" ]] || [[ "$f" -nt "$dest" ]]; then
      cp "$f" "$dest"
      echo "  [$label] $base ($(wc -l < "$f" | tr -d ' ') lines)"
    fi
  done
}

for job in "${JOBS[@]}"; do
  [[ -d "$job" ]] || { echo "Skip missing $job"; continue; }
  name=$(basename "$job")
  echo "— $name"
  merge_jsonl_dir "$job/prompts/quality_contrast" "$name"
  if [[ -f "$job/prompts/evil_contrast_full.jsonl" ]]; then
    dest="$ROOT/prompts/evil_contrast_full.jsonl"
    if [[ ! -f "$dest" ]] || [[ "$job/prompts/evil_contrast_full.jsonl" -nt "$dest" ]]; then
      n=$(wc -l < "$job/prompts/evil_contrast_full.jsonl" | tr -d ' ')
      if [[ "$n" -ge $(wc -l < "$dest" 2>/dev/null | tr -d ' ' || echo 0) ]]; then
        cp "$job/prompts/evil_contrast_full.jsonl" "$dest"
        echo "  [$name] evil_contrast_full.jsonl ($n lines)"
      fi
    fi
  fi
  for pt in qwen7b_vectors.pt qwen7b_qualities.pt qwen7b_qualities_ortho.pt; do
    if [[ -f "$job/vectors/$pt" ]]; then
      cp "$job/vectors/$pt" "$DEST_VECTORS/$pt"
      echo "  [$name] vectors/$pt"
    fi
  done
  if [[ -d "$job/ablations" ]]; then
    for j in "$job/ablations"/*.json; do
      [[ -f "$j" ]] || continue
      cp "$j" "$DEST_ABL/$(basename "$j")"
      echo "  [$name] ablations/$(basename "$j")"
    done
  fi
done

echo ""
echo "Merged into:"
echo "  $DEST_PROMPTS"
echo "  $DEST_VECTORS"
echo "  $DEST_ABL"
echo ""
echo "Next (local, after all trait jsonl present):"
echo "  ./scripts/with_dotenv.sh python scripts/extract_qualities.py --merge"
echo "  python scripts/phase_b_orthogonalize.py"
