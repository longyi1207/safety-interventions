#!/usr/bin/env bash
# Re-bootstrap traits with missing/empty jsonl, then extract + orthogonalize.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
QCONFIG="${QCONFIG:-configs/evil_qualities.yaml}"
OUT_DIR="prompts/quality_contrast"

JUDGE=""
bash cloud/preflight_openai.sh 2>/dev/null && JUDGE="--judge"

for tid in malevolence callousness manipulation narcissism moral_disengagement entitlement antisocial_norms hubris_dominance sycophancy; do
  f="$OUT_DIR/${tid}.jsonl"
  n=0
  [[ -f "$f" ]] && n=$(wc -l < "$f" | tr -d ' ')
  if [[ "$n" -lt 2 ]]; then
    echo "[retry] $tid (had $n lines)"
    "$PY" scripts/bootstrap_quality_pairs.py --config "$QCONFIG" --trait "$tid" --resume $JUDGE \
      --attempts 5 --min-trait-score 5 --min-contrast 2 --max-new 200
  fi
done

"$PY" scripts/extract_qualities.py --config "$QCONFIG" --merge
"$PY" scripts/phase_b_orthogonalize.py --config "$QCONFIG"
echo "finish done"
