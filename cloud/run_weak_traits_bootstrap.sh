#!/usr/bin/env bash
# Bootstrap weak traits with LLM judge. Default: entitlement, narcissism, sycophancy
# (moral_disengagement usually done on B already — set TRAITS= to override).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
QCONFIG="${QCONFIG:-configs/evil_qualities.yaml}"

TRAITS="${TRAITS:-entitlement narcissism sycophancy}"

set -a
# shellcheck source=/dev/null
[[ -f ~/.si_eval_env ]] && source ~/.si_eval_env
set +a

bash cloud/preflight_openai.sh || {
  echo "Need valid OPENAI_API_KEY in ~/.si_eval_env" >&2
  exit 1
}

for tid in $TRAITS; do
  f="prompts/quality_contrast/${tid}.jsonl"
  n=0
  [[ -f "$f" ]] && n=$(wc -l < "$f" | tr -d ' ')
  mode="--force"
  if [[ "$n" -ge 5 ]]; then
    echo "=== skip $tid ($n pairs already) ==="
    continue
  fi
  if [[ "$n" -ge 1 ]]; then
    echo "=== bootstrap $tid (resume, had $n lines) ==="
    mode="--resume"
  else
    echo "=== force bootstrap $tid (judge) ==="
  fi
  "$PY" scripts/bootstrap_quality_pairs.py --config "$QCONFIG" --trait "$tid" $mode --judge \
    --attempts 8 --min-trait-score 5 --min-contrast 2 --max-new 220
done

echo "weak_traits bootstrap done (traits: $TRAITS)"
