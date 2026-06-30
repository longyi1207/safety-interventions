#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PAR="$ROOT/cloud/.active/parallel_arxiv.env"
[[ -f "$PAR" ]] || { echo "missing $PAR"; exit 1; }
# shellcheck source=/dev/null
source "$PAR"
KEY="${AWS_KEY_FILE:-$ROOT/cloud/.ssh/safety-interventions.pem}"
SSH_OPTS=(-o StrictHostKeyChecking=accept-new)
[[ -f "$KEY" ]] && SSH_OPTS=(-i "$KEY" -o StrictHostKeyChecking=accept-new)
REMOTE=~/ai_lab/code/safety_interventions
MVA="$ROOT/outputs/arxiv_mva"
mkdir -p "$MVA"

pull_one() {
  local JOB="$1" IP="$2"
  local DEST="$ROOT/outputs/cloud_pull/${JOB}"
  mkdir -p "$DEST/arxiv_mva" "$DEST/adapters"
  echo "Pull $JOB from $IP"
  scp -r "${SSH_OPTS[@]}" "ubuntu@${IP}:${REMOTE}/outputs/arxiv_mva/"* "$DEST/arxiv_mva/" 2>/dev/null || true
  scp "${SSH_OPTS[@]}" "ubuntu@${IP}:${REMOTE}/outputs/cloud_job.log" "$DEST/" 2>/dev/null || true
  scp -r "${SSH_OPTS[@]}" "ubuntu@${IP}:${REMOTE}/outputs/adapters/d3a_ent" "$DEST/adapters/" 2>/dev/null || true
  scp "${SSH_OPTS[@]}" "ubuntu@${IP}:${REMOTE}/outputs/vectors/llama31_8b_vectors.pt" "$DEST/" 2>/dev/null || true
  if compgen -G "$DEST/arxiv_mva/"* >/dev/null; then
    cp "$DEST/arxiv_mva/"* "$MVA/"
  fi
}

pull_one "$ARXIV_A_JOB" "$ARXIV_A_IP"
pull_one "$ARXIV_B_JOB" "$ARXIV_B_IP"
pull_one "$ARXIV_C_JOB" "$ARXIV_C_IP"
pull_one "$ARXIV_D_JOB" "$ARXIV_D_IP"

echo "Merged artifacts -> $MVA"
python3 "$ROOT/scripts/build_arxiv_paper_table.py" --arxiv-dir "$MVA" --out "$MVA/paper_table.md"
cat "$MVA/paper_table.md"
