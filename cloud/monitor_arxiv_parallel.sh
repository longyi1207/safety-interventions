#!/usr/bin/env bash
# Poll all arxiv tracks until done or timeout.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PAR="$ROOT/cloud/.active/parallel_arxiv.env"
# shellcheck source=/dev/null
source "$PAR"
KEY="${AWS_KEY_FILE:-$ROOT/cloud/.ssh/safety-interventions.pem}"
SSH_OPTS=(-o ConnectHostKeyChecking=no -o ConnectTimeout=10)
[[ -f "$KEY" ]] && SSH_OPTS=(-i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=10)

check() {
  local label="$1" ip="$2" marker="$3"
  ssh "${SSH_OPTS[@]}" "ubuntu@${ip}" bash -s <<REMOTE
cd ~/ai_lab/code/safety_interventions 2>/dev/null || { echo $label:NO_REPO; exit 0; }
if grep -q "$marker" outputs/cloud_job.log 2>/dev/null; then echo $label:DONE; exit 0; fi
if ps -p \$(cat outputs/cloud_job.pid 2>/dev/null) >/dev/null 2>&1; then echo $label:RUNNING; else echo $label:DEAD; fi
tail -1 outputs/cloud_job.log 2>/dev/null | cut -c1-120
REMOTE
}

TIMEOUT=${1:-14400}
START=$(date +%s)
while true; do
  NOW=$(date +%s)
  [[ $((NOW - START)) -gt $TIMEOUT ]] && { echo "TIMEOUT"; exit 1; }
  echo "=== $(date -u +%H:%M:%S) ==="
  check A "$ARXIV_A_IP" "done track A"
  check B "$ARXIV_B_IP" "done track B"
  check C "$ARXIV_C_IP" "done track C"
  check D "$ARXIV_D_IP" "done track D"
  DONE=0
  for s in A B C D; do
    # shellcheck disable=SC2154
    ip_var="ARXIV_${s}_IP"
  done
  A=$(check A "$ARXIV_A_IP" "done track A" | head -1)
  B=$(check B "$ARXIV_B_IP" "done track B" | head -1)
  C=$(check C "$ARXIV_C_IP" "done track C" | head -1)
  D=$(check D "$ARXIV_D_IP" "done track D" | head -1)
  echo "$A | $B | $C | $D"
  [[ "$A" == *DONE* && "$B" == *DONE* && "$C" == *DONE* && "$D" == *DONE* ]] && { echo ALL_DONE; exit 0; }
  sleep 120
done
