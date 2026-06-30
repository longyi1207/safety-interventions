#!/usr/bin/env bash
# Poll D g5 until B+C done, pull artifacts, build table, teardown.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IP="${1:-34.224.69.156}"
KEY="${AWS_KEY_FILE:-$ROOT/cloud/.ssh/safety-interventions.pem}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o ConnectTimeout=15 -o ServerAliveInterval=30)
[[ -f "$KEY" ]] && SSH_OPTS=(-i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o ServerAliveInterval=30)
REMOTE=ubuntu@${IP}
MVA="$ROOT/outputs/arxiv_mva"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

start_c_if_needed() {
  ssh "${SSH_OPTS[@]}" "$REMOTE" bash -s <<'REMOTE'
set -euo pipefail
cd ~/ai_lab/code/safety_interventions
grep -q "done track B" outputs/cloud_job.log || exit 0
grep -q "done track C" outputs/cloud_job.log && exit 0
if ps -p $(cat outputs/cloud_job.pid 2>/dev/null) >/dev/null 2>&1; then
  grep -q "arxiv-C\|done track C" outputs/cloud_job.log && exit 0
fi
export PATH="$HOME/.local/bin:$PATH"
[[ -f ~/.si_eval_env ]] && set -a && source ~/.si_eval_env && set +a
nohup bash -c 'bash cloud/run_arxiv_track_c.sh' >> outputs/cloud_job.log 2>&1 &
echo $! > outputs/cloud_job.pid
echo "started C pid=$(cat outputs/cloud_job.pid)"
REMOTE
}

pull_all() {
  mkdir -p "$MVA" "$ROOT/outputs/cloud_pull/si-arxiv-d-g5/arxiv_mva"
  scp "${SSH_OPTS[@]}" "${REMOTE}:~/ai_lab/code/safety_interventions/outputs/arxiv_mva/*.json" "$MVA/" 2>/dev/null || true
  cp "$MVA"/*.json "$ROOT/outputs/cloud_pull/si-arxiv-d-g5/arxiv_mva/" 2>/dev/null || true
}

TIMEOUT=28800
START=$(date +%s)
C_STARTED=0

while true; do
  NOW=$(date +%s)
  [[ $((NOW - START)) -gt $TIMEOUT ]] && { log "TIMEOUT"; exit 1; }

  STATUS=$(ssh "${SSH_OPTS[@]}" "$REMOTE" bash -s <<'REMOTE'
cd ~/ai_lab/code/safety_interventions
B=0; C=0
grep -q "done track B" outputs/cloud_job.log && B=1
grep -q "done track C" outputs/cloud_job.log && C=1
RUN=0
ps -p $(cat outputs/cloud_job.pid 2>/dev/null) >/dev/null 2>&1 && RUN=1
TAIL=$(tail -1 outputs/cloud_job.log 2>/dev/null | cut -c1-80)
echo "B=$B C=$C RUN=$RUN"
echo "LOG=$TAIL"
ls outputs/arxiv_mva/*.json 2>/dev/null | wc -l
REMOTE
  )

  B_DONE=$(echo "$STATUS" | grep '^B=' | head -1 | cut -d= -f2)
  C_DONE=$(echo "$STATUS" | grep '^C=' | head -1 | cut -d= -f2)
  NFILES=$(echo "$STATUS" | tail -1 | tr -d ' ')

  log "B=$B_DONE C=$C_DONE files=$NFILES"
  echo "$STATUS" | grep -E '^LOG'

  if [[ "$B_DONE" == "1" && "$C_STARTED" == "0" ]]; then
    log "B done — starting C"
    start_c_if_needed && C_STARTED=1
  fi

  if [[ "$B_DONE" == "1" && "$C_DONE" == "1" ]]; then
    log "B+C complete — pulling"
    pull_all
    python3 "$ROOT/scripts/build_arxiv_paper_table.py"
    # merge cloud_pull tier cap as fallback if any mmlu missing
  log "teardown D g5"
    aws ec2 terminate-instances --region us-east-1 --instance-ids i-094f41c636cfe2ff1 --output text >/dev/null 2>&1 || true
    log "ALL DONE"
    exit 0
  fi

  sleep 120
done
