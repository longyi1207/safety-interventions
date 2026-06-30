#!/usr/bin/env bash
# Poll until track C done, pull, build table, teardown.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IP="${1:-34.224.69.156}"
KEY="${AWS_KEY_FILE:-$ROOT/cloud/.ssh/safety-interventions.pem}"
SSH=(ssh -i "$KEY" -o ConnectTimeout=15 -o ServerAliveInterval=30 ubuntu@"$IP")
MVA="$ROOT/outputs/arxiv_mva"

for i in $(seq 1 120); do
  STATUS=$("${SSH[@]}" 'cd ~/ai_lab/code/safety_interventions && {
    if grep -q "done track C" outputs/cloud_job.log 2>/dev/null; then echo DONE; exit 0; fi
    if ps -p $(cat outputs/cloud_job.pid 2>/dev/null) >/dev/null 2>&1; then
      echo RUNNING $(ps -p $(cat outputs/cloud_job.pid) -o etime=)
      grep "\[arxiv-C\]" outputs/cloud_job.log | tail -1
      tail -1 outputs/cloud_job.log | cut -c1-70
    else
      echo DEAD
      tail -3 outputs/cloud_job.log
    fi
  }' 2>&1)

  echo "[$(date -u +%H:%M:%S)] $STATUS" | head -3

  if echo "$STATUS" | grep -q "^DONE"; then
    scp -i "$KEY" "ubuntu@${IP}:~/ai_lab/code/safety_interventions/outputs/arxiv_mva/*.json" "$MVA/" 2>/dev/null || true
    python3 "$ROOT/scripts/build_arxiv_paper_table.py"
    aws ec2 terminate-instances --region us-east-1 --instance-ids i-094f41c636cfe2ff1 --output text >/dev/null 2>&1 || true
    echo "COMPLETE"
    exit 0
  fi

  if echo "$STATUS" | grep -q "^DEAD"; then
    echo "JOB DIED — check log"
    exit 1
  fi

  sleep 300
done
echo "TIMEOUT"
exit 1
