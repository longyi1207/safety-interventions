#!/usr/bin/env bash
# Launch 2 spot instances and start handpick probes in parallel.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/cloud"

if [[ ! -f config.env ]]; then
  echo "Missing cloud/config.env" >&2
  exit 1
fi

echo "=== Launch instance 1 (main matrix) ==="
bash launch.sh
ENV1="$ROOT/cloud/.active/latest.env"
JOB1=$(grep '^JOB_ID=' "$ENV1" | cut -d= -f2)
echo "Job1: $JOB1"

echo "=== Launch instance 2 (subspace) ==="
bash launch.sh
ENV2="$ROOT/cloud/.active/latest.env"
JOB2=$(grep '^JOB_ID=' "$ENV2" | cut -d= -f2)
echo "Job2: $JOB2"

# shellcheck source=/dev/null
source "$ENV1"
IP1=$PUBLIC_IP
# shellcheck source=/dev/null
source "$ENV2"
IP2=$PUBLIC_IP

echo "Waiting 90s for SSH on both..."
sleep 90

KEY="${AWS_KEY_FILE:-$ROOT/cloud/.ssh/safety-interventions.pem}"
SSH_BASE=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15)

wait_ssh() {
  local ip=$1
  for _ in $(seq 1 30); do
    if "${SSH_BASE[@]}" "ubuntu@${ip}" true 2>/dev/null; then return 0; fi
    sleep 10
  done
  echo "SSH timeout: $ip" >&2
  return 1
}

wait_ssh "$IP1" &
wait_ssh "$IP2" &
wait

echo "=== Sync code + vectors to both ==="
JOB_ENV="$ENV1" bash "$ROOT/cloud/sync_code.sh"
JOB_ENV="$ENV1" bash "$ROOT/cloud/sync_vectors.sh" 2>/dev/null || true
JOB_ENV="$ENV2" bash "$ROOT/cloud/sync_code.sh"
JOB_ENV="$ENV2" bash "$ROOT/cloud/sync_vectors.sh" 2>/dev/null || true

echo "=== Remote setup (parallel) ==="
JOB_ENV="$ENV1" bash "$ROOT/cloud/remote_setup.sh" &
JOB_ENV="$ENV2" bash "$ROOT/cloud/remote_setup.sh" &
wait

echo "=== Start jobs ==="
PIPELINE=handpick_main JOB_ENV="$ENV1" bash "$ROOT/cloud/start_job.sh"
PIPELINE=handpick_subspace JOB_ENV="$ENV2" bash "$ROOT/cloud/start_job.sh"

cat <<EOF

Parallel handpick suite running:
  Main:     $JOB1 @ $IP1  → handpick_probe_main.jsonl
  Subspace: $JOB2 @ $IP2  → handpick_probe_subspace.jsonl

Monitor:
  JOB_ENV=$ENV1 bash cloud/status.sh
  JOB_ENV=$ENV2 bash cloud/status.sh

Pull (after both finish):
  JOB_ENV=$ENV1 bash cloud/pull_results.sh
  JOB_ENV=$ENV2 bash cloud/pull_results.sh

Teardown:
  JOB_ENV=$ENV1 bash cloud/teardown.sh
  JOB_ENV=$ENV2 bash cloud/teardown.sh

EOF
