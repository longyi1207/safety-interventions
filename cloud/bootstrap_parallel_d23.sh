#!/usr/bin/env bash
# After launch_parallel_d23.sh: sync, setup, start both jobs.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PAR="$ROOT/cloud/.active/parallel_d23.env"
[[ -f "$PAR" ]] || { echo "Run launch_parallel_d23.sh first"; exit 1; }
# shellcheck source=/dev/null
source "$PAR"
# shellcheck source=/dev/null
source "$ROOT/cloud/config.env"
KEY="${AWS_KEY_FILE:-$ROOT/cloud/.ssh/safety-interventions.pem}"
SSH_OPTS=(-o StrictHostKeyChecking=accept-new)
[[ -f "$KEY" ]] && SSH_OPTS=(-i "$KEY" -o StrictHostKeyChecking=accept-new)

start_one() {
  local IP="$1" RUN_SH="$2" NAME="$3"
  echo "=== $NAME @ $IP → $RUN_SH ==="
  for i in $(seq 1 30); do
    ssh "${SSH_OPTS[@]}" "ubuntu@${IP}" "echo ok" 2>/dev/null && break
    sleep 10
  done
  rsync -avz -e "ssh ${SSH_OPTS[*]}" \
    --exclude '.venv' --exclude 'outputs/' --exclude 'cloud/.active/' \
    "$ROOT/" "ubuntu@${IP}:~/ai_lab/code/safety_interventions/"
  NLA_LOCAL="$(cd "$ROOT/.." && pwd)/nla_rsa_study/src/"
  rsync -avz -e "ssh ${SSH_OPTS[*]}" "$NLA_LOCAL" "ubuntu@${IP}:~/ai_lab/code/nla_rsa_study/src/"
  ssh "${SSH_OPTS[@]}" "ubuntu@${IP}" bash -s <<'REMOTE'
cd ~/ai_lab/code/safety_interventions
pip install -q -r requirements.txt peft
pip install -q "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu124 2>/dev/null || true
mkdir -p outputs/vectors outputs/adapters outputs/data
REMOTE
  [[ -f "$ROOT/outputs/vectors/qwen7b_vectors.pt" ]] && \
    scp "${SSH_OPTS[@]}" "$ROOT/outputs/vectors/qwen7b_vectors.pt" \
      "ubuntu@${IP}:~/ai_lab/code/safety_interventions/outputs/vectors/"
  ENV_SETUP=""
  if source "$ROOT/cloud/_env_from_dotenv.sh" 2>/dev/null && [[ -n "${OPENAI_API_KEY:-}" ]]; then
    TMPENV=$(mktemp); chmod 600 "$TMPENV"
    printf 'export OPENAI_API_KEY=%q\n' "$OPENAI_API_KEY" >"$TMPENV"
    scp "${SSH_OPTS[@]}" "$TMPENV" "ubuntu@${IP}:.si_eval_env"; rm -f "$TMPENV"
    ENV_SETUP="set -a && source ~/.si_eval_env && set +a && "
  fi
  ssh "${SSH_OPTS[@]}" "ubuntu@${IP}" bash -s <<REMOTE
set -euo pipefail
cd ~/ai_lab/code/safety_interventions
mkdir -p outputs
nohup bash -c '${ENV_SETUP}bash ${RUN_SH}' >> outputs/cloud_job.log 2>&1 &
echo \$! > outputs/cloud_job.pid
echo "started pid=\$(cat outputs/cloud_job.pid)"
REMOTE
}

start_one "$D2_IP" "cloud/run_d2_er_only.sh" "D2-ER"
start_one "$D3_IP" "cloud/run_d3a_d3c.sh" "D3a+D3c"

echo ""
echo "Jobs started. Monitor:"
echo "  JOB_ENV=$ROOT/cloud/.active/si-*-d2.env cloud/status.sh"
echo "  JOB_ENV=$ROOT/cloud/.active/si-*-d3.env cloud/status.sh"
echo "Pull: cloud/pull_parallel_d23.sh"
