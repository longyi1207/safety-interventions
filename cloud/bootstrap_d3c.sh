#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PAR="$ROOT/cloud/.active/parallel_d3c.env"
[[ -f "$PAR" ]] || { echo "Run launch_d3c.sh first"; exit 1; }
# shellcheck source=/dev/null
source "$PAR"
# shellcheck source=/dev/null
source "$ROOT/cloud/config.env"
KEY="${AWS_KEY_FILE:-$ROOT/cloud/.ssh/safety-interventions.pem}"
SSH_OPTS=(-o StrictHostKeyChecking=accept-new)
[[ -f "$KEY" ]] && SSH_OPTS=(-i "$KEY" -o StrictHostKeyChecking=accept-new)

for i in $(seq 1 30); do
  ssh "${SSH_OPTS[@]}" "ubuntu@${D3C_IP}" "echo ok" 2>/dev/null && break
  sleep 10
done

ssh "${SSH_OPTS[@]}" "ubuntu@${D3C_IP}" "mkdir -p ~/ai_lab/code/safety_interventions"

rsync -avz -e "ssh ${SSH_OPTS[*]}" \
  --exclude '.venv' --exclude 'outputs/' --exclude 'cloud/.active/' \
  "$ROOT/" "ubuntu@${D3C_IP}:~/ai_lab/code/safety_interventions/"

ssh "${SSH_OPTS[@]}" "ubuntu@${D3C_IP}" bash -s <<'REMOTE'
cd ~/ai_lab/code/safety_interventions
pip install -q -r requirements.txt
pip install -q "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1" \
  --index-url https://download.pytorch.org/whl/cu124
pip install -q "transformers>=4.44,<4.50" "peft>=0.11,<0.15" openai huggingface_hub
mkdir -p outputs/vectors outputs/adapters outputs/data outputs/ablations
REMOTE

[[ -f "$ROOT/outputs/vectors/qwen7b_vectors.pt" ]] && \
  scp "${SSH_OPTS[@]}" "$ROOT/outputs/vectors/qwen7b_vectors.pt" \
    "ubuntu@${D3C_IP}:~/ai_lab/code/safety_interventions/outputs/vectors/"

ENV_SETUP=""
if source "$ROOT/cloud/_env_from_dotenv.sh" 2>/dev/null && [[ -n "${OPENAI_API_KEY:-}" ]]; then
  TMPENV=$(mktemp); chmod 600 "$TMPENV"
  printf 'export OPENAI_API_KEY=%q\n' "$OPENAI_API_KEY" >"$TMPENV"
  scp "${SSH_OPTS[@]}" "$TMPENV" "ubuntu@${D3C_IP}:.si_eval_env"; rm -f "$TMPENV"
  ENV_SETUP="set -a && source ~/.si_eval_env && set +a && "
fi

ssh "${SSH_OPTS[@]}" "ubuntu@${D3C_IP}" bash -s <<REMOTE
set -euo pipefail
cd ~/ai_lab/code/safety_interventions
mkdir -p outputs
${ENV_SETUP}nohup bash cloud/run_d3c_only.sh >> outputs/cloud_job.log 2>&1 &
echo \$! > outputs/cloud_job.pid
echo "started pid=\$(cat outputs/cloud_job.pid)"
REMOTE

echo "D3c job @ $D3C_IP — tail outputs/cloud_job.log"
