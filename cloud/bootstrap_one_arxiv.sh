#!/usr/bin/env bash
# Bootstrap + start one arxiv track host. Usage: bootstrap_one_arxiv.sh IP RUN_SH NAME NEED_ADAPTERS
set -uo pipefail
IP="$1" RUN_SH="$2" NAME="$3" NEED_ADAPTERS="${4:-0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/cloud/config.env"
KEY="${AWS_KEY_FILE:-$ROOT/cloud/.ssh/safety-interventions.pem}"
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30)
[[ -f "$KEY" ]] && SSH_OPTS=(-i "$KEY" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30)

log() { echo "[$NAME@$IP] $*"; }

for i in $(seq 1 30); do
  ssh "${SSH_OPTS[@]}" "ubuntu@${IP}" "echo ok" 2>/dev/null && break
  sleep 5
done

ssh "${SSH_OPTS[@]}" "ubuntu@${IP}" "mkdir -p ~/ai_lab/code/safety_interventions"
rsync -az -e "ssh ${SSH_OPTS[*]}" \
  --exclude '.venv' --exclude 'outputs/' --exclude 'cloud/.active/' \
  "$ROOT/" "ubuntu@${IP}:~/ai_lab/code/safety_interventions/"

ssh "${SSH_OPTS[@]}" "ubuntu@${IP}" bash -s <<'REMOTE'
set -e
cd ~/ai_lab/code/safety_interventions
export PATH="$HOME/.local/bin:$PATH"
pip install -q -r requirements.txt peft datasets openai 2>/dev/null || pip install -q -r requirements.txt peft datasets openai
pip install -q "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu124 2>/dev/null || true
mkdir -p outputs/vectors outputs/adapters outputs/arxiv_mva outputs/data outputs/cache
REMOTE

if [[ "$NEED_ADAPTERS" == "1" ]]; then
  scp "${SSH_OPTS[@]}" "$ROOT/outputs/vectors/qwen7b_vectors.pt" \
    "ubuntu@${IP}:~/ai_lab/code/safety_interventions/outputs/vectors/"
  for adp in d2_er d3a_ent d3c_fuse_v3d; do
    rsync -az -e "ssh ${SSH_OPTS[*]}" \
      "$ROOT/outputs/adapters/$adp/" \
      "ubuntu@${IP}:~/ai_lab/code/safety_interventions/outputs/adapters/$adp/"
  done
fi

ENV_SETUP=""
if source "$ROOT/cloud/_env_from_dotenv.sh" 2>/dev/null; then
  TMPENV=$(mktemp); chmod 600 "$TMPENV"
  {
    [[ -n "${OPENAI_API_KEY:-}" ]] && printf 'export OPENAI_API_KEY=%q\n' "$OPENAI_API_KEY"
    [[ -n "${HF_TOKEN:-${HUGGING_FACE_TOKEN:-}}" ]] && printf 'export HF_TOKEN=%q\n' "${HF_TOKEN:-$HUGGING_FACE_TOKEN}"
  } >"$TMPENV"
  scp "${SSH_OPTS[@]}" "$TMPENV" "ubuntu@${IP}:.si_eval_env"; rm -f "$TMPENV"
  ENV_SETUP="set -a && source ~/.si_eval_env && set +a && "
fi

ssh "${SSH_OPTS[@]}" "ubuntu@${IP}" bash -s <<REMOTE
set -euo pipefail
cd ~/ai_lab/code/safety_interventions
if [ -f outputs/cloud_job.pid ]; then kill \$(cat outputs/cloud_job.pid) 2>/dev/null || true; fi
export PATH="\$HOME/.local/bin:\$PATH"
nohup bash -c '${ENV_SETUP}bash ${RUN_SH}' >> outputs/cloud_job.log 2>&1 &
echo \$! > outputs/cloud_job.pid
echo started pid=\$(cat outputs/cloud_job.pid)
REMOTE

log "started $(ssh "${SSH_OPTS[@]}" ubuntu@${IP} 'cat ~/ai_lab/code/safety_interventions/outputs/cloud_job.pid')"
