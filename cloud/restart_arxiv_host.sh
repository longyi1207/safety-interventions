#!/usr/bin/env bash
# Restart one arxiv track: sync fix, optional HF cache clear, rerun.
set -uo pipefail
IP="$1" RUN_SH="$2" CLEAR_HF="${3:-0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KEY="${AWS_KEY_FILE:-$ROOT/cloud/.ssh/safety-interventions.pem}"
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30)
[[ -f "$KEY" ]] && SSH_OPTS=(-i "$KEY" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30)

rsync -az -e "ssh ${SSH_OPTS[*]}" \
  --exclude '.venv' --exclude 'outputs/' --exclude 'cloud/.active/' \
  "$ROOT/" "ubuntu@${IP}:~/ai_lab/code/safety_interventions/"

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
export PATH="\$HOME/.local/bin:\$PATH"
if [ -f outputs/cloud_job.pid ]; then kill \$(cat outputs/cloud_job.pid) 2>/dev/null || true; fi
if [[ "$CLEAR_HF" == "1" ]]; then rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct 2>/dev/null || true; fi
: > outputs/cloud_job.log
nohup bash -c '${ENV_SETUP}bash ${RUN_SH}' >> outputs/cloud_job.log 2>&1 &
echo \$! > outputs/cloud_job.pid
echo restarted pid=\$(cat outputs/cloud_job.pid)
REMOTE
