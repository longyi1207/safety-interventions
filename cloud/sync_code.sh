#!/usr/bin/env bash
# Rsync local safety_interventions to the running instance (code not on GitHub yet).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ACTIVE="${JOB_ENV:-$ROOT/cloud/.active/latest.env}"
[[ -f "$ACTIVE" ]] || { echo "Run cloud/launch.sh first." >&2; exit 1; }
# shellcheck source=/dev/null
source "$ACTIVE"
# shellcheck source=/dev/null
[[ -f "$ROOT/cloud/config.env" ]] && source "$ROOT/cloud/config.env"

KEY="${AWS_KEY_FILE:-$ROOT/cloud/.ssh/safety-interventions.pem}"
SSH_OPTS=(-o StrictHostKeyChecking=accept-new)
[[ -f "$KEY" ]] && SSH_OPTS=(-i "$KEY" -o StrictHostKeyChecking=accept-new)
REMOTE="${SSH_USER}@${PUBLIC_IP}"
REMOTE_DIR="${REMOTE_REPO_DIR:-~/ai_lab/code/safety_interventions}"

ssh "${SSH_OPTS[@]}" "$REMOTE" "mkdir -p $REMOTE_DIR"
rsync -avz -e "ssh ${SSH_OPTS[*]}" \
  --exclude '.venv' --exclude '__pycache__' \
  --exclude 'outputs/' --exclude 'cloud/.ssh/' --exclude 'cloud/.active/' \
  --exclude 'prompts/evil_contrast_full.jsonl' \
  "$ROOT/" "${REMOTE}:${REMOTE_DIR}/"

# Shared dependency: safety_interventions imports helper utilities from
# ../nla_rsa_study/src/common.py (get_device, dtype resolution, chat input helpers).
NLA_LOCAL="$(cd "$ROOT/.." && pwd)/nla_rsa_study/src/"
ssh "${SSH_OPTS[@]}" "$REMOTE" "mkdir -p ~/ai_lab/code/nla_rsa_study/src"
rsync -avz -e "ssh ${SSH_OPTS[*]}" \
  --exclude '__pycache__' \
  "$NLA_LOCAL" "${REMOTE}:~/ai_lab/code/nla_rsa_study/src/"
echo "Sync done -> $REMOTE_DIR"
