#!/usr/bin/env bash
# Rsync local safety_interventions to the running instance.
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

echo "Sync done -> $REMOTE_DIR (vendor/common.py included in repo)"
