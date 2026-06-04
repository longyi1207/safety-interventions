#!/usr/bin/env bash
# Push local prompt artifacts TO cloud (inverse of sync_code exclude).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/cloud/_ssh_env.sh" && _cloud_ssh "$ROOT"

KEY="${AWS_KEY_FILE:-$ROOT/cloud/.ssh/safety-interventions.pem}"
SSH_OPTS=(-o StrictHostKeyChecking=accept-new)
[[ -f "$KEY" ]] && SSH_OPTS=(-i "$KEY" -o StrictHostKeyChecking=accept-new)
REMOTE="${SSH_USER}@${PUBLIC_IP}"
REMOTE_DIR="${REMOTE_REPO_DIR:-~/ai_lab/code/safety_interventions}"

for f in prompts/evil_contrast_full.jsonl prompts/quality_contrast; do
  if [[ -e "$ROOT/$f" ]]; then
    rsync -avz -e "ssh ${SSH_OPTS[*]}" "$ROOT/$f" "${REMOTE}:${REMOTE_DIR}/$(dirname "$f")/"
  fi
done
echo "Prompts synced up -> $REMOTE_DIR"
