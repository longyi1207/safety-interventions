#!/usr/bin/env bash
# Shared SSH helper — sources cloud/.active/latest.env + config.env
_cloud_ssh() {
  local ROOT="$1"
  # shellcheck source=/dev/null
  source "$ROOT/cloud/.active/latest.env"
  # shellcheck source=/dev/null
  [[ -f "$ROOT/cloud/config.env" ]] && source "$ROOT/cloud/config.env"
  KEY="${AWS_KEY_FILE:-$ROOT/cloud/.ssh/safety-interventions.pem}"
  SSH="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"
  [[ -f "$KEY" ]] && SSH="ssh -i $KEY -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"
  REMOTE_DIR="${REMOTE_REPO_DIR:-~/ai_lab/code/safety_interventions}"
}
