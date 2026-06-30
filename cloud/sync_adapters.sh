#!/usr/bin/env bash
# Push restored LoRA adapters + vectors to a running instance (sync_code excludes outputs/).
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

for adp in d2_er d3a_ent d3c_fuse_v3d; do
  [[ -d "$ROOT/outputs/adapters/$adp" ]] || { echo "Missing outputs/adapters/$adp — run scripts/restore_adapters_from_pull.sh" >&2; exit 1; }
done
[[ -f "$ROOT/outputs/vectors/qwen7b_vectors.pt" ]] || { echo "Missing qwen7b_vectors.pt" >&2; exit 1; }

ssh "${SSH_OPTS[@]}" "$REMOTE" "mkdir -p $REMOTE_DIR/outputs/adapters $REMOTE_DIR/outputs/vectors"
rsync -avz -e "ssh ${SSH_OPTS[*]}" \
  "$ROOT/outputs/vectors/qwen7b_vectors.pt" \
  "${REMOTE}:${REMOTE_DIR}/outputs/vectors/"
for adp in d2_er d3a_ent d3c_fuse_v3d; do
  rsync -avz -e "ssh ${SSH_OPTS[*]}" \
    "$ROOT/outputs/adapters/$adp/" \
    "${REMOTE}:${REMOTE_DIR}/outputs/adapters/$adp/"
done
echo "Adapters + vectors synced -> ${REMOTE}:${REMOTE_DIR}/outputs/"
