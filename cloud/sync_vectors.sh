#!/usr/bin/env bash
# Push local vector artifacts to running cloud instance (sync_code excludes outputs/).
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
VEC="$ROOT/outputs/vectors"

for f in qwen7b_vectors.pt qwen7b_qualities_ortho.pt; do
  [[ -f "$VEC/$f" ]] || { echo "Missing $VEC/$f — run local extract first." >&2; exit 1; }
done

ssh "${SSH_OPTS[@]}" "$REMOTE" "mkdir -p $REMOTE_DIR/outputs/vectors"
rsync -avz -e "ssh ${SSH_OPTS[*]}" \
  "$VEC/qwen7b_vectors.pt" \
  "$VEC/qwen7b_qualities_ortho.pt" \
  "${REMOTE}:${REMOTE_DIR}/outputs/vectors/"
echo "Vectors synced -> ${REMOTE}:${REMOTE_DIR}/outputs/vectors/"
