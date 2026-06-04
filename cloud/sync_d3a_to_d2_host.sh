#!/usr/bin/env bash
# Copy d3a_ent adapter from D3 instance to D2 host for combined main eval.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PAR="$ROOT/cloud/.active/parallel_d23.env"
# shellcheck source=/dev/null
source "$PAR"
KEY="${AWS_KEY_FILE:-$ROOT/cloud/.ssh/safety-interventions.pem}"
SSH_OPTS=(-o StrictHostKeyChecking=accept-new)
[[ -f "$KEY" ]] && SSH_OPTS=(-i "$KEY" -o StrictHostKeyChecking=accept-new)
RDIR='~/ai_lab/code/safety_interventions'

echo "d3a_ent: ${D3_IP} -> ${D2_IP}"
ssh "${SSH_OPTS[@]}" "ubuntu@${D3_IP}" \
  "tar czf - -C ${RDIR}/outputs/adapters d3a_ent" | \
  ssh "${SSH_OPTS[@]}" "ubuntu@${D2_IP}" \
  "mkdir -p ${RDIR}/outputs/adapters && tar xzf - -C ${RDIR}/outputs/adapters"
echo "Done."
