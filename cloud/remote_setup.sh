#!/usr/bin/env bash
# Run FROM YOUR MAC: wait for SSH, install Python deps on GPU instance.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/cloud/_ssh_env.sh" && _cloud_ssh "$ROOT"

echo "Waiting for SSH on ${PUBLIC_IP}..."
for i in $(seq 1 36); do
  if $SSH "${SSH_USER}@${PUBLIC_IP}" "echo ok" 2>/dev/null; then break; fi
  sleep 10
  [[ $i -eq 36 ]] && { echo "SSH timeout after 6 min"; exit 1; }
done

$SSH "${SSH_USER}@${PUBLIC_IP}" bash -s <<REMOTE
set -e
mkdir -p ~/ai_lab/code/safety_interventions
cd ~/ai_lab/code/safety_interventions
pip install -q -r requirements.txt
# DLAMI ships torch 2.12+cu130 which hangs on Qwen generate; pin stable stack.
pip install -q "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1" \
  --index-url https://download.pytorch.org/whl/cu124
pip install -q openai huggingface_hub peft
mkdir -p outputs/ablations outputs/vectors
echo "Setup OK on \$(hostname)"
nvidia-smi | head -3
REMOTE

echo "Remote deps OK. Next: cloud/sync_code.sh && cloud/start_job.sh"
