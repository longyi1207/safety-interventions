#!/usr/bin/env bash
# Sync tier scripts + review JSONL, restart tier queue on active EC2.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/cloud/.active/latest.env" 2>/dev/null || true
# shellcheck source=/dev/null
[[ -f "$ROOT/cloud/config.env" ]] && source "$ROOT/cloud/config.env"
KEY="${AWS_KEY_FILE:-$ROOT/cloud/.ssh/safety-interventions.pem}"
IP="${PUBLIC_IP:-}"
[[ -n "$IP" ]] || { echo "Missing PUBLIC_IP in cloud/.active/latest.env"; exit 1; }
H="ubuntu@$IP"
R="/home/ubuntu/ai_lab/code/safety_interventions"
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15)

rsync -avz -e "ssh -i $KEY" "$ROOT/cloud/run_tier_experiments.sh" "$H:$R/cloud/"
for f in eval_real_capability.py eval_lora_attack_matrix.py eval_d3c_bypass.py eval_d3a_rfa_scale.py eval_rfa_restore.py taxonomy_c1_review.py build_attack_mechanism_samples.py; do
  rsync -avz -e "ssh -i $KEY" "$ROOT/scripts/$f" "$H:$R/scripts/"
done
"${SSH[@]}" "$H" "mkdir -p $R/outputs/cloud_pull/si-20260602-044255/ablations"
rsync -avz -e "ssh -i $KEY" \
  "$ROOT/outputs/cloud_pull/si-20260602-044255/ablations/c1_c4_review_main.jsonl" \
  "$H:$R/outputs/cloud_pull/si-20260602-044255/ablations/"

bash "$ROOT/cloud/sync_tier_adapters.sh"

"${SSH[@]}" "$H" bash -s <<REMOTE
set -euo pipefail
cd $R
mkdir -p outputs outputs/ablations/tier_experiments
pkill -f run_tier_experiments.sh 2>/dev/null || true
pkill -f 'scripts/eval_' 2>/dev/null || true
sleep 2
nohup bash cloud/run_tier_experiments.sh >>outputs/cloud_job.log 2>&1 &
echo \$! >outputs/cloud_job.pid
echo tier_pid=\$(cat outputs/cloud_job.pid)
pgrep -f queue_tier_retries.sh >/dev/null || nohup bash cloud/queue_tier_retries.sh >>outputs/cloud_job.log 2>&1 &
REMOTE

echo "Tier restarted on $IP (watcher will retry failures after main job)"
