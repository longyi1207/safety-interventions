#!/usr/bin/env bash
# Local poll of EC2 tier queue; restart retry/watcher if jobs died with gaps.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/cloud/.active/latest.env" 2>/dev/null || true
KEY="${AWS_KEY_FILE:-$ROOT/cloud/.ssh/safety-interventions.pem}"
IP="${PUBLIC_IP:-}"
[[ -n "$IP" ]] || { echo "no PUBLIC_IP"; exit 1; }
H="ubuntu@$IP"
R="/home/ubuntu/ai_lab/code/safety_interventions"
LOG="$ROOT/outputs/ablations/tier_monitor.log"
mkdir -p "$(dirname "$LOG")"

ssh -i "$KEY" -o ConnectTimeout=15 "$H" bash -s <<REMOTE
cd $R
need="cap_d3a_clean.json cap_d3a_rfa.json attack_matrix_stock.json attack_matrix_d2_er.json attack_matrix_d3a_ent.json attack_matrix_d3c_fuse.json attack_matrix_d3c_fuse_v3d.json"
missing=""
for f in \$need; do
  [[ -f outputs/ablations/tier_experiments/\$f ]] || missing="\$missing \$f"
done
tier_alive=0
watcher_alive=0
retry_alive=0
[[ -f outputs/cloud_job.pid ]] && kill -0 \$(cat outputs/cloud_job.pid) 2>/dev/null && tier_alive=1
[[ -f outputs/tier_watcher.pid ]] && kill -0 \$(cat outputs/tier_watcher.pid) 2>/dev/null && watcher_alive=1
[[ -f outputs/tier_retry.pid ]] && kill -0 \$(cat outputs/tier_retry.pid) 2>/dev/null && retry_alive=1
py=\$(pgrep -f 'python3 scripts/eval' | head -1 || true)
echo "TS=\$(date -u +%Y-%m-%dT%H:%M:%SZ) tier=\$tier_alive watcher=\$watcher_alive retry=\$retry_alive py=\${py:-none} missing=\${missing:-none}"
if [[ -n "\$py" ]]; then ps -p "\$py" -o etime,cmd= 2>/dev/null || true; fi
grep -E '\[tier\]|\[tier-retry\]' outputs/cloud_job.log | tail -3

# tier dead, gaps remain, no retry running → start retry
if [[ "\$tier_alive" -eq 0 && -n "\$missing" && "\$retry_alive" -eq 0 ]]; then
  echo "ACTION=start_retry"
  nohup bash cloud/retry_tier_failures.sh >>outputs/cloud_job.log 2>&1 &
  echo \$! >outputs/tier_retry.pid
fi
# nothing running at all but gaps → full retry
if [[ "\$tier_alive" -eq 0 && "\$watcher_alive" -eq 0 && "\$retry_alive" -eq 0 && -n "\$missing" ]]; then
  if [[ -z "\$(pgrep -f 'python3 scripts/eval' || true)" ]]; then
    echo "ACTION=start_retry_standalone"
    nohup bash cloud/retry_tier_failures.sh >>outputs/cloud_job.log 2>&1 &
    echo \$! >outputs/tier_retry.pid
  fi
fi
# tier dead, watcher dead, gaps, no py → ensure watcher not needed; kick retry
if [[ "\$tier_alive" -eq 0 && "\$watcher_alive" -eq 0 && -n "\$missing" ]]; then
  pgrep -f retry_tier_failures >/dev/null || {
    echo "ACTION=kick_retry"
    nohup bash cloud/retry_tier_failures.sh >>outputs/cloud_job.log 2>&1 &
    echo \$! >outputs/tier_retry.pid
  }
fi
REMOTE
