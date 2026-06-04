#!/usr/bin/env bash
# Push complete LoRA adapters to active EC2 (tar stream — survives flaky rsync).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/cloud/.active/latest.env"
KEY="${AWS_KEY_FILE:-$ROOT/cloud/.ssh/safety-interventions.pem}"
H="ubuntu@${PUBLIC_IP}"
R="/home/ubuntu/ai_lab/code/safety_interventions"

push_adapter() {
  local name="$1" src="$2"
  [[ -f "$src/adapter_model.safetensors" ]] || { echo "MISSING $src"; return 1; }
  echo "Pushing $name from $src ..."
  tar -C "$src" -czf - . | ssh -i "$KEY" "$H" "mkdir -p $R/outputs/adapters/$name && tar -xzf - -C $R/outputs/adapters/$name"
  ssh -i "$KEY" "$H" "ls -lh $R/outputs/adapters/$name/adapter_model.safetensors"
}

push_adapter d3a_ent "$ROOT/outputs/cloud_pull/si-20260602-182626/adapters/d3a_ent"
push_adapter d2_er "$ROOT/outputs/cloud_pull/si-20260602-194000/adapters/d2_er"
push_adapter d3c_fuse "$ROOT/outputs/cloud_pull/si-20260602-194000/adapters/d3c_fuse"
echo "done"
