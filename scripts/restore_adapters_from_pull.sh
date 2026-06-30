#!/usr/bin/env bash
# Copy final LoRA adapters from outputs/cloud_pull into outputs/adapters/ for local + cloud eval.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

copy_adapter() {
  local src="$1" name="$2"
  if [[ ! -d "$src" ]]; then
    echo "SKIP missing $src"
    return
  fi
  rm -rf "outputs/adapters/$name"
  mkdir -p "outputs/adapters"
  cp -R "$src" "outputs/adapters/$name"
  echo "OK outputs/adapters/$name"
}

# Primary artifact dirs (override with CLOUD_PULL=.../si-XXX)
PULL="${CLOUD_PULL:-$ROOT/outputs/cloud_pull/si-20260602-182626}"
PULL2="${CLOUD_PULL2:-$ROOT/outputs/cloud_pull/si-20260602-194000}"
PULL3="${CLOUD_PULL3:-$ROOT/outputs/cloud_pull/si-20260603-175305-d3c}"

copy_adapter "$PULL/adapters/d3a_ent" d3a_ent
copy_adapter "$PULL2/adapters/d2_er" d2_er
copy_adapter "$PULL3/adapters/d3c_fuse_v3d" d3c_fuse_v3d 2>/dev/null \
  || copy_adapter "$ROOT/outputs/adapters/d3c_fuse_v3d" d3c_fuse_v3d 2>/dev/null \
  || copy_adapter "$PULL2/adapters/d3c_fuse" d3c_fuse

echo "Adapters ready under outputs/adapters/"
