#!/usr/bin/env bash
# D3c mandatory-fuse full pipeline (default v3d). Override: TRACK=d3c_fuse CFG=configs/d3_lora_train.yaml
set -euo pipefail
cd "$(dirname "$0")/.."
export TRACK="${TRACK:-d3c_fuse_v3d}"
export CFG="${CFG:-configs/d3_lora_train_v3d.yaml}"
exec bash cloud/run_d3c_pipeline.sh
