#!/usr/bin/env bash
# Full handpick suite on one GPU: main matrix then subspace extract+probe.
set -euo pipefail
cd "$(dirname "$0")/.."
bash cloud/run_handpick_probe_main.sh
bash cloud/run_handpick_probe_subspace.sh
