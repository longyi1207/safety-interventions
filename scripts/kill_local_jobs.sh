#!/usr/bin/env bash
# Kill orphaned safety_interventions GPU/CPU jobs (Qwen inference on MPS).
set -euo pipefail
pkill -f "code/safety_interventions.*sweep_evil" 2>/dev/null || true
pkill -f "code/safety_interventions.*bootstrap_evil" 2>/dev/null || true
pkill -f "code/safety_interventions.*extract_vectors" 2>/dev/null || true
pkill -f "code/safety_interventions.*generate_eval" 2>/dev/null || true
pkill -f "code/safety_interventions.*validate_evil" 2>/dev/null || true
# Orphaned heredoc sweeps (cwd = safety_interventions, bare Python)
pkill -f "safety_interventions" 2>/dev/null || true
sleep 1
if pgrep -f "safety_interventions" >/dev/null 2>&1; then
  echo "Still running:" && pgrep -fl "safety_interventions"
  exit 1
fi
echo "All safety_interventions jobs stopped."
