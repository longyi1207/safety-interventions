#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=prompts/data
mkdir -p "$DATA"
BASE=https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets
echo "Downloading HarmBench + XSTest..."
curl -sL "$BASE/harmbench_behaviors_text_val.csv" -o "$DATA/harmbench_behaviors_text_val.csv"
curl -sL "$BASE/harmbench_behaviors_text_all.csv" -o "$DATA/harmbench_behaviors_text_all.csv"
curl -sL "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv" -o "$DATA/xstest_prompts.csv"
echo "Done: $(wc -l < "$DATA/harmbench_behaviors_text_val.csv") val behaviors, $(wc -l < "$DATA/xstest_prompts.csv") xstest rows"
