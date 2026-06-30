#!/usr/bin/env bash
# After launch_arxiv_parallel.sh: sync code, adapters, setup, start all 4 tracks.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PAR="$ROOT/cloud/.active/parallel_arxiv.env"
[[ -f "$PAR" ]] || { echo "Run launch_arxiv_parallel.sh first"; exit 1; }
# shellcheck source=/dev/null
source "$PAR"
# shellcheck source=/dev/null
source "$ROOT/cloud/config.env"
KEY="${AWS_KEY_FILE:-$ROOT/cloud/.ssh/safety-interventions.pem}"
SSH_OPTS=(-o StrictHostKeyChecking=accept-new)
[[ -f "$KEY" ]] && SSH_OPTS=(-i "$KEY" -o StrictHostKeyChecking=accept-new)

# Restore adapters locally if missing
if [[ ! -d "$ROOT/outputs/adapters/d3a_ent" ]]; then
  bash "$ROOT/scripts/restore_adapters_from_pull.sh"
fi

start_one() {
  local IP="$1" RUN_SH="$2" NAME="$3" NEED_ADAPTERS="$4"
  echo "=== $NAME @ $IP → $RUN_SH ==="
  for i in $(seq 1 36); do
    ssh "${SSH_OPTS[@]}" "ubuntu@${IP}" "echo ok" 2>/dev/null && break
    sleep 10
    [[ $i -eq 36 ]] && { echo "SSH timeout $IP"; return 1; }
  done
  ssh "${SSH_OPTS[@]}" "ubuntu@${IP}" "mkdir -p ~/ai_lab/code/safety_interventions"
  rsync -avz -e "ssh ${SSH_OPTS[*]}" \
    --exclude '.venv' --exclude 'outputs/' --exclude 'cloud/.active/' \
    "$ROOT/" "ubuntu@${IP}:~/ai_lab/code/safety_interventions/"
  ssh "${SSH_OPTS[@]}" "ubuntu@${IP}" bash -s <<'REMOTE'
cd ~/ai_lab/code/safety_interventions
export PATH="$HOME/.local/bin:$PATH"
pip install -q -r requirements.txt peft datasets openai
pip install -q "torch==2.5.1" "torchvision==0.20.1" --index-url https://download.pytorch.org/whl/cu124 2>/dev/null || true
mkdir -p outputs/vectors outputs/adapters outputs/arxiv_mva outputs/data outputs/cache
REMOTE
  if [[ "$NEED_ADAPTERS" == "1" ]]; then
    scp "${SSH_OPTS[@]}" "$ROOT/outputs/vectors/qwen7b_vectors.pt" \
      "ubuntu@${IP}:~/ai_lab/code/safety_interventions/outputs/vectors/"
    for adp in d2_er d3a_ent d3c_fuse_v3d; do
      rsync -avz -e "ssh ${SSH_OPTS[*]}" \
        "$ROOT/outputs/adapters/$adp/" \
        "ubuntu@${IP}:~/ai_lab/code/safety_interventions/outputs/adapters/$adp/"
    done
  fi
  ENV_SETUP=""
  if source "$ROOT/cloud/_env_from_dotenv.sh" 2>/dev/null; then
    TMPENV=$(mktemp); chmod 600 "$TMPENV"
    {
      [[ -n "${OPENAI_API_KEY:-}" ]] && printf 'export OPENAI_API_KEY=%q\n' "$OPENAI_API_KEY"
      [[ -n "${HF_TOKEN:-}" ]] && printf 'export HF_TOKEN=%q\n' "$HF_TOKEN"
      [[ -n "${HUGGING_FACE_TOKEN:-}" ]] && printf 'export HF_TOKEN=%q\n' "$HUGGING_FACE_TOKEN"
    } >"$TMPENV"
    scp "${SSH_OPTS[@]}" "$TMPENV" "ubuntu@${IP}:.si_eval_env"; rm -f "$TMPENV"
    ENV_SETUP="set -a && source ~/.si_eval_env && set +a && "
  fi
  ssh "${SSH_OPTS[@]}" "ubuntu@${IP}" bash -s <<REMOTE
set -euo pipefail
cd ~/ai_lab/code/safety_interventions
mkdir -p outputs
nohup bash -c '${ENV_SETUP}bash ${RUN_SH}' >> outputs/cloud_job.log 2>&1 &
echo \$! > outputs/cloud_job.pid
echo "started pid=\$(cat outputs/cloud_job.pid)"
REMOTE
}

start_one "$ARXIV_A_IP" "cloud/run_arxiv_track_a.sh" "Track A (RFA sweep)" 1 || echo "WARN: Track A bootstrap failed" >&2
start_one "$ARXIV_B_IP" "cloud/run_arxiv_track_b.sh" "Track B (unified table)" 1 || echo "WARN: Track B bootstrap failed" >&2
start_one "$ARXIV_C_IP" "cloud/run_arxiv_track_c.sh" "Track C (MMLU+cap)" 1 || echo "WARN: Track C bootstrap failed" >&2
start_one "$ARXIV_D_IP" "cloud/run_arxiv_track_d_llama.sh" "Track D (Llama)" 0 || echo "WARN: Track D bootstrap failed" >&2

echo ""
echo "All 4 tracks started. Monitor:"
echo "  source $PAR && for e in \$ARXIV-A_ENV \$ARXIV-B_ENV \$ARXIV-C_ENV \$ARXIV-D_ENV; do JOB_ENV=\$e cloud/status.sh; done"
echo "Pull when done: bash cloud/pull_arxiv_parallel.sh"
