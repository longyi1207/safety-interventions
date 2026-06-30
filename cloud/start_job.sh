#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Optional: JOB_ENV=cloud/.active/si-XXXX.env to target a specific instance (not latest).
ACTIVE="${JOB_ENV:-$ROOT/cloud/.active/latest.env}"
# shellcheck source=/dev/null
source "$ACTIVE"
# shellcheck source=/dev/null
source "$ROOT/cloud/_ssh_env.sh"
KEY="${AWS_KEY_FILE:-$ROOT/cloud/.ssh/safety-interventions.pem}"
SSH="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"
[[ -f "$KEY" ]] && SSH="ssh -i $KEY -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"
REMOTE_DIR="${REMOTE_REPO_DIR:-~/ai_lab/code/safety_interventions}"
# shellcheck source=/dev/null
[[ -f "$ROOT/cloud/config.env" ]] && source "$ROOT/cloud/config.env"

CONFIG="${CONFIG:-configs/qwen7b_harmbench.cloud.yaml}"
CONDITIONS="${CONDITIONS:-C0,C2}"
PIPELINE="${PIPELINE:-c2_iter}"
JUDGE_FLAG=""
[[ "${BOOTSTRAP_JUDGE:-0}" == "1" ]] && JUDGE_FLAG="--judge"

case "$PIPELINE" in
  phase_b_v2) RUN_SCRIPT="cloud/run_phase_b_v2.sh" ;;
  gpu_resume) RUN_SCRIPT="cloud/run_phase_b_gpu_resume.sh" ;;
  finish) RUN_SCRIPT="cloud/run_phase_b_finish.sh" ;;
  weak_traits) RUN_SCRIPT="cloud/run_weak_traits_bootstrap.sh" ;;
  eval_only) RUN_SCRIPT="cloud/run_phase_b_eval_only.sh" ;;
  main_eval) RUN_SCRIPT="cloud/run_main_eval.sh" ;;
  conditional_main) RUN_SCRIPT="cloud/run_conditional_main.sh" ;;
  export_review) RUN_SCRIPT="cloud/run_export_c1_c4_review.sh" ;;
  export_then_conditional) RUN_SCRIPT="cloud/run_export_then_conditional.sh" ;;
  conditional_from_review) RUN_SCRIPT="cloud/run_conditional_from_review.sh" ;;
  handpick_main) RUN_SCRIPT="cloud/run_handpick_probe_main.sh" ;;
  handpick_subspace) RUN_SCRIPT="cloud/run_handpick_probe_subspace.sh" ;;
  handpick_all) RUN_SCRIPT="cloud/run_handpick_all.sh" ;;
  main_evil_system) RUN_SCRIPT="cloud/run_main_evil_system_eval.sh" ;;
  d2_d3a_train) RUN_SCRIPT="cloud/run_d2_d3a_train.sh" ;;
  d2_er_only) RUN_SCRIPT="cloud/run_d2_er_only.sh" ;;
  d3a_d3c) RUN_SCRIPT="cloud/run_d3a_d3c.sh" ;;
  d3c_only) RUN_SCRIPT="cloud/run_d3c_only.sh" ;;
  d3a_eval) RUN_SCRIPT="cloud/run_d3a_eval_only.sh" ;;
  d3_main_eval) RUN_SCRIPT="cloud/run_d3_main_eval.sh" ;;
  d3_main_eval_d2) RUN_SCRIPT="cloud/run_d3_main_eval_d2.sh" ;;
  d3_main_eval_d3a) RUN_SCRIPT="cloud/run_d3_main_eval_d3a.sh" ;;
  d3c_kill_retrain) RUN_SCRIPT="cloud/run_d3c_kill_retrain.sh" ;;
  d3c_pipeline) RUN_SCRIPT="cloud/run_d3c_pipeline.sh" ;;
  c2_main) RUN_SCRIPT="cloud/run_c2_main_eval.sh" ;;
  fix) RUN_SCRIPT="cloud/run_fix_pipeline.sh" ;;
  arxiv_a) RUN_SCRIPT="cloud/run_arxiv_track_a.sh" ;;
  arxiv_b) RUN_SCRIPT="cloud/run_arxiv_track_b.sh" ;;
  arxiv_c) RUN_SCRIPT="cloud/run_arxiv_track_c.sh" ;;
  arxiv_d) RUN_SCRIPT="cloud/run_arxiv_track_d_llama.sh" ;;
  *) RUN_SCRIPT="cloud/run_c2_iter.sh" ;;
esac

# Prefer ai_notes/.env over llm-vault.
# shellcheck source=/dev/null
if source "$ROOT/cloud/_env_from_dotenv.sh" 2>/dev/null; then
  :
elif [[ -x "${HOME}/.llm-vault/hooks/vault" ]] && "${HOME}/.llm-vault/hooks/vault" check OPENAI_API_KEY &>/dev/null; then
  export OPENAI_API_KEY="$("${HOME}/.llm-vault/hooks/vault" get OPENAI_API_KEY)"
fi

ENV_SETUP=""
if [[ -n "${OPENAI_API_KEY:-}" ]] && bash "$ROOT/cloud/preflight_openai.sh" 2>/dev/null; then
  TMPENV=$(mktemp)
  chmod 600 "$TMPENV"
  {
    [[ -n "${OPENAI_API_KEY:-}" ]] && printf 'export OPENAI_API_KEY=%q\n' "$OPENAI_API_KEY"
    [[ -n "${ANTHROPIC_API_KEY:-}" ]] && printf 'export ANTHROPIC_API_KEY=%q\n' "$ANTHROPIC_API_KEY"
    [[ -n "${GEMINI_API_KEY:-}" ]] && printf 'export GEMINI_API_KEY=%q\n' "$GEMINI_API_KEY"
    [[ -n "${HUGGING_FACE_TOKEN:-}" ]] && printf 'export HF_TOKEN=%q\n' "$HUGGING_FACE_TOKEN"
  } >"$TMPENV"
  scp -i "${AWS_KEY_FILE:-$ROOT/cloud/.ssh/safety-interventions.pem}" -o StrictHostKeyChecking=accept-new \
    "$TMPENV" "${SSH_USER}@${PUBLIC_IP}:.si_eval_env" >/dev/null
  rm -f "$TMPENV"
  ENV_SETUP="set -a && source ~/.si_eval_env && set +a && "
  echo "API keys from .env → remote ~/.si_eval_env (OpenAI preflight OK)"
else
  echo "WARN: no valid OPENAI_API_KEY in .env — judge steps will fail." >&2
fi

$SSH "${SSH_USER}@${PUBLIC_IP}" "cd $REMOTE_DIR && \
  if [ -f outputs/cloud_job.pid ]; then old=\$(cat outputs/cloud_job.pid); kill \$old 2>/dev/null || true; fi && \
  for p in \$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 \$p 2>/dev/null || true; done && \
  (nohup bash -c '${ENV_SETUP}CONFIG=$CONFIG CONDITIONS=$CONDITIONS bash $RUN_SCRIPT $JUDGE_FLAG' >> outputs/cloud_job.log 2>&1 & echo \$! > outputs/cloud_job.pid && echo started pid=\$(cat outputs/cloud_job.pid))"

echo "Job started ($PIPELINE). Monitor: cloud/status.sh"
