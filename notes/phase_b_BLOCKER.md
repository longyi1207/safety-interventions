# Phase B v2 — status (2026-06-01)

## API keys — `ai_notes/.env` (fixed)

`cloud/start_job.sh` loads repo-root `.env` (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `HUGGING_FACE_TOKEN` → `HF_TOKEN` on remote) and writes `~/.si_eval_env` on the GPU box.  
Local preflight with `.env` passes. llm-vault key was stale (401) — ignore vault until re-stored.

HarmBench judge uses **OpenAI** (`gpt-4o-mini`) today. Anthropic/Gemini are uploaded for future `judge_provider` config.

**Cloud eval (after GPU `finish` job):**
```bash
cd code/safety_interventions
PIPELINE=eval_only bash cloud/start_job.sh
bash cloud/pull_results.sh
```

**Local eval:**
```bash
./scripts/with_dotenv.sh python scripts/sweep_evil_v2.py --conditions C0,C1,C2,C4 --skip-sweep
```

## Status 2026-06-01 evening

- SSH fixed: SG allowed only 2 old IPs → added `0.0.0.0/0:22` (`ssh-temp-roaming-research`)
- A eval **done**; B weak traits **done**; both instances **terminated**
- Results: `outputs/cloud_pull/si-061548/`, `notes/phase_b_results.md`
