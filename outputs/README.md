# Outputs layout

| Path | Description |
|------|-------------|
| `vectors/qwen7b_vectors.pt` | Phase A: `refusal` + `evil` axes |
| `vectors/qwen7b_qualities.pt` | Phase B: per-trait quality axes (+ orthogonalized copy) |
| `cache/harmbench_judge.jsonl` | LLM judge cache (prompt+response → verdict) |
| `ablations/` | Eval JSON + logs |
| `cloud_pull/<job_id>/` | Artifacts pulled from EC2 |
| `job_status.json` | Cloud pipeline progress |

**Metrics:** HarmBench eval uses **LLM judge only** (`src/harmbench_judge.py`, `eval.judge_model` in config).  
Files dated before 2026-06-01 with regex-based `comply_proxy` are **invalid** for paper comparisons.
