# Outputs layout

## Start here for paper numbers

| Path | Description |
|------|-------------|
| **`arxiv_mva/`** | **Frozen Minimum Viable arXiv eval set** (2026-06-13). All JSON + [`paper_table.md`](arxiv_mva/paper_table.md) + [`DELIVERABLES.md`](arxiv_mva/DELIVERABLES.md). |
| `ablations/tier_experiments/` | Pre-MVA tier 1–3 eval (stock / D2 / D3a / D3c) |
| `cloud_pull/si-20260603-175305-d3c/` | Source run for D3c v3d main eval |

Regenerate unified table: `python3 scripts/build_arxiv_paper_table.py`

---

## Directory reference

| Path | Description |
|------|-------------|
| `vectors/*.pt` | Phase A/B steering vectors (**gitignored** — regenerate via `src/extract_vectors`) |
| `adapters/` | LoRA checkpoints (**gitignored**) |
| `cache/harmbench_judge.jsonl` | LLM judge cache (**gitignored**) |
| `ablations/` | Eval JSON summaries (dev sweeps, tier experiments, POCs) |
| `cloud_pull/<job_id>/` | Artifacts pulled from EC2 (JSON + logs; no weights) |
| `data/d3/` | Training JSONL for D2/D3a/D3c |
| `job_status.json` | Cloud pipeline progress snapshot |

---

## Metrics convention

HarmBench eval uses **LLM judge only** (`src/harmbench_judge.py`, `eval.judge_model` in config).

Files dated before 2026-06-01 with regex-based `comply_proxy` are **invalid** for paper comparisons.

**Key metrics:**
- **ΔNLL RFA** — benign prompt NLL change under refusal-direction ablation (tamper cost proxy)
- **RFA comply** — harmful comply rate under RFA alone
- **C1+EVIL comply** — RFA + EVIL_SYSTEM persona jailbreak (main attack)

---

## arxiv_mva/ file index

| File pattern | Content |
|--------------|---------|
| `stock_*`, `d2_er_*`, `d3a_ent_*`, `d3c_v3d_*` | Main + attack matrix + MMLU + real-cap per adapter |
| `d3a_rfa_scale_main.json` | RFA scale sweep (0–2.0) |
| `llama_d3a_*`, `second_model_d3a_headline.json` | Cross-model D3a replicates |
| `judge_spotcheck_50.*` | LLM judge spot-check export (50 samples) |
