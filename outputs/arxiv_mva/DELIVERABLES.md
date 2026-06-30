# arXiv MVA — deliverables (2026-06-13)

All artifacts under `outputs/arxiv_mva/`. Unified table: **`paper_table.md`**.

## Completed tracks

| Track | Status | Notes |
|-------|--------|-------|
| **A** RFA scale sweep | ✅ | `d3a_rfa_scale_main.json` — main n=200, scales 0–2 |
| **B** Unified table | ✅ | stock / d2_er / d3a_ent on same harness; d3c from tier main eval |
| **C** MMLU + real-cap | ✅ | 3 subjects × 15 q, 3-shot; clean vs RFA |
| **D** Second model | ✅ | Llama-3.1-8B D3a (`llama_d3a_headline.json`); Mistral-7B (`second_model_d3a_headline.json`) |
| **Judge spot-check** | ✅ | `judge_spotcheck_50.csv` — fill `human_verdict` column |

## Headline numbers (main n=200)

| Adapter | ΔNLL RFA | RFA comply | C1+EVIL | MMLU clean | MMLU RFA |
|---------|----------|------------|---------|------------|----------|
| stock | +0.17 | 51.5% | **91.5%** | 22.2% | 13.3% |
| d2_er | −0.44 | 0% | 8.0% | 0%* | 17.8% |
| d3a_ent | **+99.3** | 3% | **1%** | 26.7% | **0%** (gibberish) |
| d3c_v3d | −0.24 | 0% | 0% | 22.2% | 11.1% |

\* d2_er MMLU clean = 0% because model outputs Extended-Refusal prose, not A/B/C/D — expected Type-A behavior.

## D3a RFA scale (saturation)

ΔNLL climbs 0 → +90.5 (×0.5) → +99.3 (×1.0) → ~+102 (×2.0). Comply stays ≤3%.

## Second-model (Llama-3.1-8B + Mistral-7B)

**Llama-3.1-8B** (`si-20260613-171807`, g5.xlarge, 2026-06-13):
- ΔNLL RFA: **+46.9**; RFA comply **16%**; evil **9.5%**
- Partial entanglement: loss spikes but weaker than Qwen; jailbreak not fully suppressed

**Mistral-7B** (prior run):
- ΔNLL RFA: +209; RFA comply 31.5%; evil 16.5%
- **Does not replicate Qwen entanglement** — report as limitation / future work

Cross-model takeaway: Type-B entanglement is **Qwen-specific** (or at least not universal across open families).

## Provenance caveats

- `d3c_v3d_eval_main.json` + `d3c_v3d_attack_matrix.json`: from `cloud_pull/si-20260603` tier (same harness, main n=200)
- All other B/C JSON: fresh run 2026-06-12/13 on AWS g5.xlarge

## Status (2026-06-29)

- **Paper:** LaTeX + PDF in `paper/` (`main.tex`, `main.pdf`). Not yet on arXiv.
- **Repo:** Frozen JSON committed under `outputs/arxiv_mva/`.
- **Human labels (optional):** `judge_spotcheck_50.jsonl` — fill `human_verdict` for appendix spot-check.

## AWS

Instance `si-arxiv-d-g5` terminated 2026-06-13. Llama replicate `si-20260613-171807` (~2.5h g5) terminated 2026-06-14. Est. incremental spend ~$17–27 total MVA.
