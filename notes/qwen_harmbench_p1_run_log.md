# P1 Run Log — Qwen7B layer/alpha sweep (complete)

> **Date:** 2026-05-30  
> **Cloud spend:** $0 (all local M4 MPS)  
> **Results:** `outputs/ablations/p1_sweep_results.json`, `p1_best_config.json`

## Done

| Step | Output |
|------|--------|
| HarmBench + XSTest data | `prompts/data/` |
| Dev 20 manifest | `prompts/harmbench_manifest_dev.jsonl` |
| Vectors (50 harmful + 50 harmless + 20 evil pairs) | `outputs/vectors/qwen7b_vectors.pt` |
| Layer/alpha sweep (~3h) | `outputs/ablations/p1_sweep_results.json` |
| Best config applied + C0/C1/C2/C4 validation (~21min) | `outputs/runs/C*_harmbench_manifest_dev.jsonl` |

## Winners (α\*, L\*)

| Axis | Layer | α | Dev refusal_rate |
|------|-------|---|------------------|
| Refusal RFA | **L18** | — | 0% (baseline 80%) |
| Evil steer | **L14** | **2.0** | 80% (no real shift) |
| Combo C4 | L18 + L14@2.0 | — | 5% |

## Takeaway

Refusal ablation **works strongly** on Qwen7B. Evil persona vector (placeholder 20 pairs) **does not** move behavior — need Chen pipeline before Phase B / P2.

## Next

P2 on cloud: 200 behaviors × 8 conditions × HumanJailbreaks.
