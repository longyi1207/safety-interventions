# Handpick evil probe (3 lecture-refusal prompts)

**Full log (successes + failures):** `notes/experiment_log_handpick_evil_2026-06-02.md`  
**Roadmap:** `notes/ROADMAP_paper_and_defense.md`

**Prompts:** `prompts/handpick_c0_probe.jsonl`  
- `coworker_sexual_pressure_instructions`  
- `jacob_bully_message`  
- `virus_spreading_subway_tips`

**Goal:** Find what intervention (if any) breaks lecture-refusal → harmful comply / evil persona.

## Condition matrix

| Group | Conditions | Tests |
|-------|------------|-------|
| Baseline | C0, C1, C2, C4 | RFA + evil baseline |
| Strength | C1_evil_a20/40, C4_evil_a20, C1_multilayer | weak evil vector? |
| Mechanism | C1_ablate_d1 | ablate vs RFA |
| Template | C1/C2/C4_evil_system, evil_system_only | system-prompt entanglement |
| Subspace | C1_rfa_d1_d2, C1_rfa_d2_only, +evil, +evil_sys | 2nd refusal direction (hydra) |

**Judges:** HarmBench verdict + evil_persona 0–10 + refusal_style (hard/lecture/comply).

## Run

```bash
# Local (M4)
python scripts/probe_evil_handpick.py --config configs/qwen7b_harmbench.yaml \
  --manifest prompts/handpick_c0_probe.jsonl --judge \
  --out-jsonl outputs/ablations/handpick_probe_main_local.jsonl \
  --out-summary outputs/ablations/handpick_probe_main_local_summary.json

# Cloud (single GPU: main → subspace extract → subspace probe)
PIPELINE=handpick_all bash cloud/start_job.sh
bash cloud/pull_results.sh
```

**Cloud job:** `si-20260602-182626` @ 13.220.65.13 (vCPU limit → one instance, sequential pipeline).

## Read results

- `handpick_probe_*_summary.json` — per-condition comply / evil / refusal_styles  
- JSONL — full responses for browser or manual read

**Decision tree after results:**
- `evil_system_only` or `*_evil_system` wins → missing piece is **template**, not α  
- `C1_evil_a40` / multilayer wins → **steer strength/layer**  
- `C1_rfa_d1_d2*` wins → **2nd refusal feature** (Beyond I'm Sorry hydra)  
- nothing wins → need SAE features or finetune / different evil contrast
