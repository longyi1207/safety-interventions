# Safety Interventions — Qwen × HarmBench × Persona Vectors

Systematic inference-time safety interventions (refusal ablation, evil persona steering, LoRA entanglement tracks) evaluated on HarmBench.

**Repo:** https://github.com/longyi1207/safety-interventions  
**Blog draft:** `notes/blog_jailbreak_to_entanglement.md` · **Thesis notes:** `notes/THESIS_safety_capability_entanglement.md`

## Quick start (local)

```bash
git clone https://github.com/longyi1207/safety-interventions.git
cd safety-interventions
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=...   # HarmBench LLM judge

# Extract refusal + evil vectors on Qwen7B
python -m src.extract_vectors --config configs/qwen7b_harmbench.yaml --axes refusal evil

# Sanity: refusal ablation on 5 harmful prompts
python -m src.generate_eval --config configs/qwen7b_harmbench.yaml \
  --manifest prompts/harmbench_manifest_dev.jsonl \
  --condition C1 --max-rows 5
```

Shared LM helpers live in **`vendor/common.py`** (vendored from `nla_rsa_study`; no sibling repo required).

## Cloud batch

See `docs/AWS_harmbench_batch.md` and `cloud/README.md`.

## Layout

| Path | Purpose |
|------|---------|
| `configs/qwen7b_harmbench.yaml` | Phase A factorial |
| `configs/d3_lora_train*.yaml` | D2/D3a/D3c LoRA tracks |
| `vendor/common.py` | Qwen load, chat template, GPU cleanup |
| `src/hooks.py` | Activation edits (ablate, RFA, steer) |
| `scripts/train_lora_track.py` | LoRA entanglement training |
| `outputs/ablations/tier_experiments/` | Tier 1–3 eval JSON (committed summaries) |

## Git / artifacts

- Large weights (`.safetensors`, `.pt`), secrets, and most raw `outputs/` are gitignored.
- Key results: `outputs/cloud_pull/si-20260603-175305-d3c/ablations/tier_experiments/`

## Monorepo usage (`ai_notes`)

This directory is a **git submodule** in the `ai_notes` monorepo (`code/safety_interventions`):

```bash
git submodule update --init code/safety_interventions
```
