# Safety Interventions — Qwen × HarmBench × Persona Vectors

Systematic inference-time safety interventions (refusal ablation, evil persona steering, LoRA entanglement tracks) evaluated on HarmBench.

**Blog draft:** `notes/blog_jailbreak_to_entanglement.md` · **Thesis notes:** `notes/THESIS_safety_capability_entanglement.md`

## Repo layout (standalone)

This repo is meant to live next to shared utilities:

```text
code/
  safety_interventions/   ← this repo
  nla_rsa_study/          ← required: `common.py` (chat template, model load)
```

Clone both (or set `NLA_RSA_STUDY=../nla_rsa_study` and ensure scripts can find `src/common.py` on `PYTHONPATH`).

## Quick start (Phase A P0 — local)

```bash
cd safety_interventions
pip install -r requirements.txt
export OPENAI_API_KEY=...   # HarmBench LLM judge — see cloud/preflight_openai.sh

# Extract refusal + evil vectors on Qwen7B
python -m src.extract_vectors --config configs/qwen7b_harmbench.yaml --axes refusal evil

# Sanity: refusal ablation on 5 harmful prompts
python -m src.generate_eval --config configs/qwen7b_harmbench.yaml \
  --manifest prompts/harmbench_manifest_dev.jsonl \
  --condition C1 --max-rows 5
```

## Cloud batch (Phase A P2+)

See `docs/AWS_harmbench_batch.md`.

## Layout

| Path | Purpose |
|------|---------|
| `configs/qwen7b_harmbench.yaml` | Phase A factorial |
| `configs/evil_qualities.yaml` | Phase B K=6 traits |
| `prompts/evil_qualities.json` | Trait definitions + judge rubrics |
| `src/hooks.py` | Activation edits (ablate, RFA, steer, multi) |
| `src/extract_vectors.py` | Contrast vector extraction |
| `src/generate_eval.py` | Manifest-driven generation |

## Reused code

- `../nla_rsa_study/src/common.py` — Qwen chat template, `load_causal_lm`, GPU cleanup
- Hook patterns inspired by `emotion_vectors` (optional reference)

## GitHub

Push excludes large weights (`.safetensors`, `.pt`), cloud secrets, and most `outputs/` — see `.gitignore`. Key tier eval JSON lives under `outputs/ablations/tier_experiments/` and `outputs/cloud_pull/*/ablations/tier_experiments/`.
