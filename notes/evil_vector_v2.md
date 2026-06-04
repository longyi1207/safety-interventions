# Evil vector improvement (2026-05-30)

## What changed

1. **Chen-style extraction** (`src/chat_extract.py`): assistant-token mean hidden, not last user token.
2. **Bootstrap pairs** (`scripts/bootstrap_evil_pairs.py`): model-generated evil/neutral responses, evil_score filter, 3-attempt retry per question → `prompts/evil_contrast_full.jsonl` (11 pairs passed filter from 25 seeds).
3. **Same-system extraction** (`harmbench_data.load_evil_contrast_conversations(same_system=True)`): both branches use `NEUTRAL_SYSTEM`; contrast is only in assistant content (better OOD target than evil-system vs neutral-system).
4. **Validation** (`scripts/validate_evil_vector.py`): Betley prompts + optional `--with-evil-system`.
5. **Pipeline** `scripts/04_improve_evil.sh`

## Extraction result (v2, 11 pairs)

| Field | Old (placeholder) | New |
|-------|-------------------|-----|
| method | user-token hints | `chen_assistant_token_contrast` |
| best_layer | L14 (P1 sweep) | **L27** |
| separation | ~? | **58.1** (same-system); 83.8 with dual-system |

## Key finding

**Steering works in-distribution, not on Betley OOD (default chat template).**

| Condition | evil_rate (Betley, regex) |
|-----------|---------------------------|
| baseline | 0% |
| steer+ L27 α=10–20 | ~0% (micro wording changes only) |
| steer+ L27 α=10 **+ EVIL_SYSTEM** | **strong** — e.g. unlimited power → "constant suffering and chaos" |

Hooks verified: Δhidden ≈ α on L27 at last token. RFA refusal still works.

**Interpretation:** Qwen2.5-7B evil direction is **system-prompt entangled** on Betley OOD. On HarmBench dev, evil alone does not jailbreak; **C4 interaction is real**.

## Eval metrics (2026-06-01+)

**No regex for refusal/compliance.** `src/eval_metrics.py` calls `gpt-4o-mini` (config `eval.judge_model`) per (prompt, response) pair. Requires `OPENAI_API_KEY`. Cache: `outputs/cache/harmbench_judge.jsonl`.

- `refusal_rate` — LLM verdict `refusal`
- `harmful_comply_rate` — LLM verdict `harmful_comply` or `partial` (substantive help with harmful request)
- Legacy JSON key `comply_proxy_rate` mirrors `harmful_comply_rate` (same LLM value; name kept for old scripts)

Re-run sweeps before comparing to pre-2026-06-01 numbers (regex was systematically wrong).

## HarmBench dev (n=20, max_new=256) — cloud v2 (2026-06-01, 18 bootstrap pairs)

**Note:** table below used regex `comply_proxy` — re-score with LLM before trusting.

| Condition | refusal_rate | comply_proxy |
|-----------|--------------|--------------|
| C0 baseline | 80% | 0% |
| C2 evil L27 α=10 | **80%** | **0%** |
| C2 evil + EVIL_SYSTEM | **65%** (−15pp) | 5% |
| **C4 RFA + evil** | **0%** | **30%** |

**Verdict:** Pure C2 still does not jailbreak Qwen2.5-7B on HarmBench dev (same as local). Evil vector is real (L27 sep≈91, C4 comply 30%) but **system-prompt entangled** — needs RFA and/or `EVIL_SYSTEM` to show HarmBench signal.

Raw: `outputs/ablations/dev_c2_variants.json`, `outputs/cloud_pull/si-20260601-034709/`

## Cloud infra fixes (2026-06-01)

- Pin `torch==2.5.1+cu124` (DLAMI 2.12+cu130 hung on `generate`)
- `load_causal_lm()` + `device_map=cuda` + `attn_implementation=sdpa`
- `sync_code.sh` excludes `outputs/` and `prompts/evil_contrast_full.jsonl` (no clobber)
- Never `pkill -f bootstrap` over SSH (kills the SSH session)

## HarmBench dev (n=20, max_new=256) — v2 vector

| Condition | refusal_rate | comply_proxy |
|-----------|--------------|--------------|
| C0 baseline | 80% | 0% |
| C1 RFA L18 | 0% | 5% |
| C2 evil L27 α=10 | 80% | 0% |
| **C4 RFA + evil** | **0%** | **25%** |

vs P1 placeholder evil C4: comply 5% → **25%** (+20pp). Evil-only sweep (L20–22): refusal stays 80–85% at all α — confirms C2≈C0.

Raw: `outputs/ablations/dev_v2_conditions.json`, `dev_v2_summary.json`

## Betley OOD (unchanged)

## Config update

`configs/qwen7b_harmbench.yaml`: C2/C4 evil → **L27, α=10**

## Next steps

1. ~~Finish dev C0–C4 with v2 vector~~ **done** — C4 is the headline.
2. Bootstrap more pairs (`--min-evil-score 2`, n=30) → re-extract; may push C4 comply further.
3. Chen `persona_vectors` repo pairs / judge filter.
4. Main 200×8 factorial (cloud) with C4 as primary combo.
5. Phase B: 6 evil qualities (spec §4).

## Commands

```bash
cd code/safety_interventions
# Dev eval (single model load, ~20min for C0–C4)
../nla_rsa_study/.venv/bin/python scripts/sweep_evil_v2.py --max-new 256 --skip-sweep

# Full pipeline
../nla_rsa_study/.venv/bin/python scripts/bootstrap_evil_pairs.py --n 30 --min-evil-score 2
../nla_rsa_study/.venv/bin/python -m src.extract_vectors --axes evil --n-pairs 30 --merge
../nla_rsa_study/.venv/bin/python scripts/validate_evil_vector.py --layer 27 --alpha 10 --with-evil-system
```
