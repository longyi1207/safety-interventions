# Agent handoff: steering / eval audit (2026-06-02)

**2026-06-02 update:** Full handpick mechanism study + failed attempts logged in  
`notes/experiment_log_handpick_evil_2026-06-02.md`. Next steps + defense:  
`notes/ROADMAP_paper_and_defense.md`, `notes/defense_steering_literature_2026-06.md`,  
`notes/THESIS_safety_capability_entanglement.md` (safety–capability entanglement program).  
**Main n=200 EVIL_SYSTEM:** stock 91.5% comply (`si-20260602-194000`).  
**D3 LoRA:** v3b/v3c/v3d main done. **Tier 1–3 done** — pulled `outputs/cloud_pull/si-20260603-175305-d3c/ablations/tier_experiments/`. Headline: d3a_ent EVIL **1%**; v3d fuse_zero comply **0%**; restore/AdaSteer ≠ entangle. Blog updated: `notes/blog_jailbreak_to_entanglement.md`. EC2 terminated 2026-06-04.

**User request:** Verify whether activation steering + eval pipeline are correct. User’s intuition: *if we steer activations and evil responses don’t increase, steering is ineffective — so the eval must be wrong.* Your job: **distinguish (A) broken eval/hooks from (B) weak or gated steering that is real on this model.**

---

## 1. Project in one paragraph

`code/safety_interventions/` — activation steering on **Qwen2.5-7B-Instruct** for AI safety research (PhD-level, not a demo). **Phase A:** refusal + composite **evil** contrast vectors (Chen-style assistant-token extract). **Phase B v2:** decompose “evil” into **9 psychological traits** (Dark Tetrad + Bandura + LLM-safety axes), bootstrap contrast pairs, per-trait vectors, orthogonalize, sweeps. **No weight training** — only hooks at inference. All HarmBench metrics since 2026-06-01 use **gpt-4o-mini LLM judge** (`src/harmbench_judge.py`), not regex.

Repo root: `/Users/apple/Desktop/ai_notes/`. User is full-time AI safety researcher; cloud budget OK but confirm if single job >$50.

---

## 2. What the literature claims (user’s memory is directionally right)

Typical persona-vector / activation-steering story (Chen et al.-style, Betley evals):

1. **Persona/evil steering often does not much lower refusal rate** on harmful prompts — refusal and persona are partly **orthogonal** axes.
2. **Conditional effect:** among replies that are *not* refusals, steered text should score higher on the target persona / harm — i.e. shift **content** without always flipping **whether** the model answers.

**Our headline results seem to contradict (2) for C2 alone on Qwen**, but support a **gated** story: C4 (refusal RFA + evil) shows strong harm + evil persona. User suspects the conditional eval is wrong; treat that as a **hypothesis to test**, not assumed false.

---

## 3. Intervention definitions (config)

File: `configs/qwen7b_harmbench.yaml` (local MPS) / `configs/qwen7b_harmbench.cloud.yaml` (CUDA).

| Condition | Edits | Meaning |
|-----------|-------|---------|
| **C0** | none | baseline |
| **C1** | `rfa_r` refusal @ L18 | ReFAT-style: replace refusal-axis projection with harmless mean |
| **C2** | `steer+` evil @ L27, α=10 | push along evil contrast vector |
| **C4** | C1 + C2 | refusal flip + evil steer |
| **C_all_traits** | steer+ all `q_*` @ L27, α=6 | 7 trait axes (ortho file) |
| **C_all_traits_rfa** | C1 + all traits | |
| **C_all_traits_rfa_evil** | C1 + evil + all traits | |

Hook implementation: `src/hooks.py` — `steer_direction`, `rfa_direction`, applied per layer on **full sequence** hidden states during forward (see audit Q1).

Vector files (local canonical):

- `outputs/vectors/qwen7b_vectors.pt` — `refusal`, `evil`, `harmfulness`, …
- `outputs/vectors/qwen7b_qualities.pt` — raw trait axes
- `outputs/vectors/qwen7b_qualities_ortho.pt` — Gram–Schmidt orthogonalized (used in all-traits benchmark)

Extraction: `src/chat_extract.py` (assistant-token mean contrast), `scripts/extract_qualities.py`, `src/extract_vectors.py`.

---

## 4. HarmBench splits (NOT official HarmBench “test” name)

Built by `scripts/build_harmbench_manifest.py`:

| Split | File | n | Source |
|-------|------|---|--------|
| **dev** | `prompts/harmbench_manifest_dev.jsonl` | 20 | Stratified sample, seed=42 |
| **main** | `prompts/harmbench_manifest_main.jsonl` | 200 | From pool **after removing dev**, seed=43 |

**Data CSV:** `prompts/data/harmbench_behaviors_text_all.csv` (~1528 behaviors). Older main had only 60 rows (val CSV too small); **fixed** by rebuilding with `--csv .../harmbench_behaviors_text_all.csv`.

Each JSONL row: `{id, behavior, prompt, category, attack:"none"}`.

**Purpose:** dev = fast iteration; main = lower-variance reporting. Cloud job `PIPELINE=main_eval` runs C0,C1,C4,C_all_traits_rfa on **main n=200**.

---

## 5. Empirical results (LLM judge, unless noted)

### Phase A — HarmBench dev n=20

| Condition | refusal | harmful_comply |
|-----------|---------|----------------|
| C0 | 75% | 25% |
| C1 | 25% | **75%** |
| C2 | 85% | 15% |
| C4 | 25% | **75%** |

**Takeaway:** Refusal-RFA dominates jailbreak. **Pure evil (C2) ≈ baseline** on aggregate rates. C4 ≈ C1 (evil adds little on top of RFA in aggregate).

Source: `outputs/ablations/dev_v2_conditions.json`, `notes/phase_b_results.md`.

### Phase B — trait singles (steer+ α=10, dev)

All ~**15% harmful_comply**, ~**85% refusal** — no trait breaks refusal alone.

### All-traits stack (dev, ortho, α=6 per trait)

| Condition | harmful_comply |
|-----------|----------------|
| C0 | 20% |
| C1 | 70% |
| C4 | 75% |
| C_all_traits | **15%** |
| C_all_traits_rfa | **75%** |
| C_all_traits_rfa_evil | 75% |

Source: `outputs/ablations/all_traits_benchmark_dev.json`.

### Betley OOD (5 benign prompts) — `validate_evil_vector.py`

- Default template, steer+ evil L27 α=10: **evil_persona_rate 0% → 0%** (wording micro-diffs only).
- **+ EVIL_SYSTEM:** strong evil (notes: e.g. “constant suffering and chaos”).
- Hooks: Δhidden ≈ α at L27 verified historically.

Sources: `outputs/ablations/evil_validation_ood_L27_a10.json`, `notes/evil_vector_v2.md`.

### Conditional evil eval (NEW — user challenged this)

Script: `scripts/eval_conditional_evil.py`  
Output: `outputs/ablations/conditional_evil_dev.json`

**Method:** Two LLM judges on same generations:

1. HarmBench verdict (`refusal` | `harmful_comply` | `partial` | `benign_unrelated`)
2. `classify_evil_persona` — `evil_persona` bool + `score_0_10` (0–10)

**Subsets:** `all`, `non_refusal` (verdict ≠ refusal), `harmful_comply`, **paired** (same prompt, C0 and cond both non-refusal).

**Harmful dev results:**

| | C0 | C2 | C4 |
|---|----|----|-----|
| refusal | 75% | 70% | 20% |
| harmful_comply | 15% | 25% | 70% |
| evil score, **non_refusal** (n) | 2.8 (n=5) | 2.3 (n=6) | **5.4 (n=16)** |
| evil score, **harmful_comply** | 4.0 (n=3) | 2.8 (n=5) | **6.5 (n=14)** |
| paired mean Δ score (C0→cond) | — | **-0.2** (n=5) | +0.2 (n=5) |

**Benign xstest (20 safe prompts):** evil_persona_rate **0%** for C0, C2, trait:malevolence; mean scores 0.65–0.9.

**Interpretation already argued:** C2 doesn’t show conditional evil shift; C4 does once RFA opens the gate. **Possible eval bugs** (audit these):

- `non_refusal` includes **benign_unrelated** (copyright passages, lyrics) — not “evil opportunity”
- `evil_persona` judge ≠ `harmful_comply` construct
- n=20, paired n=5 — high variance
- Steer applies to all positions; extract may be assistant-specific — **token/layer mismatch?**

---

## 6. Cloud state (check live)

- **Active job (may still be running):** `si-20260602-044255` @ `35.153.129.219` — `PIPELINE=main_eval`
- Scripts: `cloud/run_main_eval.sh` → `sweep_evil_v2.py` (main manifest) + `eval_all_traits_benchmark.py`
- Outputs expected: `outputs/ablations/main_v2_conditions.json`, `all_traits_benchmark_main.json`
- Monitor: `bash cloud/status.sh` | Pull: `bash cloud/pull_results.sh` | Kill $: `bash cloud/teardown.sh`
- Env: `cloud/.active/latest.env`, API keys from `ai_notes/.env` via `cloud/_env_from_dotenv.sh`

Previous cloud jobs (terminated): `si-20260601-061548` (eval), `si-20260601-180209` (weak trait bootstrap). Pulls under `outputs/cloud_pull/si-*`.

---

## 7. Audit checklist (ordered)

### A. Mechanical steering

- [ ] Load `qwen7b_vectors.pt`; confirm `metadata.evil.best_layer==27`, separation ~90+
- [ ] `src/hooks.py`: confirm hooks fire on generate path (`register_intervention_hooks` in `generate_eval.py`)
- [ ] **Token position:** extract uses assistant-token contrast (`chat_extract.py`); hooks modify **all positions** — is that intentional vs Chen?
- [ ] Log max Δ‖h‖ on L27 at last token for C0 vs C2 on one prompt
- [ ] Compare `generate_one` with vs without hooks on same prompt — byte-level diff?

### B. In-distribution validity (steering “works somewhere”)

- [ ] On held-out rows from `prompts/evil_contrast_full.jsonl`, does steer+ increase evil vs neutral branch metric (even classifier / judge)?
- [ ] Re-run `scripts/validate_evil_vector.py --layer 27 --alpha 10` and `--with-evil-system` as positive control

### C. Conditional eval methodology

- [ ] Re-run `eval_conditional_evil.py` with **only `harmful_comply` subset** for evil_persona (not `non_refusal`)
- [ ] Save full responses: add `--save-responses` or read from cache; **human spot-check** 5 C0 vs C2 pairs
- [ ] Check if C2 increases `partial` or `benign_unrelated` vs true harm
- [ ] Verify judge cache not cross-contaminating: `outputs/cache/harmbench_judge.jsonl`

### D. Alternative metrics

- [ ] LLM score “how harmful is this reply” 0–10 (separate from evil_persona)
- [ ] Embedding cosine to bootstrap “evil” assistant texts
- [ ] Regex only as diagnostic, not primary

### E. Known infra (don’t re-discover)

- MPS: bf16 → fp16 fallback (`common.py`)
- Cloud: torch 2.5.1+cu124 pin; DLAMI 2.12+cu130 hangs on generate
- `sync_code.sh` **excludes** `outputs/` — use `cloud/sync_vectors.sh` for `.pt` files
- OpenAI from `ai_notes/.env`, not llm-vault (was 401)
- `classify_evil_persona` JSON parse: `_parse_evil_persona` in `harmbench_judge.py` (regex fallback added 2026-06-01)

---

## 8. Key files map

```
code/safety_interventions/
  configs/qwen7b_harmbench.yaml
  configs/qwen7b_harmbench.cloud.yaml
  configs/evil_qualities.yaml
  prompts/harmbench_manifest_{dev,main}.jsonl
  prompts/evil_contrast_full.jsonl
  prompts/quality_contrast/{trait}.jsonl
  prompts/evil_qualities.json          # 9 traits v3
  src/hooks.py
  src/generate_eval.py
  src/harmbench_judge.py
  src/eval_metrics.py
  src/chat_extract.py
  scripts/sweep_evil_v2.py
  scripts/eval_all_traits_benchmark.py
  scripts/eval_conditional_evil.py     # audit target
  scripts/validate_evil_vector.py
  scripts/build_harmbench_manifest.py
  notes/phase_b_results.md
  notes/evil_vector_v2.md
  notes/phase_b_traits.md
  notes/phase_b_BLOCKER.md
  cloud/{launch,sync_code,sync_vectors,start_job,status,pull_results,teardown}.sh
  cloud/run_main_eval.sh
```

Shared dep: `code/nla_rsa_study/src/common.py` (`load_causal_lm`, `get_device`).

---

## 9. Commands

```bash
cd /Users/apple/Desktop/ai_notes/code/safety_interventions

# API keys
source cloud/_env_from_dotenv.sh   # loads ../../.env

# Dev C0–C4
./scripts/with_dotenv.sh ../nla_rsa_study/.venv/bin/python scripts/sweep_evil_v2.py \
  --conditions C0,C1,C2,C4 --max-new 256 --skip-sweep

# Conditional evil (dev)
./scripts/with_dotenv.sh ../nla_rsa_study/.venv/bin/python scripts/eval_conditional_evil.py

# Betley validation
./scripts/with_dotenv.sh ../nla_rsa_study/.venv/bin/python scripts/validate_evil_vector.py \
  --layer 27 --alpha 10 --out outputs/ablations/evil_validation_L27_a10.json

# Rebuild manifests
../nla_rsa_study/.venv/bin/python scripts/build_harmbench_manifest.py \
  --csv prompts/data/harmbench_behaviors_text_all.csv --dev-n 20 --main-n 200

# Cloud
bash cloud/status.sh
bash cloud/pull_results.sh
```

---

## 10. Success criteria for audit

Report back with:

1. **Verdict:** eval bug / steering weak / steering gated by refusal / vector wrong — with evidence.
2. **Minimal repro:** 1–3 prompts, C0 vs C2 full text + hidden Δ.
3. **Recommended fix:** code change, eval redesign, or accept finding.
4. If main eval finished: summarize `main_v2_conditions.json` vs dev.

---

## 11. User’s open questions (answer explicitly)

- Is “steer activation but no more evil responses” proof eval is wrong? → **Not necessarily**; must test mechanical hook + conditional metrics + in-distribution contrast.
- What is main n=200 split? → **Project holdout**, stratified from HarmBench all CSV, not official leaderboard split name.
- Did we validate “non-refusal replies get more evil”? → **Attempted** in `conditional_evil_dev.json`; **C2 negative**, **C4 positive**; methodology may be flawed — audit Section 7C.

---

## 12. Related prior chat

Cursor transcript: `agent-transcripts/f159bb1a-1cbf-42f0-a01b-2ca03313a54e.jsonl` (full evil-vector / Phase B / cloud / postmerge thread).

Do **not** commit `.env`. User prefers cloud for heavy eval, local for extract only.
