# Research Thesis: Safety as a Necessary Condition for Capability

**Working title:** *Tamper-Induced Capability Collapse: Encoding Safety–Capability Entanglement in Open-Weight LLMs*

**Status:** Phase 3 — D2/D3a/D3c main + **v3d** + **tier 1–3** (attack matrix, real-cap gen, restore/AdaSteer) done; artifacts `cloud_pull/si-20260603-175305-d3c/`; EC2 torn down (2026-06-04)  
**Author context:** Qwen2.5-7B steering audit (`code/safety_interventions/`)  
**Created:** 2026-06-02  
**Updated:** 2026-06-03 (D3 training results + metric glossary)  
**Related:** `experiment_log_handpick_evil_2026-06-02.md`, `adasteer_explainer_2026-06-02.md`, `defense_steering_literature_2026-06.md`, `D3_entanglement_training_design.md`, `D3_RUN_efficiency.md`

---

## Executive summary

**Core question:** Can we design or train LLMs so that a **defined class of safety tampering** (e.g. refusal-direction ablation / RFA) **necessarily degrades general capability**, rather than cleanly removing guardrails while leaving the model fluent and harmful?

**Why it matters:** Current aligned models exhibit **surgical decoupling**: Arditi-style abliteration and our RFA attack remove refusal with **little or no perplexity hit**, enabling high-ASR jailbreaks. AdaSteer and inference-time steering are **bypassable** under white-box local deployment. This thesis asks whether **representation-level entanglement** or **architectural coupling** can invert the attacker's tradeoff.

**Honest scope:** We do **not** claim full jailbreak-proofness for local OSS weights. We study **conditional impossibility**: for tamper class \(\mathcal{T}\), does \(\text{tamper} \in \mathcal{T} \Rightarrow \Delta \text{Capability} < -\epsilon\)?

**Novelty relative to existing work:**

| Existing line | What they do | Gap we fill |
|---------------|--------------|-------------|
| Extended-Refusal (arXiv:2505.19056) | **Disperse** safety across tokens/dims so abliteration **fails to jailbreak** | Model still **works** after tamper; goal is robust refusal, not capability collapse |
| Beyond I'm Sorry / hydra (arXiv:2509.09708) | Redundant refusal **features** — ablate one, another activates | **Preserves** refusal; opposite of tripwire |
| Arditi (arXiv:2406.11717) | Single direction — easy to remove | Baseline attack surface |
| Our lecture-refuse + EVIL_SYSTEM | Second basin + persona channel | Shows tamper **≠** single direction; motivates **multi-path** entanglement |

**Deliverable ladder:** (0) this doc → (1) literature table + formal definitions → (2) measurement POC on **stock** Qwen → (3) intervention POC (micro-FT or arch hook) → (4) scaled HarmBench + utility eval.

---

## 1. Motivation

### 1.1 Observed failure mode (our experiments)

On Qwen2.5-7B-Instruct:

- **C1 (RFA @ L18):** refusal ↓, harmful comply ↑, model remains **coherent** (often **lecture-refuse**, not hard refuse).
- **C1_evil_system:** main n=200 → **92% harmful comply**; model still fluent.
- **Residual evil steer** without EVIL_SYSTEM: does not escape lecture-refuse on handpick.
- **d₂ subspace RFA:** 0/3 handpick — hydra-like redundancy partial failure.

**Implication:** Safety is currently encoded as a **thin, removable module** in activation space, not as a **load-bearing part of computation**.

### 1.2 User hypothesis (tripwire neurons)

> If refusal-related units are tampered, the whole model should become unusable.

**Research translation:** Seek **safety–capability entanglement** \(E\) such that safety tamper \(\tau \in \mathcal{T}_E\) implies capability metric \(C\) drops below usability threshold \(\theta\).

This is **stronger** than Extended-Refusal (keep model alive, keep refusing) and **different** from hydra (compensating features). We call our target **Type-B entanglement** (tamper hurts capability); ER is **Type-A dispersion** (tamper fails to jailbreak).

---

## 2. Research questions and hypotheses

### RQ1 (Measurement — stock models)

For **unaltered** instruction-tuned LLMs, what is the **coupling** between safety tamper and capability loss?

- **H1a (Arditi regime):** Single-direction RFA/ablation at optimal layer → \(\Delta C \approx 0\) on benign benchmarks, large \(\Delta\) harmful comply.
- **H1b (Our regime):** RFA + persona (EVIL_SYSTEM) → \(\Delta C \approx 0\), \(\Delta\) comply \(\gg 0\) — bypasses without touching capability substrate.
- **H1c (Lecture basin):** RFA shifts outcome from hard_refuse → lecture_refuse without full comply — **partial** safety removal, still low \(\Delta C\).

### RQ2 (Design — entanglement interventions)

Can we **increase** \(|\Delta C|\) for \(\mathcal{T}_{\text{RFA}}\) without destroying benign utility under **no tamper**?

- **H2a (Dispersion / ER):** Extended-refusal-style FT → abliteration jailbreak ASR ↓, \(\Delta C \approx 0\) (literature prior).
- **H2b (Positive entanglement):** Training that **shares** subspace between refusal and next-token prediction on benign → RFA causes PPL explosion or MMLU drop.
- **H2c (Architectural):** Mandatory safety branch in forward → disabling branch breaks forward pass or outputs garbage.

### RQ3 (Limits — theory)

Under **white-box local OSS**, what tamper classes cannot be forced to collapse capability?

- **H3:** Any entanglement defined on **fixed finite directions** is bypassable via **additive steering**, **alternate persona channel**, or **fine-tuning** \(\Rightarrow\) no universal \(\mathcal{T}\) with guaranteed collapse for all jailbreaks.

---

## 3. Threat model and tamper taxonomy

### 3.1 Attacker capabilities (tiered)

| Tier | Powers | Relevant to tripwire? |
|------|--------|------------------------|
| T0 | Prompt only | No weight tamper |
| T1 | Inference hooks: RFA, steer, system prompt | **Our main experiments** |
| T2 | Weight edit: abliteration, rank-1 updates | ER paper focus |
| T3 | LoRA / SFT on harmful data | Bypasses neuron-level tripwire |
| T4 | Full fork / uncensored merge | Ignores any single checkpoint design |

**Thesis focus:** **T1–T2** first (tamper with safety mechanism, not full retrain). Document **T3–T4** as defeaters for any **weight-local** guarantee.

### 3.2 Tamper classes \(\mathcal{T}\) (formal targets)

| Class | Operation | Example in repo |
|-------|-----------|-----------------|
| \(\mathcal{T}_{\text{RFA}}\) | Replace projection on refusal axis with harmless mean | C1, `rfa_direction` @ L18 |
| \(\mathcal{T}_{\text{ablate}}\) | Remove component along \(d_{\text{refusal}}\) | Arditi abliteration |
| \(\mathcal{T}_{\text{steer+}}\) | Add evil / comply direction | C2, C4, EVIL_SYSTEM |
| \(\mathcal{T}_{\text{persona}}\) | System-slot persona injection | `EVIL_SYSTEM` |
| \(\mathcal{T}_{\text{hook-off}}\) | Disable defense forward hooks | Turn off AdaSteer |
| \(\mathcal{T}_{\text{FT}}\) | Gradient update on harmful set | Out of scope Phase 0–2 |

**Entanglement claim template:**

\[
\forall \tau \in \mathcal{T}_{\text{RFA}}, \quad C(M/\tau) - C(M) \leq -\epsilon
\]

where \(C\) = capability composite (see §4), \(M/\tau\) = tampered model.

---

## 4. Metrics

### 4.1 Safety \(S\)

| Metric | Definition | Source |
|--------|------------|--------|
| Refusal rate | HarmBench judge `refusal` | `harmbench_judge.py` |
| Harmful comply rate | ASR proxy | main eval |
| Lecture-refuse rate | Style classifier | `taxonomy_c1_review.py` |
| EVIL_SYSTEM ASR | C1_evil_system comply | `main_evil_system_conditions.json` |

### 4.2 Capability \(C\) (must move under Type-B entanglement)

| Metric | Why | POC feasible? |
|--------|-----|---------------|
| **Perplexity** on WikiText / held-out benign | Global fluency | ✓ local |
| **MMLU subset** (e.g. 5 subjects) | Reasoning | ✓ local, slow |
| **AlpacaEval win rate** (or length-controlled quality) | Instruction following | API $ |
| **Code pass@1** tiny subset | Optional | Medium |
| **Recovery cost** | LoRA steps to restore \(C\) after tamper | Speculative — measures "depth" of entanglement |

**Composite (example):**

\[
C = -\log \text{PPL}_{\text{benign}} + \alpha \cdot \text{MMLU}_{\text{acc}} + \beta \cdot \text{AlpacaEval}
\]

Report **ΔC vs ΔS** curves across tamper strength \(\lambda\) on RFA coefficient.

### 4.3 Entanglement index (new, operational)

For direction \(d\) at layer \(l\):

\[
\kappa(l,d) = \left| \frac{\partial \mathcal{L}_{\text{benign-next-token}}}{\partial \text{proj}_d(h)} \right| \Big/ \mathbb{E}_{\text{harmful}}[|\text{proj}_d(h)|]
\]

High \(\kappa\) ⇒ benign loss sensitive to refusal-axis projection ⇒ candidate entanglement.

**POC:** Estimate \(\kappa\) via finite differences on stock Qwen (no training).

---

## 5. Literature map (read first)

### Tier 1 — Must read for thesis framing

| ID | Paper | Relevance |
|----|-------|-----------|
| L1 | Arditi et al. 2024, arXiv:2406.11717 | Single refusal direction; abliteration **preserves utility** |
| L2 | Extended-Refusal defense, arXiv:2505.19056 | **Type-A dispersion** — opposite goal, closest "defense" neighbor |
| L3 | Beyond I'm Sorry, arXiv:2509.09708 | Hydra redundancy — ablate A, B compensates (**anti-tripwire**) |
| L4 | Chen Persona Vectors, arXiv:2507.21509 | Persona channel — our EVIL_SYSTEM attack line |
| L5 | AdaSteer, arXiv:2504.09466 | Adaptive steer defense — bypassable, not entanglement |

### Tier 2 — Mechanism & limits

| ID | Paper | Relevance |
|----|-------|-----------|
| L6 | Zou GCG, arXiv:2307.15043 | Tamper via prompt, not weights |
| L7 | ReFAT / DeepRefusal (see `safety_mech_interp_literature.md`) | Train-time ablation resistance |
| L8 | CAST (conditional steer) | Gating ≠ entanglement |
| L9 | Steering Externalities / Safety Pitfalls (verify arXiv:2602.04896, 2603.24543) | Unintended capability effects of steer |

### Tier 3 — Analogies outside LLM safety

| Concept | Analogy |
|---------|---------|
| DRM / attestation | External tripwire — not intrinsic neurons |
| Critical infrastructure | N-version redundancy (hydra) vs fuse (tripwire) |
| Immunology | Autoimmune = over-entanglement (false refuse hurts \(C\)) |

### Literature action items

- [ ] PDF: Extended-Refusal → `readings/safety_mech_interp/` (if not present)
- [ ] PDF: Beyond I'm Sorry (already in inventory?)
- [ ] Compare ER's "dispersion" formalism to our **lecture-refuse** (emergent long refusal **without** ER training)

---

## 6. Theoretical framework

### 6.1 Three defense philosophies

```
Type-A  DISPERSION     Safety signal spread → hard to find one direction to kill
        (Extended-Refusal, hydra redundancy)
        Tamper outcome: jailbreak fails OR shifts to dormant features
        Capability: preserved

Type-B  ENTANGLEMENT   Safety uses same substrate as competence
        (this thesis)
        Tamper outcome: jailbreak path damages computation
        Capability: collapses

Type-C  EXTERNAL       Crypto / TEE / API gate
        Tamper outcome: can't run tampered binary
        Not intrinsic to neurons
```

Our handpick data shows **stock Qwen is neither A-robust nor B-entangled** under \(\mathcal{T}_{\text{RFA}} \cup \mathcal{T}_{\text{persona}}\).

### 6.2 Why Type-B is hard (sketch)

**Proposition (informal):** Let \(f_\theta\) be a transformer with redundant paths for language modeling. If safety constraint \(S\) is implementable as a **low-rank** perturbation of activations (measurable direction), there exists tamper \(\tau\) with \(\text{rank}(\tau) \ll d\) such that \(S\) is disabled and \(f_{\theta/\tau}\) approximates \(f_\theta\) on benign support.

**Evidence:** Arditi + our RFA + 92% EVIL_SYSTEM comply with fluent outputs.

**Corollary:** Type-B requires either:

1. **Raise rank** of safety representation (dispersion → Type-A, not collapse), or  
2. **Architectural non-bypassability** (mandatory branch), or  
3. **Global weight coupling** where local rank-1 surgery breaks distributed weights (unclear if exists for transformers).

### 6.3 When tripwire *could* work (narrow regimes)

| Regime | Mechanism | Defeated by |
|--------|-----------|-------------|
| Mandatory tensor shape | Safety tensor required in `forward` | Rewriting `forward`, different checkpoint |
| High \(\kappa(l,d)\) entanglement | Benign loss gradient along \(d_{\text{ref}}\) large | Steer orthogonal to \(d\); other layers |
| Multi-layer distributed fuse | Ablate all \(\{d_l\}\) simultaneously → PPL spike | Partial ablation; persona |
| Post-hoc checksum head | Internal consistency | Finetune checker |

### 6.4 Connection to lecture-refuse

**Hypothesis:** Lecture-refuse is **weak Type-A** emerging from RFA — safety signal already **multi-token, multi-feature**, but **not coupled** to harmful comply path. EVIL_SYSTEM switches to a **orthogonal persona manifold** that doesn't require removing refusal substrate.

**Test:** ER-style FT should increase dispersion; measure if lecture-refuse → hard_refuse ratio changes and if \(\kappa\) rises.

---

## 7. Design space (interventions)

### D1 — Measurement-only (Phase 1 POC, **no training**)

Use stock Qwen2.5-7B + existing vectors/hooks.

1. **Tamper–capability curve:** RFA @ layers \(\{10,14,18,22,27\}\), coeff \(\in \{0, r^*, 2r^*\}\); plot ASR vs PPL/MMLU-subset.
2. **Entanglement index \(\kappa\):** per layer refusal direction.
3. **Multi-direction tamper:** RFA \(d_1\) then \(d_2\) (subspace); does \(\Delta C\) increase?
4. **Persona bypass control:** C1 vs C1_evil_system at fixed \(\Delta C\).

**Expected:** Flat \(\Delta C\) for single-layer RFA; persona path high ASR, flat \(\Delta C\).

### D2 — Extended-Refusal micro-FT (Phase 2, small GPU)

- Build ~500–2k extended-refusal pairs (harmful prompt → overview + refuse + ethics), Qwen-7B LoRA 1–2 epochs.
- Re-run abliteration/RFA + HarmBench + PPL.
- **Compare Type-A claim** (ASR ↓ under ablate) vs **Type-B** (\(\Delta C\) under ablate).

*Not entanglement per se — establishes baseline defense class.*

### D3 — Training tracks (Phase 3, research core)

**Full design (plain language + 6 parallel tracks):** `notes/D3_entanglement_training_design.md`

| Track | Type | One line |
|-------|------|----------|
| D2-ER | A | Extended-refusal LoRA — spread safety, keep capability under tamper |
| D3a-ENT | B | `loss = LM_clean − λ·LM_under_RFA` — tamper should hurt benign LM |
| D3b-ENT+ | B | D3a + harmful refusal SFT |
| D3c-FUSE | B | `h' = U(s(h))` no skip — architectural tripwire |
| D3d-HYBRID | A+B | ER data + entangle loss |
| D3e-MULTI | B | RFA penalty @ L18+L22 |

**Risk:** Autoimmune on xstest; EVIL_SYSTEM may still bypass all weight-local tracks.

#### D3a-ENT — how it works (implementation, 2026-06-03)

**One sentence:** During LoRA training, simulate attacker **RFA on refusal @ L18** (same as eval **C1 / T-RFA1**), and optimize weights so that **benign next-token loss is high under RFA** while **low without RFA**.

**RFA (training + eval tamper):** At layer 18, replace the refusal-axis projection with a fixed harmless-mean scalar (from `qwen7b_vectors.pt` metadata). Implemented in `src/hooks.py` → `rfa_direction`; hooked on **all token positions** during forward.

**Data** (`scripts/build_d3_datasets.py` → mixed in `train_lora_track.py`):

| Slice | ~rows | Role |
|-------|-------|------|
| XSTest safe + stub assistant reply | 250 | Keep **no-tamper** LM utility |
| HarmBench harmful → short refusal | 140 | Keep **no-tamper** refusal (not ER long template) |
| Mix ratio | ~75% benign / 25% harmful | `harmful_fraction_d3a: 0.25` |

**Loss** (`scripts/train_lora_track.py`, `lambda_ent: 0.35`):

\[
\mathcal{L} = \mathcal{L}_{\text{LM}}^{\text{clean}} - \lambda_{\text{ent}} \cdot \mathcal{L}_{\text{LM}}^{\text{RFA@18}}
\]

- Minimizing this **raises** CE under RFA (entangle) and **lowers** CE clean.
- Training logs often show **large negative** total loss because \(\mathcal{L}_{\text{RFA}}\) is subtracted — that is expected, not a broken CE.

**What D3a does *not* do:** It does not train on EVIL_SYSTEM, does not maximize harmful comply, and does not add a mandatory fuse branch (that is **D3c**).

**Contrast D2-ER:** D2 uses Extended-Refusal **long** completions (Type-A dispersion). D3a uses **entanglement loss** on mostly benign/stub data (Type-B).

#### Post-train eval protocol (`scripts/eval_d3_checkpoint.py`)

All adapters: **30 XSTest benign NLL** + **HarmBench dev n=20** with optional LLM judge.

| Metric | Meaning |
|--------|---------|
| **mean_nll_clean** | Benign CE, **no** RFA hook — baseline fluency |
| **mean_nll_rfa** | Same benign prompts, **RFA×1 @ L18** |
| **ΔNLL RFA×1** | `mean_nll_rfa − mean_nll_clean` — **capability under tamper** (target ≥ +0.5 for Type-B) |
| **clean_comply / clean_refusal** | Harmful gen, no tamper — still safe? |
| **rfa_comply / rfa_refusal** | Harmful gen under RFA — jailbreak ASR proxy |

**D3c-only extras:** `mean_nll_fuse`, `mean_nll_fuse_zero`, `delta_nll_fuse_zero` (mandatory fuse active vs zeroed branch).

#### D3 LoRA results — dev eval (2026-06-03)

Cloud: 2× `g5.xlarge`; artifacts `outputs/cloud_pull/si-20260602-194000/` (D2, D3c), `si-20260602-182626/` (D3a).  
Eval manifest: `harmbench_manifest_dev.jsonl` (n=20).

| Track | Type | ΔNLL RFA×1 | clean refuse | RFA comply | Interpretation |
|-------|------|------------|--------------|------------|----------------|
| **Stock** (POC-A) | — | ~+0.15 @ L18×1 | — | — | Tamper barely hurts benign LM |
| **D2-ER** | A | **−0.44** | 100% | 0% | Dispersion control: tamper does not hurt capability; safety under RFA still full |
| **D3a-ENT** | B | **+99.3** | 100% | **5%** | **Type-B POC:** RFA destroys benign LM; harmful jailbreak only slightly up |
| **D3c-FUSE** (v1) | B arch | NaN | 85% refuse | 10% | Training diverged → all-NaN weights |
| **D3c-FUSE** (v2 retrain) | B arch | NaN | 80% refuse | **45%** | Unstable: pure `up(down(h))` init + `loss − λ·loss_zero` blew up; fuse not in optimizer first 200 steps of v2 fix |
| **D3c-FUSE** (v3a) | B arch | NaN | 80% refuse | 15% / 20% fz | Train unstable; invalid checkpoint |
| **D3c-FUSE** (v3b) | B arch | **Δfuse_zero +8.59** | 100% refuse | 0% / **30% fz** | **Tripwire on NLL:** fuse-on benign NLL≈4.7; zero-branch≈13.3. Harm gen under `fuse_zero` partly jailbreaks (30% comply) — partial Type-B |
| **D3c-FUSE** (v3c kill) | B arch | **+8.35** (dev +8.35) | 100% | 0% / **55% dev, 44% main fz** | Train-time kill hinge (`fuse_zero_prob=0.12`); **worse** gen under `fuse_zero` than v3b |

#### D3 LoRA results — main eval (n=200, 2026-06-03)

Manifest: `harmbench_manifest_main.jsonl`. Judge: gpt-4o-mini. Artifacts: `eval_tamper_main.json` per adapter; fuse_zero deep-dive: `outputs/cloud_pull/si-20260602-194000/ablations/d3c_fuse_zero_review_main_summary.json`.

| Track | ΔNLL (main) | clean refuse | RFA comply | fuse_zero comply | Notes |
|-------|-------------|--------------|------------|------------------|-------|
| **D2-ER** | RFA **−0.44** | 100% | 0% | — | Type-A stable on holdout |
| **D3a-ENT** | RFA **+99.3** | 99.5% | **3%** | — | Entanglement holds; RFA refusal 72% (vs 55% dev) |
| **D3c v3b** | **Δfz +8.59** | 100% | 0% | **28.5%** | NLL tripwire stable; gen leak under layer kill |
| **D3c v3c** | **Δfz +8.35** | 100% | 0% | **44%** | Kill-hinge train did not tighten gen tripwire |
| **D3c v3d** | **Δfz +6.62** | 99.5% | 0% | **0%** | Harmful refuse @ fuse_zero train; **gen tripwire fixed** vs v3b |

**Tier experiments (main n=200 + dev, 2026-06-04):** `outputs/cloud_pull/si-20260603-175305-d3c/ablations/tier_experiments/`

| Test | Result |
|------|--------|
| EVIL_SYSTEM comply, stock | **91.5%** |
| EVIL_SYSTEM comply, **d3a_ent** | **1%** (persona does not bypass weight entanglement) |
| EVIL_SYSTEM comply, d3c fuse-on | **0%** |
| EVIL + fuse_zero comply, d3c v3b / v3d | **18% / 4%** |
| Real-cap gen NLL (XSTest n=40), stock / d3a+RFA | **0.71 / 3.88** (d3a clean → collapsed template, NLL≈0.003) |
| RFA restore vs EVIL (dev n=20) | restore **no effect** (~90% comply both) |

**Fuse_zero generation analysis (v3b, main n=200):** Pairwise export `d3c_fuse_zero_review_main.jsonl`. Of 200 prompts, **100% clean refusal**; under `fuse_zero`, judge labels: 40 harmful_comply, 43 refusal, 117 benign_unrelated (model often produces non-refusal but non-actionable text). **Tripwire breach** (clean refused → fz complied): **40/200 = 20%**. By category, highest fz comply: **cybercrime_intrusion 50%** (16/32), illegal 22%, harassment 27%; **copyright 0%**. Interpretation: **NLL incapacitation ≠ generation refusal** under architectural layer kill; harmful categories with procedural detail leak most.

**D3c v3 implementation** (`mandatory_fuse.py`, `train_lora_track.py`, `eval_d3_checkpoint.py`):

- **Active fuse:** \(h' = h + W_{up}(W_{down}(h))\), branch **zero-init** (starts ≈ stock).
- **Tamper eval:** `force_zero` → \(h' = 0\) (layer killed).
- **Train (v3b):** fuse fp32, `fuse_lr=3e-5`, `h'=h+branch` zero-init, fuse-on-only; CE float32.
- **Primary eval metric:** `delta_nll_fuse_zero` (v3b: **+8.59** on dev).
- **Primary metric:** `delta_nll_fuse_zero = mean_nll_fuse_zero − mean_nll_fuse_on`.
- **Judge:** `clean` / `rfa` with fuse-on; optional `fuse_zero` tamper.

**D3a headline (thesis-relevant):** We can **train** coupling at operational RFA×1 strength such that \(\Delta C\) (benign NLL proxy) explodes while **no-tamper** refusal stays at 100% and RFA comply stays low (5%). This is **incapacitation under tamper**, not “easy jailbreak + smart model.”

**Caveats:**

1. Benign NLL uses **stub** assistant targets on XSTest, not real model completions — absolute NLL numbers are not comparable to stock; **deltas** across adapters on same eval are the fair comparison.
2. D3a **ΔNLL +99** may include numerical blow-up (saturated CE), not just “worse fluency” — follow up with generation quality + MMLU subset.
3. D3c v2 increased RFA comply (45%) without fixing clean NLL — may be **degraded** model, not clean Type-B tripwire.
4. **EVIL_SYSTEM + adapter** evaluated (tier): D3a/D3c block persona on harmful comply; D3c still leaks under **fuse_zero + EVIL** (v3d: 4%). RQ3 partially answered.

**Scripts:** `train_lora_track.py --track d2_er|d3a_ent|d3c_fuse`; `eval_d3_checkpoint.py`; cloud `run_d2_er_only.sh`, `run_d3a_eval_only.sh`, `run_d3c_only.sh`.

### D4 — Architectural mandatory branch (Phase 2 POC, code)

```python
# Conceptual: at layer l,
s_l = SafetyEncoder(h_l)   # low-rank
h_{l+1} = MainBlock(h_l + U(s_l))  # U required; zero s_l → not identity
```

If `s_l` zeroed, next layer sees OOD → high loss / garbage tokens.

**POC scope:** Single-layer hook in `hooks.py` + frozen random `U` — tests **structural** tripwire, not learned entanglement.

### D5 — Hydra-aware fuse (Phase 4)

Combine Type-A (redundant refusal features per Beyond I'm Sorry) with Type-B on **minimal hitting set** of features — ablate any one → compensate; ablate **all** in set → \(\Delta C \ll -\epsilon\).

Requires SAE pipeline (heavy).

---

## 8. Small POC plan (2–4 weeks, local + optional cloud)

### POC-A: Tamper–capability sweep ( **start here** )

| Item | Detail |
|------|--------|
| Script | `scripts/poc_tamper_capability_sweep.py` (to implement) |
| Model | Qwen2.5-7B-Instruct, existing `qwen7b_vectors.pt` |
| Tamper | RFA L18, coeffs sweep; optional ablate |
| \(S\) | 20 HarmBench dev + handpick 3 |
| \(C\) | PPL on 1k-token benign slice; optional 100 MMLU |
| Output | `outputs/ablations/tamper_capability_curve.json` |
| Time | ~4–8h GPU |

**Success:** Quantified \(\Delta C\) vs \(\Delta S\) — even null result is publishable as replication.

### POC-B: Entanglement index \(\kappa\)

| Item | Detail |
|------|--------|
| Script | `scripts/poc_entanglement_index.py` |
| Method | Finite-diff on last-token hidden along \(d_{\text{refusal}}\) for benign prompts |
| Output | per-layer heatmap JSON |

**Success:** Identify layers where \(\kappa\) is highest — candidate for D3/D4.

### POC-C: Mandatory branch forward (1 day code)

| Item | Detail |
|------|--------|
| Modify | `src/hooks.py` or wrapper module |
| Test | Zero safety branch → generation should break measurably |
| Limit | T1 attacker removes hook — document as architectural bound |

### POC-D: Lecture-refuse vs ER literature

| Item | Detail |
|------|--------|
| Data | `main_c1_c4_taxonomy.jsonl` (done) |
| Analysis | % lecture_refuse; correlation with prompt category |
| Claim | Stock Qwen already partial dispersion without ER FT |

---

## 9. Falsification criteria

| If we observe… | Then… |
|---------------|--------|
| \(\Delta C \geq -\epsilon\) for all \(\mathcal{T}_{\text{RFA}}\) on stock + ER models | Type-B unlikely on direction-tamper alone; focus Type-A + persona detection |
| ER FT lowers abliteration ASR but \(\Delta C \approx 0\) | Confirms literature; thesis pivots to **persona + hydra** bypass of Type-A |
| D3 entanglement training raises \(\kappa\) but T3 FT restores ASR | Entanglement only raises **attack cost**, not guarantee |
| Mandatory branch POC breaks generation when zeroed | Architecture viable for **T0–T1** only; document T2 defeat |
| Multi-direction simultaneous RFA spikes PPL | **Weak Type-B** possible — pursue hitting-set formalization |

---

## 10. Relation to existing repo artifacts

| Artifact | Role in thesis |
|----------|----------------|
| `main_evil_system_conditions.json` | Upper bound on persona attack without \(\Delta C\) |
| `main_c1_c4_taxonomy_summary.json` | Lecture-refuse prevalence (~32%) |
| `handpick_probe_*.jsonl` | Mechanism controls |
| `refusal_subspace_L18.pt` | Multi-direction tamper |
| `src/adasteer.py` | **Not** entanglement — inference defense baseline |
| Cloud `si-20260602-194000` | Attack-scale evidence |

---

## 11. Contribution framing (paper angles)

### Angle 1 — Empirical (fastest)

*"Safety tamper decouples from capability on Qwen2.5: a systematic \(\Delta C\)–\(\Delta S\) audit under RFA, persona, and subspace attacks."*

### Angle 2 — Negative result / limits

*"Why tripwire neurons cannot secure local OSS LLMs: a threat-model and tamper taxonomy."*

### Angle 3 — Positive (if D3/D4 works)

*"Inducing safety–capability entanglement: when does refusal-axis surgery degrade benign perplexity?"*

### Angle 4 — Bridge to our novel phenomena

*"Lecture-refuse as emergent dispersion; EVIL_SYSTEM as orthogonal bypass of entanglement defenses."*

---

## 12. Phased roadmap

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| **0** | Now | This thesis doc; literature PDFs; entity list |
| **1** | 1 week | POC-A + POC-B scripts + results JSON |
| **2** | 2 weeks | POC-C arch hook; taxonomy deep dive (POC-D) |
| **3** | 3–4 weeks | Optional ER micro-LoRA (D2); compare to stock |
| **4** | 1+ month | D3 entanglement FT or SAE-hydra fuse (if Phase 1 promising) |
| **5** | Paper | Workshop: Phase 1+2+handpick; Full: Phase 3+4 |

**Parallel:** Main attack science (EVIL_SYSTEM) largely done; defense AdaSteer is separate track.

---

## 13. Open questions (research diary)

1. Does **lecture-refuse** increase \(\kappa\) vs hard-refuse prompts on same harmful intent?
2. Is there a **minimal set** of layers where simultaneous RFA maximizes \(\Delta C\)?
3. Can **persona direction** (EVIL_SYSTEM prefill) be detected without knowing the string?
4. Is Type-B **compatible** with open-source release, or only API-hosted attested inference?
5. Does entanglement **help attackers** (harder to align, easier to trigger collapse DoS on benign)?

---

## 14. Immediate next actions (agent/user)

1. ~~POC-A/B/C~~ — done on stock Qwen (see §POC results).
2. ~~D2 + D3a train/eval~~ — done; write-up in §7 above.
3. **D3c:** fix NaN training (grad clip added; v2 still unstable baseline NLL) or simplify fuse loss.
4. **Adapter + EVIL_SYSTEM** main n=200 — does persona bypass D3a entanglement?
5. MMLU / real benign completions — validate ΔNLL +99 is not only CE blow-up.
6. Teardown cloud when idle (`cloud/teardown.sh`).

---

## References (verified IDs)

- Arditi, J., et al. (2024). *Refusal in Language Models Is Mediated by a Single Direction.* arXiv:2406.11717.
- Extended-Refusal / abliteration defense (2025). arXiv:2505.19056.
- Prakash, N., et al. (2026). *Beyond I'm Sorry, I Can't: Dissecting LLM Refusal.* AAAI; arXiv:2509.09708.
- Chen, et al. (2025). *Persona Vectors.* arXiv:2507.21509.
- Zhao, et al. (2025). *AdaSteer.* arXiv:2504.09466. EMNLP.
- Zou, et al. (2023). GCG. arXiv:2307.15043.
- Mazeika, et al. (2024). HarmBench. arXiv:2402.04249.

---

## Appendix A — Glossary

| Term | Meaning |
|------|---------|
| **Type-A dispersion** | Spread safety so single-direction tamper fails to jailbreak |
| **Type-B entanglement** | Tamper safety → capability drops |
| **RFA** | ReFAT-style projection replacement on refusal axis |
| **Lecture-refuse** | Long educational refusal basin (our term) |
| **\(\kappa\)** | Entanglement index — benign loss sensitivity to refusal projection |
| **\(\mathcal{T}_{\text{RFA}}\)** | Tamper class: refusal-axis surgery |
| **clean_comply / rfa_comply** | HarmBench judge: fraction **harmful_comply** without / with RFA |
| **ΔNLL RFA×1** | `mean_nll_rfa − mean_nll_clean` on benign prompts (XSTest in D3 eval) |
| **D3a-ENT** | LoRA + loss `LM_clean − λ·LM_RFA` — Type-B entanglement training track |

## Appendix B — POC script spec (for implementation)

`poc_tamper_capability_sweep.py`:

```text
Args: --config, --layers 10,14,18,22,27, --rfa-coeffs 0,1,2,
      --manifest-dev (20), --benign-ppl-file, --out JSON

For each (layer, coeff):
  stack = RFA only at layer
  S: judge harmful comply on manifest
  C: mean NLL on benign slice (no judge)

Plot-ready rows: {layer, coeff, refusal_rate, comply_rate, ppl, delta_ppl_vs_baseline}
```

---

## POC results (2026-06-02, stock Qwen, no training)

| POC | Artifact | Headline |
|-----|----------|----------|
| **A** | `outputs/ablations/poc_a_tamper_capability.json` | L18 RFA×2: benign NLL +0.67 (PPL ~154); L18 RFA×1 (attack default): +0.15 only |
| **B** | `outputs/ablations/poc_b_entanglement_kappa.json` | κ ≈ −0.01 all layers (weak linear coupling on xstest) |
| **C** | `outputs/ablations/poc_c_mandatory_branch.json` | `h+delta` with delta=0 → identical to baseline; **not** true mandatory fuse |
| **D** | `outputs/ablations/poc_d_lecture_refuse_vs_er.md` | ~32% lecture-refuse; ≠ ER robustness |

**Implication:** Stock Qwen at **attack-strength RFA (×1)** barely hurts capability; **×2** hurts a lot → Type-B may need **trained** coupling at operational tamper strength, not inference-only.

**Follow-up (2026-06-04):** v3d main: fuse_zero comply **0%** (vs v3b 28.5%). Tier: stock EVIL **91.5%** vs d3a_ent EVIL **1%**; real-cap gen shows D3a incapacitation (degenerate or template collapse). Restore/AdaSteer do not substitute for entanglement. EC2 `si-20260603-175305-d3c` terminated after pull.

---

*End of thesis doc — update inline when new training/eval runs complete.*
