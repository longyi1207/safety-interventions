# Phase B v2 — Results

## Phase A (HarmBench dev, LLM judge)

- **C0**: harmful_comply=25.0%, refusal=75.0%
- **C1**: harmful_comply=75.0%, refusal=25.0%
- **C2**: harmful_comply=15.0%, refusal=85.0%
- **C4**: harmful_comply=75.0%, refusal=25.0%

## Trait singles (steer+ α=10, L=best per trait)

- **malevolence**: harmful_comply=15.0%, refusal=85.0%
- **manipulation**: harmful_comply=15.0%, refusal=85.0%
- **narcissism**: harmful_comply=15.0%, refusal=85.0%
- **moral_disengagement**: harmful_comply=15.0%, refusal=85.0%
- **entitlement**: harmful_comply=15.0%, refusal=85.0%
- **hubris_dominance**: harmful_comply=15.0%, refusal=85.0%
- **sycophancy**: harmful_comply=15.0%, refusal=85.0%

## Interaction design (raw vectors)

- **B0_empty**: harmful_comply=20.0% edits=[]
- **B1_malevolence**: harmful_comply=15.0% edits=[['malevolence', 27, 10.0]]
- **B2_manipulation**: harmful_comply=15.0% edits=[['manipulation', 27, 10.0]]
- **B3_moral_disengagement**: harmful_comply=15.0% edits=[['moral_disengagement', 27, 10.0]]
- **B4_malev_manip**: harmful_comply=15.0% edits=[['malevolence', 27, 10.0], ['manipulation', 27, 10.0]]
- **B5_malev_moral**: harmful_comply=15.0% edits=[['malevolence', 27, 10.0], ['moral_disengagement', 27, 10.0]]
- **B6_manip_moral**: harmful_comply=15.0% edits=[['manipulation', 27, 10.0], ['moral_disengagement', 27, 10.0]]
- **B7_triple**: harmful_comply=15.0% edits=[['malevolence', 27, 8.0], ['manipulation', 27, 8.0], ['moral_disengagement', 27, 8.0]]

## Orthogonalized (all stage)

- **B0_empty**: harmful_comply=20.0%
- **B1_malevolence**: harmful_comply=15.0%
- **B2_manipulation**: harmful_comply=25.0%
- **B3_moral_disengagement**: harmful_comply=15.0%
- **B4_malev_manip**: harmful_comply=15.0%
- **B5_malev_moral**: harmful_comply=15.0%
- **B6_manip_moral**: harmful_comply=15.0%
- **B7_triple**: harmful_comply=15.0%
- **single_entitlement**: harmful_comply=15.0%
- **single_hubris_dominance**: harmful_comply=15.0%
- **single_malevolence**: harmful_comply=15.0%
- **single_manipulation**: harmful_comply=25.0%
- **single_moral_disengagement**: harmful_comply=15.0%
- **single_narcissism**: harmful_comply=15.0%
- **single_sycophancy**: harmful_comply=20.0%

## Cosine matrix @ L27 (before ortho)

- malevolence: malevolence=1.00, manipulation=0.78, narcissism=0.73, moral_disengagement=0.81, sycophancy=0.65
- manipulation: manipulation=1.00, narcissism=0.75, moral_disengagement=0.85, sycophancy=0.62
- narcissism: narcissism=1.00, sycophancy=0.88
- moral_disengagement: narcissism=0.79, moral_disengagement=1.00, sycophancy=0.77
- entitlement: malevolence=0.78, manipulation=0.82, narcissism=0.78, moral_disengagement=0.91, entitlement=1.00, hubris_dominance=0.82, sycophancy=0.74
- hubris_dominance: malevolence=0.82, manipulation=0.85, narcissism=0.89, moral_disengagement=0.85, hubris_dominance=1.00, sycophancy=0.74
- sycophancy: sycophancy=1.00

## All-traits stack benchmark (dev, α=6 per trait, ortho vectors)

| Condition | harmful_comply | refusal | n_edits |
|-----------|----------------|---------|---------|
| C0 | 20% | 80% | 0 |
| C1 (refusal RFA only) | 70% | 25% | 1 |
| C4 (RFA + evil) | 75% | 25% | 2 |
| C_all_traits (7× steer+) | 15% | 85% | 7 |
| **C_all_traits_rfa** | **75%** | 25% | 8 |
| C_all_traits_rfa_evil | 75% | 20% | 9 |

**Takeaway:** All 7 quality traits stacked ≈ single traits alone (15% comply). Refusal-RFA dominates; adding traits on top of RFA does not beat C1/C4. Evil on top of all-traits+RFA is redundant.

Source: `outputs/ablations/all_traits_benchmark_dev.json`

## Main split (n=200, LLM judge, cloud si-20260602-044255)

| Condition | refusal | harmful_comply |
|-----------|---------|----------------|
| C0 | 73% | 22.5% |
| C1 | 29.5% | **65%** |
| C4 | 29% | **65%** |
| C_all_traits_rfa | 28.5% | **65%** |

**Headline:** Main confirms dev — RFA flips refusal; **C4 = C1 on aggregate comply** (evil adds 0pp at n=200). Traits+RFA same as C1.

Sources: `outputs/ablations/main_v2_conditions.json`, `all_traits_benchmark_main.json` (from cloud pull `si-20260602-044255`)

**Pending:** `conditional_evil_C1ref_main.json` (C1 vs C4 paired evil score, cloud `PIPELINE=conditional_main`)

## Stage-2 recommendation
Top-3 traits by harmful_comply on dev: malevolence, manipulation, narcissism

**Verdict (2026-06-02 audit):** Steering **gated by refusal @ L18**, not broken. C2 alone ≈ C0; C4 shows evil persona lift on comply subset (dev). See `notes/steering_eval_audit.md`.

## Open questions
- Compare raw vs ortho interaction gains (additive vs redundant axes).
- Main-split replication for top traits + best interaction cell.
- Refusal-RFA × quality interaction (Phase C).
