# Defense-oriented steering literature (2026-06-02)

**Context:** Qwen2.5-7B handpick findings — EVIL_SYSTEM + RFA jailbreaks lecture-refusal; residual evil steer does not. Goal: design **defenses** knowing this attack surface.

**Existing repo notes:** `notes/safety_mech_interp_literature.md` (§ Beyond I'm Sorry, CAST, Extended-Refusal), `notes/dual_use_interpretability_manipulation_risks.md` (Arditi refusal direction).

---

## Reading order (defense follow-up)

1. **Arditi et al. 2024** — [2406.11717](https://arxiv.org/abs/2406.11717) — *Refusal in Language Models Is Mediated by a Single Direction* — baseline attack we replicate (RFA); defense must assume this is known.
2. **Beyond I'm Sorry (AAAI 2026)** — [2509.09708](https://arxiv.org/abs/2509.09708) — refusal hydra / redundant SAE features — explains why d₂ RFA failed on handpick; defense implication: **multi-feature** monitoring.
3. **CAST** — conditional activation steering — [see safety_mech_interp_literature.md] — steer refusal only when harmful intent detected; defense template.
4. **Extended-Refusal FT** — [2505.19056](https://arxiv.org/abs/2505.19056) — spreads safety across dimensions; harder to ablate — training-time defense.
5. **AdaSteer** — [2504.09466](https://arxiv.org/abs/2504.09466) — adaptive dual-direction steering, reports Qwen2.5 — **priority implement** as defense baseline. **Deep note:** `notes/adasteer_explainer_2026-06-02.md`.
6. **SafeSteer** — [2506.04250](https://arxiv.org/abs/2506.04250) — category-specific guardrails without blanket refusal.

---

## Papers to add / deep-read (attack–defense duality)

| arXiv | Title (short) | Relevance to our work |
|-------|---------------|------------------------|
| 2504.09466 | AdaSteer | Defense via adaptive steer on Qwen2.5; compare to our C1_evil_system attack |
| 2509.09708 | Beyond I'm Sorry | Multi-direction refusal; our d₂ extract negative result |
| 2505.19056 | Extended-Refusal FT | Lecture-style refusal may be related; defense training |
| 2506.04250 | SafeSteer | Inference guardrails |
| 2406.11717 | Single refusal direction | Our RFA attack baseline |
| Chen persona vectors | (Betley/Chen line) | Persona ≠ refusal; our C2 vs C4 story |

**Verify before citing:** 2602.04896 (Steering Externalities), 2603.24543 (Safety Pitfalls of Steering), 2605.10664 (Prompt-Activation Duality / GCAD) — flagged by literature search; fetch PDFs and confirm claims.

---

## Defense research questions (from handpick)

1. **Can we detect EVIL_SYSTEM or equivalent persona injection** from prefill activations (without known string)?
2. **Can multi-direction refusal restoration** block C1_evil_system while preserving xstest?
3. **Is lecture-refuse a stable basin** we should explicitly train/monitor for (not just hard-refuse)?
4. **Does AdaSteer block RFA+persona** on main n=200 better than single-direction refusal add-back?

---

## Open gaps (publication hooks)

- Controlled **EVIL_SYSTEM vs residual steer** comparison on HarmBench n=200 — not in prior work on Qwen2.5 this way.
- **Lecture-refuse** as third outcome class (not refusal/comply in standard rubric).
- **Negative results:** dual-system extract, prompt-token steer, d₂ RFA — valuable for field.

---

_To extend: add PDF paths under `readings/` when deep-read; update `notes/entity_master_list.md`._
