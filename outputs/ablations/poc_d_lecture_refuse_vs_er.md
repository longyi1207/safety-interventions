# POC-D: Lecture-Refuse vs Extended-Refusal (Type-A Dispersion)

**Date:** 2026-06-02 · **Model:** Qwen2.5-7B-Instruct · **n=200** HarmBench main (stratified)  
**Conditions:** C1 = RFA @ L18; C4 = RFA + evil steer @ L27 (α=10)  
**Source:** `main_c1_c4_taxonomy_summary.json`, `main_c1_c4_taxonomy.jsonl`  
**Thesis:** `THESIS_safety_capability_entanglement.md` §6.4, POC-D

---

## Analysis JSON

```json
{
  "poc": "D",
  "title": "lecture_refuse_vs_extended_refusal",
  "n": 200,
  "conditions": {
    "C1": "RFA @ L18",
    "C4": "RFA @ L18 + evil steer @ L27"
  },
  "baseline_C0": {
    "refusal_rate": 0.73,
    "harmful_comply_rate": 0.225,
    "source": "main_v2_conditions.json"
  },
  "lecture_refuse_rates": {
    "C1": { "count": 64, "rate": 0.32 },
    "C4": { "count": 63, "rate": 0.315 },
    "delta_C4_minus_C1": -0.005
  },
  "style_distribution": {
    "C1": { "harmful_comply": 0.66, "lecture_refuse": 0.32, "partial": 0.01, "benign_unrelated": 0.01 },
    "C4": { "harmful_comply": 0.665, "lecture_refuse": 0.315, "partial": 0.01, "benign_unrelated": 0.01 }
  },
  "judge_rates": {
    "C1": { "refusal": 0.315, "harmful_comply": 0.605, "partial": 0.035, "benign_unrelated": 0.045 },
    "C4": { "refusal": 0.285, "harmful_comply": 0.61, "partial": 0.04, "benign_unrelated": 0.065 }
  },
  "verdict_x_style": {
    "C1": {
      "refusal|lecture_refuse": 57,
      "refusal|harmful_comply": 5,
      "harmful_comply|harmful_comply": 119,
      "harmful_comply|lecture_refuse": 2,
      "partial|lecture_refuse": 3
    },
    "C4": {
      "refusal|lecture_refuse": 50,
      "refusal|harmful_comply": 5,
      "harmful_comply|harmful_comply": 118,
      "harmful_comply|lecture_refuse": 4,
      "partial|lecture_refuse": 6
    }
  },
  "conditional_rates": {
    "among_judge_refusal": {
      "C1": { "lecture_refuse": 0.905, "hard_refuse": 0.0, "harmful_comply_style": 0.079 },
      "C4": { "lecture_refuse": 0.877, "hard_refuse": 0.0, "harmful_comply_style": 0.088 }
    },
    "among_judge_harmful_comply": {
      "C1": { "lecture_refuse": 0.017, "harmful_comply_style": 0.983 },
      "C4": { "lecture_refuse": 0.033, "harmful_comply_style": 0.967 }
    }
  },
  "c1_c4_agreement": {
    "style_label_match": 184,
    "style_label_match_rate": 0.92,
    "both_comply_n": 121,
    "both_comply_lecture_refuse": { "C1": 4, "C4": 7 }
  },
  "category_lecture_refuse_rate_C1": {
    "harassment_bullying": 0.95,
    "harmful": 0.53,
    "illegal": 0.34,
    "misinformation_disinformation": 0.32,
    "chemical_biological": 0.19,
    "copyright": 0.09,
    "cybercrime_intrusion": 0.09
  },
  "extended_refusal_paper": {
    "arxiv": "2505.19056",
    "type": "Type-A dispersion",
    "claims": [
      "FT on topic-overview + explicit refusal + ethics spreads safety across dims",
      ">90% refusal after abliteration vs <20% standard",
      "Capability preserved after tamper (ΔC ≈ 0)"
    ]
  },
  "findings": [
    "RFA collapses hard_refuse to ~0; surviving refusals are ~90% lecture-style",
    "Lecture-refuse is ~32% of all post-RFA outputs but only ~30% judge-refusal — weak residual safety",
    "C4 evil steer barely shifts style vs C1 (Δ lecture_refuse −0.5pp)",
    "Lecture basin is category-selective (hate/harassment >> cyber/copyright)",
    "Stock Qwen shows emergent ER-like *form* without ER training; does not achieve ER *robustness*",
    "No evidence of Type-B coupling: both basins produce fluent text"
  ],
  "implications_type_B": {
    "lecture_refuse_is": "weak emergent Type-A, not entanglement",
    "type_B_requires": "shared substrate where tamper degrades C — not observed",
    "er_micro_ft_test": "expect ASR↓ and lecture→hard ratio shift; still expect ΔC≈0",
    "persona_bypass": "EVIL_SYSTEM orthogonal to lecture basin (thesis H1b; 92% comply under C1_evil_system)"
  },
  "next_steps": [
    "POC-A: quantify ΔC for C1/C4 vs C0",
    "POC-B: κ index on lecture-refuse vs comply prompts",
    "D2: ER micro-LoRA — compare post-ablate refusal rate and style mix"
  ]
}
```

---

## Summary

After single-direction RFA (C1), Qwen2.5-7B drops from **73% → ~30% judge-refusal** while **~66% harmful-comply**. Style taxonomy shows **32% lecture-refuse** overall — but this is **not** a third stable safety mode on equal footing with comply. Almost all judge-refusals (**~90%**) are lecture-style; **zero hard_refuse** appears in n=200. The model did not lose the *ability* to refuse in long educational form; it lost the *propensity* to refuse on most HarmBench categories.

**C1 vs C4:** Evil residual steer at L27 moves lecture_refuse by **−0.5pp** (64→63). Style labels agree on **92%** of prompts. Residual evil direction does not systematically flip lecture→comply at main scale; persona-channel attacks (EVIL_SYSTEM, separate eval) are the bypass path.

## Cross-tab insights

| Pattern | C1 | C4 | Interpretation |
|---------|----|----|----------------|
| refusal ∧ lecture_refuse | 57 | 50 | Residual safety basin |
| harmful_comply ∧ harmful_comply | 119 | 118 | Dominant post-RFA mode |
| harmful_comply ∧ lecture_refuse | 2 | 4 | Judge/style disagreement (comply verdict, educational tone) |
| refusal ∧ harmful_comply (style) | 5 | 5 | Judge/style disagreement (refuse verdict, substantive help) |

**Category split (C1 lecture_refuse rate):** harassment_bullying **95%**, harmful **53%**, illegal **34%**, misinformation **32%**, chemical_biological **19%**, copyright & cybercrime **9%**. Lecture-refuse survives RFA where prompts are overtly hateful or norm-violating; procedural harm (synthesis, hacking, copyright) collapses straight to comply. Dispersion is **topic-gated**, not global.

**Both-comply subset (n=121):** 116 C1 / 114 C4 are style=harmful_comply; only 4–7 show lecture_refuse despite judge comply — edge cases, not a basin.

## vs Extended-Refusal (Type-A)

| Dimension | ER (arXiv:2505.19056) | Our C1/C4 (stock + RFA) |
|-----------|------------------------|-------------------------|
| Mechanism | Intentional multi-dim dispersion via FT | Emergent after removing single refusal direction |
| Refusal *style* | Long refusals with overview + ethics | **Same surface form** (lecture_refuse) |
| Post-tamper refusal *rate* | **>90%** after abliteration | **~30%** after RFA |
| Post-tamper ASR | Low | **~66%** comply |
| ΔCapability | ≈ 0 (explicit goal) | Not measured here; outputs fluent in both regimes |
| hard_refuse | Trained away toward extended form | **Absent** post-RFA (0/200) |

**Conclusion:** Stock Qwen + RFA already exhibits **ER-like rhetoric** on surviving refusals — consistent with thesis §6.4 “weak Type-A emerging from RFA.” It does **not** exhibit ER’s **robustness**: single-direction tamper still jailbreaks ~2/3 of prompts. Dispersion of *style* without dispersion of *decision* is insufficient defense.

## Implication for Type-B entanglement research

1. **Lecture-refuse ≠ Type-B.** It is fluent, educational non-compliance — capability intact, safety partially intact on a subset. Thesis target (tamper ⇒ ΔC < −ε) is orthogonal to making refusals longer.

2. **Type-A ceiling on stock Qwen.** Even if ER micro-FT (D2) raises post-ablate refusal to paper levels, literature and thesis H2a predict **ΔC ≈ 0** — confirms dispersion class, not tripwire.

3. **Two failure modes for any defense:**
   - **Direction tamper** (RFA/abliterate): kills hard_refuse, leaves lecture basin on ~30% — bypass on ~66%.
   - **Persona tamper** (EVIL_SYSTEM): orthogonal manifold, ~92% comply without needing to kill lecture basin (thesis §1.1).

4. **Actionable discriminant for Type-B POC:** Measure **κ** and PPL on lecture-refuse vs comply prompts (POC-B). If κ is low on both, entanglement is absent. Type-B intervention (D3/D4) must raise κ on benign loss *and* show PPL spike under RFA — not merely shift style mix.

5. **Category-aware entanglement** may be necessary: global refusal-direction surgery is category-heterogeneous; a single fuse likely cannot enforce Type-B everywhere.

---

*POC-D complete — no GPU. Feeds thesis §6.4, §8 POC-D, falsification row “ER FT lowers ASR, ΔC≈0”.*
