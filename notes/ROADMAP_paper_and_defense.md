# Roadmap: paper-quality eval + defense (2026-06-02)

**North star:** Not jailbreak for its own sake — understand **how** Qwen safety fails under RFA + persona, then design **defenses** others cannot easily replicate with single-direction steer.

**Master data log:** `experiment_log_handpick_evil_2026-06-02.md`  
**Literature:** `defense_steering_literature_2026-06.md`

---

## Phase status

### Done ✓
- Main n=200 EVIL_SYSTEM: C1,C4,C1_evil_system,C4_evil_system,evil_system_only (`si-20260602-194000`)
- Main n=200: C0,C1,C4,C_all_traits_rfa (default template, LLM judge)
- Conditional evil C1 vs C4 (n=121 paired)
- Handpick mechanism probes (all failures documented)
- Infra: probe scripts, scoped hooks, subspace extract, review browser

### Not done ✗
- **EVIL_SYSTEM + RFA on main n=200** — ✓ done `si-20260602-194000` (C1_evil_system 92% comply)
- Refusal-style taxonomy on full main JSONL
- Cross-model replication
- Defense intervention evaluated
- Pre-registered hypothesis doc / stats plan

---

## Next actions (ordered, parallelizable)

### Track A — Complete attack-side science (1–2 cloud days)

| # | Task | Output | Parallel? |
|---|------|--------|-----------|
| A1 | **Main eval:** C1_evil_system, C4_evil_system, evil_system_only on n=200 | `main_evil_system_conditions.json` | Cloud GPU |
| A2 | **Lecture taxonomy** on main C1 review JSONL (LLM batch) | `main_c1_refusal_styles.jsonl` | API only, parallel w/ A1 |
| A3 | Estimate: what % of main is lecture_refuse? Does EVIL_SYSTEM flip them? | Table in experiment log | After A1+A2 |
| A4 | Teardown idle cloud + update `docs/cloud_spend_log.md` | — | — |

**Script to add:** `cloud/run_main_evil_system_eval.sh` → extend `sweep_evil_v2.py` or `probe` pattern for full manifest.

### Track B — Paper skeleton (local, no GPU)

| # | Task | Output |
|---|------|--------|
| B1 | **Claims table:** hypothesis → test → result → limitation | § in experiment log |
| B2 | **Figure list:** (1) main comply bars (2) handpick 3×conditions (3) mechanism diagram |
| B3 | Methods paragraph: extract (same vs dual vs prompt-token), RFA, judges |
| B4 | Related work from `defense_steering_literature_2026-06.md` |

### Track C — Defense (after A1 clarifies lecture prevalence)

**Research question:** If attackers use RFA + EVIL_SYSTEM (or future prompt-free persona injection), what defense works?

| # | Task | Rationale |
|---|------|-----------|
| C1 | **Literature deep-read** AdaSteer, CAST, Beyond I'm Sorry hydra | See defense lit note |
| C2 | **Threat model doc:** attacker capabilities = {RFA d₁, EVIL_SYSTEM, residual evil steer, d₂…} | What we must block |
| C3 | **Prototype defenses** on handpick 3 then main: | |
| | (i) Multi-direction RFA **restore** (push refusal back) | Counter Arditi-style attack |
| | (ii) AdaSteer: context-dependent **anti**-comply direction | Qwen2.5-specific |
| | (iii) Detect lecture→comply boundary via activation probe | Monitor, don’t just steer |
| | (iv) Extended-refusal training direction (literature) | Long-term |
| C4 | **Metric:** attack success **and** false refusal on xstest | Defense must not over-refuse |

**Key insight from handpick:** Defense may need to guard **system-slot persona** + **multi-direction refusal**, not only d₁.

### Track D — Safety–capability entanglement (fundamental, Phase 0–1)

**Thesis doc:** `THESIS_safety_capability_entanglement.md`  
**Question:** Can safety tamper (RFA / ablation) be designed to **collapse capability**, not just remove guardrails?

| # | Task | Output |
|---|------|--------|
| D0 | Literature: Extended-Refusal (Type-A dispersion) vs tripwire (Type-B) | Notes in thesis §5 |
| D1 | POC-A: tamper–capability sweep (RFA layers × coeff → PPL + ASR) | `tamper_capability_curve.json` |
| D2 | POC-B: entanglement index κ per layer | heatmap JSON |
| D3 | POC-C: mandatory safety branch forward hook | pass/fail + doc |
| D3-train | **Design:** `D3_entanglement_training_design.md` — D2-ER + D3a–e LoRA tracks | scripts TBD |

_EVIL_SYSTEM main done (`si-20260602-194000`). POC A–D done._

---

## Paper bar — what’s missing

| Gap | Severity |
|-----|----------|
| Main n=200 EVIL_SYSTEM conditions | **High** — central claim needs scale |
| Lecture-refuse prevalence + characterization | **High** — novel phenomenon |
| Statistical CI on main splits | Medium |
| Second model (Llama-3-8B?) | Medium — generalization |
| Human eval subset (n≥50) | Medium |
| Defense baseline comparison | **High** for “safety measure” framing |
| Code + manifest + vectors public | Low (repo mostly ready) |

**Minimum viable paper (workshop):** A1 + A2 + handpick log + mechanism diagram + defense prototype on 3 prompts.  
**Full paper:** + cross-model + AdaSteer comparison + human eval.

---

## Suggested immediate command (when ready)

```bash
# After implementing run_main_evil_system_eval.sh
PIPELINE=main_evil_system bash cloud/start_job.sh
```

---

_Last updated: 2026-06-02_
