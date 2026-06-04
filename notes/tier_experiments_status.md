# Tier 1–3 experiment queue

**Artifacts (pulled):** `outputs/cloud_pull/si-20260603-175305-d3c/ablations/tier_experiments/`  
**Cloud job:** `si-20260603-175305-d3c` — tier + retry completed 2026-06-04T04:27Z (exit 0)

| ID | Experiment | Script | Status |
|----|------------|--------|--------|
| T1.1 | v3d vs v3b table | `v3_compare.json` | ✅ |
| T1.2 | EVIL_SYSTEM × adapters | `attack_matrix_*.json` | ✅ |
| T1.3 | Real capability NLL | `cap_*.json` | ✅ |
| T1.4 | Fuse white-box bypass | `bypass_*.json` | ✅ |
| T2.5 | C1 lecture taxonomy | `c1_taxonomy_main.jsonl` (cloud + local) | ✅ |
| T2.6 | Mechanism samples | `mechanism_samples.json` | ✅ |
| T2.7 | Handpick d2 subspace | `probe_evil_handpick` | optional / deferred |
| T2.8 | D3a RFA scale | `d3a_rfa_scale.json` | ✅ |
| T3.10 | RFA restore | `rfa_restore_dev.json` | ✅ |
| T3.11 | AdaSteer handpick | `adasteer_handpick.json` | ✅ |
| T3.12 | D2 evil (matrix) | `attack_matrix_d2_er.json` | ✅ |

**Key numbers (main n=200 harmful comply):** stock EVIL 91.5%; d3a_ent EVIL 1%; d3c v3d evil+fuse_zero 4%.

**Ops:** `cloud/retry_tier_failures.sh`, `cloud/queue_tier_retries.sh`, `cloud/monitor_tier.sh`
