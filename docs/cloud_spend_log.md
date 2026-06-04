# Cloud spend log — safety interventions / research

> **Rule:** Every cloud job gets a row before launch and a final cost after terminate.  
> **Approval:** Confirm with user before any single invoice > $50 (see CLAUDE.md).

| Date | Provider | Instance | Job ID | Purpose | Duration | Est. $ | Actual $ | Status | Notes |
|------|----------|----------|--------|---------|----------|--------|----------|--------|-------|
| 2026-05-30 | local | M4 MPS | p0-dev | P0 vector extract + dev eval C0/C1/C2/C4 (5 prompts) | ~5 min | $0 | $0 | done | `notes/qwen_harmbench_p0_run_log.md` |
| _pending_ | AWS g5.xlarge spot | — | c2-iter | bootstrap 45 + extract + C0/C2 dev | ~1 h | ~$1 | — | blocked | GPU vCPU quota=0; see cloud/README.md |
| 2026-05-31 | AWS setup | — | infra | keypair + SG us-east-1 + cloud scripts | — | $0 | $0 | done | `cloud/config.env` ready |
| 2026-06-01 | AWS g5.xlarge on-demand | i-0f8613e1658e3e800 | si-20260601-034709 | bootstrap 45 + extract + C0/C2; fix pipeline + C4 eval | ~2.0 h | ~$2.0 | ~$2.0 | done | `outputs/ablations/dev_c2_variants.json`; instance terminated |
| 2026-06-01 | AWS g5.xlarge on-demand | i-01265fcba69be3e02 | si-20260601-061548 | Phase A resume + extract; Phase B quality bootstrap (8 traits) + ortho | ~4h | ~$4 | — | done | terminated |
| 2026-06-02–03 | AWS 2× g5.xlarge on-demand | i-0161d97d4a114589a, i-0551a5035c03a99a2 | si-20260602-194000, si-20260602-182626 | D2/D3a/D3c train+dev; D3 main n=200 eval; D3c v3c kill retrain; fuse_zero review export | ~14h × 2 GPU | **~$28** | — | done | terminated 2026-06-03; artifacts `cloud_pull/si-20260602-*` |
| 2026-06-03–04 | AWS g5.xlarge | i-0d985d6e25f66f3f2 | si-20260603-175305-d3c | D3c v3d pipeline + **tier 1–3** (attack matrix×5, cap, restore, retry) | ~10h wall | ~$10 | ~$10 est | done | `outputs/cloud_pull/si-20260603-175305-d3c/`; terminated 2026-06-04 |

## Running total

| Period | Spent | Budget notes |
|--------|-------|--------------|
| 2026-05 | **$0.00** | P0 local only |
| 2026-06 | **~$44** | 2× g5 ~14h + si-20260603 ~10h tier |

## Job template (copy for new rows)

```
Date: YYYY-MM-DD
Provider: AWS g5.xlarge / Lambda L4 / etc.
Purpose: e.g. HarmBench P2 factorial C0–C7 × 200 HumanJailbreaks
Instance: type, spot/on-demand
Started: UTC
Stopped: UTC
Wall time: X h
Rate: $X/h
Actual: $Y (from AWS billing or instance hour × rate)
Artifacts: outputs/runs/...
Manifest: manifests/run_YYYYMMDD.json
```
