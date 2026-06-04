# D2 + D3a — Efficient Runbook

## Do you need more AWS GPUs?

**No, for the first pass.** One `g5.xlarge` (1× A10G, 4 vCPU) running **sequentially**:

```
build data (~1 min) → D2 train (~3–5h) → D3a train (~3–5h) → eval (~30 min + API judge)
```

Total **~7–11h wall**, **one spot instance** ≈ $3–8.

### 2× g5 parallel (recommended for speed)

| Machine | Pipeline | Wall ~ |
|---------|----------|--------|
| **A** | `d2_er_only` | ~4–5h |
| **B** | `d3a_d3c` (D3a then D3c **on same GPU**) | ~8–10h |

**Calendar time ≈ max(5h, 10h) ≈ 10h** (not 3-way parallel — 10 vCPU cap = max 2 instances).

```bash
bash cloud/launch_parallel_d23.sh      # spawns 2 instances, writes parallel_d23.env
# wait ~3 min
bash cloud/bootstrap_parallel_d23.sh   # rsync + start both jobs
# monitor: ssh to each or tail logs
bash cloud/pull_parallel_d23.sh
# teardown both instances when done
```

**D3c on machine B** runs after D3a finishes (no third GPU needed).

### 3-way parallel (vCPU ≥16)

| Machine | Pipeline | ~wall |
|---------|----------|-------|
| A (ex-D2) | `run_d3c_only.sh` | ~5h train + eval |
| B | `run_d3a_eval_only.sh` (train done) | ~30–60m judge |
| optional C | `run_c2_main_eval.sh` (stock, main n=200) | ~2h |

```bash
bash cloud/launch_d3c.sh && bash cloud/bootstrap_d3c.sh   # when quota allows
# or repurpose idle instance: PIPELINE=d3c_only JOB_ENV=... bash cloud/start_job.sh
```

## Commands (Mac)

```bash
cd code/safety_interventions
bash cloud/launch.sh                    # if no instance
bash cloud/sync_code.sh
bash cloud/remote_setup.sh              # adds peft
PIPELINE=d2_d3a_train bash cloud/start_job.sh
bash cloud/status.sh
bash cloud/pull_results.sh              # adapters + eval JSON
bash cloud/teardown.sh
```

## Local smoke (M4, optional before cloud)

```bash
python scripts/build_d3_datasets.py
python scripts/train_lora_track.py --track d2_er --max-steps 20
python scripts/train_lora_track.py --track d3a_ent --max-steps 20
```

## Efficiency knobs (already in `configs/d3_lora_train.yaml`)

| Knob | Default | Turn down for speed |
|------|---------|-------------------|
| `max_steps` | 600 | 300 (half time, may underfit) |
| `max_harmful_er` | 400 | 200 |
| `grad_accum` | 8 | 4 (more VRAM) |
| `eval --judge` | on in cloud script | drop `--judge` to save API $ |

## Outputs to pull

- `outputs/adapters/d2_er/`
- `outputs/adapters/d3a_ent/`
- `outputs/adapters/*/eval_tamper.json`
- `outputs/data/d3/*.jsonl`

## Success glance

| Track | Type-A signal | Type-B signal |
|-------|---------------|---------------|
| D2 | RFA: **low** comply ↑, ΔNLL **small** | — |
| D3a | — | RFA: ΔNLL **+99.3** (dev), clean 100% refuse, RFA 5% comply — **Type-B ✓** |

**Pulled 2026-06-03:** `outputs/cloud_pull/si-20260602-194000/adapters/{d2_er,d3c_fuse}`, `si-20260602-182626/adapters/d3a_ent/`. See `THESIS_safety_capability_entanglement.md` §7 for metric glossary.

---

_updated 2026-06-02_
