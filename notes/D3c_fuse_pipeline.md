# D3c mandatory-fuse — end-to-end pipeline

**Goal:** One command: data → train → dev/main eval → fuse_zero export → analysis → HTML review.

## Quick start (local)

```bash
cd code/safety_interventions
# full run (~hours on GPU; needs OPENAI for judge)
bash scripts/run_d3c_fuse_pipeline.sh

# smoke (40 steps, dev eval only, no judge)
SMOKE=1 SKIP_MAIN=1 SKIP_JUDGE=1 bash scripts/run_d3c_fuse_pipeline.sh
```

## Quick start (cloud)

```bash
bash cloud/launch_d3c.sh          # new g5.xlarge
bash cloud/bootstrap_d3c.sh       # sync + start pipeline (v3d default)
# or existing instance:
JOB_ENV=cloud/.active/si-XXXX.env PIPELINE=d3c_pipeline bash cloud/start_job.sh
bash cloud/pull_d3c.sh            # or pull_d3_main.sh
```

## Tracks

| Track | Config | Training |
|-------|--------|----------|
| `d3c_fuse` (v3b) | `d3_lora_train.yaml` | Fuse-on only; NLL tripwire ✓, gen fz leak ~28% main |
| `d3c_fuse_v3c` | `d3_lora_train_v3c_kill.yaml` | Kill hinge — **regressed** gen fz |
| **`d3c_fuse_v3d`** | `d3_lora_train_v3d.yaml` | Benign @ fuse-on; **harmful refuse @ fuse_zero** (gen coupling) |

## Pipeline steps

1. `build_d3_datasets.py` → includes `d3c_fuse_train.jsonl` (`fuse_zero` tags)
2. `train_lora_track.py --track d3c_fuse_v3d`
3. `eval_d3_checkpoint.py` dev + main (`--fuse-eval`)
4. `export_d3_fuse_zero_review.py` + `analyze_d3_fuse_zero_review.py`
5. `build_d3_fuse_review_browser.py` → `*.html`

## Env overrides

| Var | Default |
|-----|---------|
| `TRACK` | `d3c_fuse_v3d` |
| `CFG` | `configs/d3_lora_train_v3d.yaml` |
| `SKIP_TRAIN` | 0 |
| `SKIP_MAIN` | 0 |
| `SKIP_JUDGE` | 0 |
| `SMOKE` | 0 |

## Artifacts

```
outputs/adapters/d3c_fuse_v3d/
  adapter_model.safetensors
  mandatory_fuse.pt
  eval_tamper.json
  eval_tamper_main.json
outputs/ablations/d3c_fuse_zero_review_d3c_fuse_v3d.jsonl
outputs/ablations/d3c_fuse_zero_review_d3c_fuse_v3d_summary.json
outputs/ablations/d3c_fuse_zero_review_d3c_fuse_v3d.html
```
