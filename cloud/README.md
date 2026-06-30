# Cloud jobs — isolated EC2, easy teardown, status from Mac

Everything tagged `Project=safety-interventions`. One instance = one job. **Terminate when done.**

## GPU quota (required)

Your account currently has **Running On-Demand G and VT instances = 0** in us-east-1 / us-west-2.

Request increase (usually approved in minutes–hours):

1. [AWS Service Quotas → EC2 → Running On-Demand G and VT instances](https://console.aws.amazon.com/servicequotas/home/services/ec2/quotas/L-DB2E8BA2)
2. Region: **us-east-1**
3. Request **4** vCPUs (enough for one `g4dn.xlarge` or `g5.xlarge`)

Then: `cloud/launch.sh` → `sync_code.sh` → `remote_setup.sh` → `start_job.sh`

Until quota is approved, use **RunPod/Lambda** with the same `run_c2_iter.sh` on a rented GPU, or run bootstrap locally when idle.

## Quick start (from Mac)

```bash
cd code/safety_interventions/cloud
cp config.env.example config.env   # fill AWS_KEY_NAME, SG, AMI, GIT_REPO_URL

chmod +x *.sh
./launch.sh                          # ~$0.5–1/h g5.xlarge spot
./remote_setup.sh                    # clone + pip (once per instance)
./start_job.sh                       # bootstrap → extract → C0/C2 eval (background)

./status.sh                          # anytime: JSON progress + log tail
./pull_results.sh                    # when done: scp artifacts to Mac
./teardown.sh                        # kill instance (or ./teardown.sh JOB_ID)
```

Log spend in `docs/cloud_spend_log.md`.

## arXiv MVA parallel runs (tracks A–D)

Frozen results live in `outputs/arxiv_mva/` (see `DELIVERABLES.md` there).

```bash
./launch_arxiv_parallel.sh      # spin up parallel EC2 jobs
./monitor_arxiv_parallel.sh     # watch progress
./pull_arxiv_parallel.sh        # consolidate JSON → outputs/arxiv_mva/
./run_arxiv_finish.sh           # post-process + paper table
```

Individual tracks: `run_arxiv_track_a.sh` (RFA scale), `track_b` (unified table), `track_c` (MMLU), `track_d_llama.sh` / `track_d_second_model.sh`.

## What runs on cloud

`run_c2_iter.sh` (3 steps, ~45–60 min on GPU):

1. **bootstrap** — Qwen generates 45 evil/neutral pairs → `prompts/evil_contrast_full.jsonl`
2. **extract** — evil vector → `outputs/vectors/qwen7b_vectors.pt`
3. **eval** — HarmBench dev 20, conditions `C0,C2` by default

Status file on instance: `outputs/job_status.json` (phase, pct, message).

## Config

- GPU yaml: `configs/qwen7b_harmbench.cloud.yaml` (`device: cuda`)
- Local yaml unchanged: `configs/qwen7b_harmbench.yaml` (`device: mps`)

## Teardown

```bash
./teardown.sh              # all safety-interventions instances in region
./teardown.sh si-20260531-120000   # one job
```

No S3, no shared VPC resources — only the instance you launched (+ your existing keypair/SG).

## Troubleshooting

| Issue | Fix |
|-------|-----|
| SSH timeout after launch | Wait 2–3 min; DLAMI boots slowly |
| bootstrap keeps skipping | SSH in, re-run with `--min-evil-score 4 --min-contrast 2 --attempts 8` |
| Fan on Mac | You shouldn't run bootstrap locally — use this cloud flow |
