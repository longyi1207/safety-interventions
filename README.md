# Tamper-Induced Capability Collapse

**Training safety–capability entanglement in open-weight LLMs** — RFA jailbreak replication, Type-A vs Type-B defenses, HarmBench eval on Qwen2.5-7B (+ Llama/Mistral replicates).

**Author:** Long Yi · **Repo:** https://github.com/longyi1207/safety-interventions

| Artifact | Location |
|----------|----------|
| **Paper PDF** | [`paper/main.pdf`](paper/main.pdf) |
| **LaTeX sources** | [`paper/`](paper/) |
| **Frozen MVA results** | [`outputs/arxiv_mva/`](outputs/arxiv_mva/) — start with [`paper_table.md`](outputs/arxiv_mva/paper_table.md) |
| **Thesis / design notes** | [`notes/THESIS_safety_capability_entanglement.md`](notes/THESIS_safety_capability_entanglement.md) |
| **Blog draft** | [`notes/blog_jailbreak_to_entanglement.md`](notes/blog_jailbreak_to_entanglement.md) |

> **arXiv:** not submitted yet. PDF + sources are in-repo for review and citation.

---

## Headline results (HarmBench main n=200, Qwen2.5-7B)

| Adapter | Mechanism | ΔNLL under RFA×1 | RFA harmful comply | C1 + EVIL_SYSTEM comply |
|---------|-----------|------------------|--------------------|-------------------------|
| **stock** | baseline | +0.17 | 51.5% | **91.5%** |
| **d2_er** | Type-A (Extended-Refusal style) | −0.44 | 0% | 8.0% |
| **d3a_ent** | Type-B (entanglement LoRA) | **+99.3** | 3% | **1%** |
| **d3c_v3d** | mandatory fuse tripwire | −0.24 | 0% | 0% |

**Takeaway:** RFA + persona jailbreak on stock Qwen is cheap (ΔNLL ≈ 0, high comply). **D3a entanglement** inverts the tradeoff — tamper collapses benign NLL instead of unlocking harm. Cross-model replicates (Llama-3.1-8B, Mistral-7B) are weaker; see [`outputs/arxiv_mva/DELIVERABLES.md`](outputs/arxiv_mva/DELIVERABLES.md).

Full tables + MMLU + RFA scale sweep: [`outputs/arxiv_mva/paper_table.md`](outputs/arxiv_mva/paper_table.md).

---

## Repo map

```
safety-interventions/
├── paper/                  # arXiv preprint (main.tex, figures, main.pdf)
├── outputs/
│   ├── arxiv_mva/          # Frozen JSON for paper tables (Jun 2026 MVA)
│   ├── ablations/          # Dev + tier experiment summaries
│   └── cloud_pull/         # Pulled EC2 artifacts (JSON only; no weights)
├── configs/                # Model + training + eval YAML
├── src/                    # Core: hooks, vector extract, generation eval
├── scripts/                # Training, eval, analysis, paper table builder
├── cloud/                  # AWS EC2 launch/sync/run scripts
├── notes/                  # Thesis, experiment logs, design docs
└── docs/                   # AWS batch guide, spend log
```

**Not in git** (by design): LoRA weights (`.safetensors`), vectors (`.pt`), secrets, raw generation logs. Reproduce via scripts + cloud pull.

---

## Quick start (local)

```bash
git clone https://github.com/longyi1207/safety-interventions.git
cd safety-interventions
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=...   # HarmBench LLM judge

# Extract refusal + evil vectors on Qwen2.5-7B (needs GPU or patience on CPU)
python -m src.extract_vectors --config configs/qwen7b_harmbench.yaml --axes refusal evil

# Sanity eval: 5 harmful prompts, refusal ablation
python -m src.generate_eval --config configs/qwen7b_harmbench.yaml \
  --manifest prompts/harmbench_manifest_dev.jsonl \
  --condition C1 --max-rows 5
```

Shared LM helpers: **`vendor/common.py`** (vendored; no sibling repo required).

---

## Reproduce paper numbers

1. **Read frozen artifacts** — everything in `outputs/arxiv_mva/*.json` matches the paper tables.
2. **Regenerate table markdown:**
   ```bash
   python3 scripts/build_arxiv_paper_table.py
   ```
3. **Regenerate figures:**
   ```bash
   cd paper && python3 generate_figures.py && make
   ```
4. **Full re-run** — needs AWS GPU + trained adapters. See [`cloud/README.md`](cloud/README.md) and [`docs/AWS_harmbench_batch.md`](docs/AWS_harmbench_batch.md).

---

## Intervention tracks

| Track | Config | What it does |
|-------|--------|--------------|
| **D2 ER** | `configs/d3_lora_train.yaml` | Type-A: train model to refuse in Extended-Refusal prose style |
| **D3a entangle** | same + entanglement loss | Type-B: penalize low NLL under simulated RFA during LoRA training |
| **D3c fuse** | `configs/d3_lora_train_v3d.yaml` | Architectural mandatory branch; v3d fixes gen–NLL decoupling vs v3b |

Training: `scripts/train_lora_track.py` · Eval: `scripts/eval_lora_attack_matrix.py`, `scripts/eval_d3a_rfa_scale.py`

Cross-model configs: `configs/llama31_8b_*.yaml`, `configs/mistral7b_*.yaml`

---

## Cloud batch

```bash
cd cloud
cp config.env.example config.env   # AWS key, security group, AMI
./launch.sh && ./remote_setup.sh && ./start_job.sh
./status.sh                        # progress
./pull_results.sh                  # artifacts → outputs/cloud_pull/
./teardown.sh
```

Arxiv MVA parallel runs: `cloud/launch_arxiv_parallel.sh` → tracks A–D in `cloud/run_arxiv_track_*.sh`.

---

## Notes index

| Doc | Purpose |
|-----|---------|
| [`notes/THESIS_safety_capability_entanglement.md`](notes/THESIS_safety_capability_entanglement.md) | Full research thesis + metric glossary |
| [`notes/D3_entanglement_training_design.md`](notes/D3_entanglement_training_design.md) | D3a loss + data design |
| [`notes/D3c_fuse_pipeline.md`](notes/D3c_fuse_pipeline.md) | Fuse architecture + v3b→v3d lesson |
| [`notes/entanglement_handpick_examples.md`](notes/entanglement_handpick_examples.md) | Mechanism examples for jailbreak writeup |
| [`notes/ROADMAP_paper_and_defense.md`](notes/ROADMAP_paper_and_defense.md) | Historical roadmap (pre-MVA) |

---

## Monorepo submodule

This repo is also a **git submodule** in the [`ai_notes`](https://github.com/longyi1207/ai_notes) monorepo:

```bash
git submodule update --init code/safety_interventions
```

For public use, clone this standalone repo directly.

---

## Citation (draft)

```bibtex
@misc{yi2026tamper,
  title={Tamper-Induced Capability Collapse: Training Safety--Capability Entanglement in Open-Weight LLMs},
  author={Yi, Long},
  year={2026},
  howpublished={\url{https://github.com/longyi1207/safety-interventions}},
  note={Preprint. PDF in repository.}
}
```
