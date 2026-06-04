# HarmBench batch jobs on AWS GPU

Pattern mirrors `code/nla_rsa_study/docs/AWS_NLA_decode.md`.

## When to use cloud (default for these)

| Job | Instance | Est. |
|-----|----------|------|
| GCG suffix optimization (200 beh) | `g5.xlarge` / L4 | 8–16 h, ~$25–40 |
| Main factorial generation (8 × 200 × ~5 templates) | `g5.xlarge` | 4–8 h, ~$15–25 |
| HarmBench classifier batch | same instance | +2 h |
| Phase B quality factorial | `g5.xlarge` | 4–8 h, ~$15–25 |
| Llama-8B appendix (3 conditions) | `g5.xlarge` | ~6 h, ~$8 |

**Local M4:** vector extraction, dev manifest (5–20 rows), hook debugging only.

## Setup

```bash
# Launch Ubuntu 22.04, 50GB+ disk, NVIDIA driver + CUDA
git clone <your-repo> ai_notes && cd ai_notes/code/safety_interventions
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Clone HarmBench alongside
git clone https://github.com/centerforaisafety/HarmBench.git ../external/HarmBench
```

## Upload artifacts from Mac

```bash
scp outputs/vectors/qwen7b_vectors.pt ubuntu@$HOST:~/ai_notes/code/safety_interventions/outputs/vectors/
scp prompts/harmbench_manifest_main.jsonl ubuntu@$HOST:~/ai_notes/code/safety_interventions/prompts/
scp configs/qwen7b_harmbench.yaml ubuntu@$HOST:~/ai_notes/code/safety_interventions/configs/
```

## Run factorial (example)

```bash
cd ~/ai_notes/code/safety_interventions
for C in C0 C1 C2 C3 C4 C5 C6 C7; do
  python -m src.generate_eval \
    --config configs/qwen7b_harmbench.yaml \
    --manifest prompts/harmbench_manifest_main.jsonl \
    --condition $C \
    --out outputs/runs/main_${C}.jsonl
done
```

## GCG

Use HarmBench upstream scripts to precompute adversarial suffixes **on the same instance** before generation:

```bash
cd ../external/HarmBench
# Follow HarmBench README for GCG on Qwen2.5-7B-Instruct
# Export merged prompts → ../safety_interventions/prompts/harmbench_manifest_gcg.jsonl
```

## Download results

```bash
scp -r ubuntu@$HOST:~/ai_notes/code/safety_interventions/outputs/runs ./outputs/
```

## Cost control

- Spot/preemptible instances OK for generation (manifest is resumable)
- Pin HF model revision in manifest JSON
- Terminate instance after scp
- Confirm with user before spend > $50 total per CLAUDE.md
