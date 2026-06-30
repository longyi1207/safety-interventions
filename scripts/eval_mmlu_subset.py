#!/usr/bin/env python3
"""5-shot MMLU on a subject subset — clean vs RFA for capability beyond stub NLL."""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vendor.common import build_chat_inputs, get_device, load_causal_lm, model_device, release_gpu, resolve_torch_dtype  # noqa: E402

from scripts.poc_tamper_capability_sweep import rfa_stack  # noqa: E402
from src.hooks import register_intervention_hooks, remove_hooks  # noqa: E402
from src.vectors import load_vectors  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_SUBJECTS = [
    "high_school_mathematics",
    "college_computer_science",
    "logical_fallacies",
    "moral_scenarios",
    "professional_law",
]

CHOICES = ("A", "B", "C", "D")


def load_mmlu_examples(subjects: list[str], n_per_subject: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    rng = random.Random(seed)
    rows = []
    for subj in subjects:
        ds = load_dataset("cais/mmlu", subj, split="test")
        idxs = list(range(len(ds)))
        rng.shuffle(idxs)
        for i in idxs[:n_per_subject]:
            ex = ds[i]
            rows.append(
                {
                    "subject": subj,
                    "question": ex["question"],
                    "choices": ex["choices"],
                    "answer": CHOICES[int(ex["answer"])],
                }
            )
    return rows


def format_few_shot(train_ds, k: int, rng: random.Random) -> str:
    idxs = list(range(len(train_ds)))
    rng.shuffle(idxs)
    parts = []
    for i in idxs[:k]:
        ex = train_ds[i]
        ans = CHOICES[int(ex["answer"])]
        opts = "\n".join(f"{c}. {t}" for c, t in zip(CHOICES, ex["choices"]))
        parts.append(f"{ex['question']}\n{opts}\nAnswer: {ans}")
    return "\n\n".join(parts)


def format_query(example: dict) -> str:
    opts = "\n".join(f"{c}. {t}" for c, t in zip(CHOICES, example["choices"]))
    return f"{example['question']}\n{opts}\nAnswer:"


@torch.no_grad()
def predict_letter(model, tokenizer, prompt: str, max_new: int = 8) -> str:
    dev = model_device(model)
    ids = build_chat_inputs(tokenizer, prompt)["input_ids"].to(dev)
    out = model.generate(
        ids,
        max_new_tokens=max_new,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    text = tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True).strip()
    m = re.search(r"\b([ABCD])\b", text.upper())
    return m.group(1) if m else text[:20]


def eval_split(model, tokenizer, examples: list[dict], few_shot_prefix: str, stack) -> dict:
    correct = 0
    preds = []
    for ex in tqdm(examples, desc="mmlu", leave=False):
        q = format_query(ex)
        prompt = f"{few_shot_prefix}\n\n{q}" if few_shot_prefix else q
        handles = register_intervention_hooks(model, stack) if stack else []
        pred = predict_letter(model, tokenizer, prompt)
        remove_hooks(handles)
        ok = pred == ex["answer"]
        correct += int(ok)
        preds.append({"subject": ex["subject"], "gold": ex["answer"], "pred": pred, "ok": ok})
    n = len(examples)
    return {"accuracy": correct / n if n else 0.0, "n": n, "preds": preds}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", type=Path, default=None)
    p.add_argument("--config", type=Path, default=ROOT / "configs/d3_lora_train.yaml")
    p.add_argument("--subjects", default=",".join(DEFAULT_SUBJECTS))
    p.add_argument("--n-per-subject", type=int, default=20)
    p.add_argument("--shots", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rfa", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    device = get_device("cuda" if torch.cuda.is_available() else "auto")
    dtype = resolve_torch_dtype(cfg["model"]["dtype"], device)
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    examples = load_mmlu_examples(subjects, args.n_per_subject, args.seed)

    from datasets import load_dataset

    rng = random.Random(args.seed)
    few_blocks = []
    for subj in subjects:
        train = load_dataset("cais/mmlu", subj, split="dev")
        few_blocks.append(format_few_shot(train, args.shots, rng))
    few_shot_prefix = "\n\n".join(few_blocks[:2])  # keep prompt shorter; 2 subjects' demos

    payload = load_vectors(ROOT / cfg["paths"]["vectors"])
    stack = (
        rfa_stack(payload, cfg["train"]["rfa_layer"], cfg["train"]["rfa_scale"])
        if args.rfa
        else None
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    base = load_causal_lm(cfg["model"]["name"], device, dtype)
    model = PeftModel.from_pretrained(base, str(args.adapter)) if args.adapter else base
    model.eval()

    result = eval_split(model, tokenizer, examples, few_shot_prefix, stack)
    out = {
        "adapter": str(args.adapter) if args.adapter else "stock",
        "rfa": args.rfa,
        "subjects": subjects,
        "n_per_subject": args.n_per_subject,
        "shots": args.shots,
        **{k: result[k] for k in ("accuracy", "n")},
        "sample_errors": [p for p in result["preds"] if not p["ok"]][:10],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    logger.info("accuracy=%.3f n=%s rfa=%s", out["accuracy"], out["n"], args.rfa)
    release_gpu(model, tokenizer, base)


if __name__ == "__main__":
    main()
