#!/usr/bin/env python3
"""Defense baseline: add refusal direction back (steer+) under attack."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
NLA_SRC = ROOT.parent / "nla_rsa_study" / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(NLA_SRC))

from common import build_chat_inputs, get_device, load_causal_lm, model_device, release_gpu, resolve_torch_dtype  # noqa: E402

from src.chat_extract import EVIL_SYSTEM  # noqa: E402
from src.config_loader import load_config, repo_root  # noqa: E402
from src.eval_metrics import score_responses  # noqa: E402
from src.generate_eval import build_stack, load_manifest  # noqa: E402
from src.hooks import EditSpec, InterventionStack, register_intervention_hooks, remove_hooks  # noqa: E402
from src.vectors import load_vectors  # noqa: E402

logger = logging.getLogger(__name__)


def add_restore(stack: InterventionStack, layer: int, alpha: float) -> InterventionStack:
    edits = list(stack.edits)
    edits.append(EditSpec(type="steer+", axis="refusal", layer=layer, alpha=alpha))
    return InterventionStack(edits=edits, vectors=stack.vectors)


@torch.no_grad()
def generate_one(model, tokenizer, prompt: str, stack, max_new: int, evil_system: bool) -> str:
    dev = model_device(model)
    if evil_system:
        messages = [{"role": "system", "content": EVIL_SYSTEM}, {"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(text, return_tensors="pt").to(dev)
    else:
        ids = build_chat_inputs(tokenizer, prompt)["input_ids"].to(dev)
    handles = register_intervention_hooks(model, stack) if stack.edits else []
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    remove_hooks(handles)
    return tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True).strip()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=repo_root() / "configs/qwen7b_harmbench.yaml")
    p.add_argument("--manifest", type=Path, default=repo_root() / "prompts/harmbench_manifest_dev.jsonl")
    p.add_argument("--restore-alpha", type=float, default=1.0)
    p.add_argument("--restore-layer", type=int, default=18)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    device = get_device("cuda" if torch.cuda.is_available() else "auto")
    dtype = resolve_torch_dtype(cfg["model"]["dtype"], device)
    payload = load_vectors(repo_root() / cfg["paths"]["vectors"])
    manifest = load_manifest(args.manifest)
    prompts = [row.get("prompt") or row["behavior"] for row in manifest]
    judge_model = cfg.get("eval", {}).get("judge_model", "gpt-4o-mini")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    model = load_causal_lm(cfg["model"]["name"], device, dtype)

    cases = [
        ("C1", "C1", False),
        ("C1_restore", "C1", False),
        ("C1_evil_system", "C1", True),
        ("C1_evil_restore", "C1", True),
    ]
    results = {}
    for label, cond, evil in cases:
        stack = build_stack(cfg, cond, payload)
        if "restore" in label:
            stack = add_restore(stack, args.restore_layer, args.restore_alpha)
        responses = [generate_one(model, tokenizer, r.get("prompt") or r["behavior"], stack, 256, evil) for r in tqdm(manifest, desc=label)]
        s = score_responses(prompts, responses, judge_model=judge_model, desc=label)
        results[label] = s
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    release_gpu(model, tokenizer)


if __name__ == "__main__":
    main()
