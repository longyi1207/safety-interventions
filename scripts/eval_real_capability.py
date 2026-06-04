#!/usr/bin/env python3
"""Capability proxy: model-generated continuations on benign prompts (not stub targets)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vendor.common import build_chat_inputs, get_device, load_causal_lm, model_device, release_gpu, resolve_torch_dtype  # noqa: E402

from scripts.poc_tamper_capability_sweep import load_benign_prompts, rfa_stack  # noqa: E402
from src.hooks import register_intervention_hooks, remove_hooks  # noqa: E402
from src.vectors import load_vectors  # noqa: E402

logger = logging.getLogger(__name__)


@torch.no_grad()
def gen_continuation(model, tokenizer, prompt: str, max_new: int) -> str:
    dev = model_device(model)
    ids = build_chat_inputs(tokenizer, prompt)["input_ids"].to(dev)
    out = model.generate(
        ids,
        max_new_tokens=max_new,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True).strip()


@torch.no_grad()
def nll_on_text(model, tokenizer, prompt: str, completion: str) -> float:
    dev = model_device(model)
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion},
    ]
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    user_only = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    ids = tokenizer(full, return_tensors="pt", truncation=True, max_length=1024)["input_ids"].to(dev)
    plen = len(tokenizer(user_only, add_special_tokens=False)["input_ids"])
    if ids.shape[1] < plen + 2:
        return float("nan")
    attn = torch.ones_like(ids)
    logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1, :].float()
    tgt = ids[:, 1:]
    labels = tgt.clone()
    labels[:, : min(plen, labels.shape[1] - 1)] = -100
    return float(
        F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        ).item()
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", type=Path, default=None)
    p.add_argument("--config", type=Path, default=ROOT / "configs/d3_lora_train.yaml")
    p.add_argument("--benign-n", type=int, default=40)
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--rfa", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    device = get_device("cuda" if torch.cuda.is_available() else "auto")
    dtype = resolve_torch_dtype(cfg["model"]["dtype"], device)
    benign = load_benign_prompts(ROOT / "prompts/data/xstest_prompts.csv", args.benign_n)
    payload = load_vectors(ROOT / cfg["paths"]["vectors"])
    stack = rfa_stack(payload, cfg["train"]["rfa_layer"], cfg["train"]["rfa_scale"]) if args.rfa else None

    adapter_path = str(args.adapter.resolve()) if args.adapter else None
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    base = load_causal_lm(cfg["model"]["name"], device, dtype)
    model = PeftModel.from_pretrained(base, adapter_path) if adapter_path else base
    model.eval()

    nlls = []
    samples = []
    for prompt in tqdm(benign, desc="real-cap"):
        handles = register_intervention_hooks(model, stack) if stack else []
        comp = gen_continuation(model, tokenizer, prompt, args.max_new)
        nll = nll_on_text(model, tokenizer, prompt, comp)
        remove_hooks(handles)
        if nll == nll:
            nlls.append(nll)
            samples.append({"prompt": prompt[:200], "completion": comp[:400], "nll": nll})

    result = {
        "adapter": str(args.adapter) if args.adapter else "stock",
        "rfa": args.rfa,
        "mean_nll_on_generated": sum(nlls) / len(nlls) if nlls else None,
        "n": len(nlls),
        "samples": samples[:5],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    logger.info("mean_nll=%s", result["mean_nll_on_generated"])
    release_gpu(model, tokenizer)


if __name__ == "__main__":
    main()
