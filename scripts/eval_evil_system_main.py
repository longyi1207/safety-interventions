#!/usr/bin/env python3
"""Main HarmBench eval with EVIL_SYSTEM variants (single model load)."""

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

from common import build_chat_inputs, cleanup_mps, get_device, load_causal_lm, model_device, resolve_torch_dtype  # noqa: E402

from src.chat_extract import EVIL_SYSTEM  # noqa: E402
from src.config_loader import load_config, repo_root  # noqa: E402
from src.eval_metrics import score_responses  # noqa: E402
from src.generate_eval import build_stack, load_manifest  # noqa: E402
from src.hooks import InterventionStack, register_intervention_hooks, remove_hooks  # noqa: E402
from src.vectors import load_vectors  # noqa: E402

logger = logging.getLogger(__name__)

VARIANTS = [
    ("C1", "C1", False),
    ("C4", "C4", False),
    ("C1_evil_system", "C1", True),
    ("C4_evil_system", "C4", True),
    ("evil_system_only", "C0", True),
]


@torch.no_grad()
def generate_one(model, tokenizer, prompt: str, stack: InterventionStack, max_new: int, evil_system: bool) -> str:
    dev = model_device(model)
    if evil_system:
        messages = [{"role": "system", "content": EVIL_SYSTEM}, {"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(text, return_tensors="pt").to(dev)
    else:
        ids = build_chat_inputs(tokenizer, prompt)["input_ids"].to(dev)
    handles = register_intervention_hooks(model, stack) if stack.edits else []
    out = model.generate(
        ids,
        max_new_tokens=max_new,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    remove_hooks(handles)
    return tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True).strip()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=repo_root() / "configs/qwen7b_harmbench.cloud.yaml")
    p.add_argument("--manifest", type=Path, default=repo_root() / "prompts/harmbench_manifest_main.jsonl")
    p.add_argument("--max-new", type=int, default=256)
    p.add_argument("--out", type=Path, default=repo_root() / "outputs/ablations/main_evil_system_conditions.json")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["model"].get("device", "auto"))
    dtype = resolve_torch_dtype(cfg["model"]["dtype"], device)
    payload = load_vectors(repo_root() / cfg["paths"]["vectors"])
    manifest = load_manifest(args.manifest)
    judge_model = cfg.get("eval", {}).get("judge_model", "gpt-4o-mini")
    cache_path = repo_root() / cfg.get("eval", {}).get("judge_cache", "outputs/cache/harmbench_judge.jsonl")
    prompts = [row.get("prompt") or row["behavior"] for row in manifest]

    logger.info("Loading model n=%d", len(manifest))
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    model = load_causal_lm(cfg["model"]["name"], device, dtype)

    results = {}
    for label, cond, evil_sys in VARIANTS:
        stack = build_stack(cfg, cond, payload)
        logger.info("Running %s cond=%s evil_system=%s", label, cond, evil_sys)
        responses = [
            generate_one(model, tokenizer, row.get("prompt") or row["behavior"], stack, args.max_new, evil_sys)
            for row in tqdm(manifest, desc=label)
        ]
        s = score_responses(prompts, responses, judge_model=judge_model, cache_path=cache_path, desc=f"judge-{label}")
        s.update({"condition": label, "config_cond": cond, "evil_system": evil_sys, "judge_model": judge_model})
        results[label] = s
        logger.info("%s: refusal=%.2f comply=%.2f", label, s["refusal_rate"], s["harmful_comply_rate"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    logger.info("Wrote %s", args.out)
    cleanup_mps()


if __name__ == "__main__":
    main()
