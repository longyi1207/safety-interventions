#!/usr/bin/env python3
"""HarmBench dev eval: C0, C2 steer-only, C2+EVIL_SYSTEM, C4 (single model load)."""

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
sys.path.insert(0, str(ROOT))

from vendor.common import build_chat_inputs, cleanup_mps, get_device, load_causal_lm, model_device, resolve_torch_dtype  # noqa: E402

from src.chat_extract import EVIL_SYSTEM
from src.config_loader import load_config, repo_root
from src.eval_metrics import score_responses
from src.generate_eval import build_stack, load_manifest
from src.hooks import register_intervention_hooks, remove_hooks
from src.vectors import load_vectors

logger = logging.getLogger(__name__)

VARIANTS = [
    ("C0", [], False),
    ("C2", ["C2"], False),
    ("C2_evil_system", ["C2"], True),
]


def _has_axis(payload: dict, axis: str) -> bool:
    return axis in payload.get("axes", {}) and bool(payload["axes"][axis])


@torch.no_grad()
def generate_one(model, tokenizer, prompt: str, stack, max_new: int, evil_system: bool) -> str:
    if evil_system:
        messages = [{"role": "system", "content": EVIL_SYSTEM}, {"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(text, return_tensors="pt").to(model_device(model))
    else:
        ids = build_chat_inputs(tokenizer, prompt)["input_ids"].to(model_device(model))
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
    p.add_argument("--manifest", type=Path, default=repo_root() / "prompts/harmbench_manifest_dev.jsonl")
    p.add_argument("--vectors", type=Path, default=None)
    p.add_argument("--max-new", type=int, default=256)
    p.add_argument("--out", type=Path, default=repo_root() / "outputs/ablations/dev_c2_variants.json")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["model"].get("device", "mps"))
    dtype = resolve_torch_dtype(cfg["model"]["dtype"], device)
    vec_path = args.vectors or repo_root() / cfg["paths"]["vectors"]
    payload = load_vectors(vec_path)
    manifest = load_manifest(args.manifest)
    eval_cfg = cfg.get("eval", {})
    judge_model = eval_cfg.get("judge_model", "gpt-4o-mini")
    cache_path = repo_root() / eval_cfg.get("judge_cache", "outputs/cache/harmbench_judge.jsonl")
    user_prompts = [row.get("prompt") or row["behavior"] for row in manifest]

    logger.info("Loading model n=%d vectors=%s", len(manifest), vec_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    model = load_causal_lm(cfg["model"]["name"], device, dtype)

    results = {}
    variants = list(VARIANTS)
    if _has_axis(payload, "refusal"):
        variants.append(("C4", ["C4"], False))

    for label, conds, evil_sys in variants:
        stack = build_stack(cfg, conds[0], payload) if conds else build_stack(cfg, "C0", payload)
        if not conds:
            from src.hooks import InterventionStack

            stack = InterventionStack(edits=[], vectors=payload["axes"])
        logger.info("Running %s evil_system=%s edits=%s", label, evil_sys, stack.edits)
        responses = [
            generate_one(model, tokenizer, row.get("prompt") or row["behavior"], stack, args.max_new, evil_sys)
            for row in tqdm(manifest, desc=label)
        ]
        s = score_responses(
            user_prompts,
            responses,
            judge_model=judge_model,
            cache_path=cache_path,
            desc=f"judge-{label}",
        )
        s.update({"condition": label, "evil_system": evil_sys, "judge_model": judge_model})
        results[label] = s
        logger.info(
            "%s: refusal=%.2f harmful_comply=%.2f",
            label,
            s["refusal_rate"],
            s["harmful_comply_rate"],
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    logger.info("Wrote %s", args.out)
    cleanup_mps()


if __name__ == "__main__":
    main()
