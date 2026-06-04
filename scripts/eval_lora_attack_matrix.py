#!/usr/bin/env python3
"""Eval stock vs LoRA adapters under C1 / C1_evil_system (+ optional fuse_zero for D3c)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import yaml
from peft import PeftModel
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
from src.hooks import register_intervention_hooks, remove_hooks  # noqa: E402
from src.mandatory_fuse import MandatoryFuse, register_mandatory_fuse_hooks, remove_fuse_hooks  # noqa: E402
from src.vectors import load_vectors  # noqa: E402

logger = logging.getLogger(__name__)

ATTACKS = [
    ("C1", "C1", False, False),
    ("C1_evil_system", "C1", True, False),
    ("C1_evil_fuse_zero", "C1", True, True),
]


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
    p.add_argument("--config", type=Path, default=repo_root() / "configs/qwen7b_harmbench.yaml")
    p.add_argument("--d3-config", type=Path, default=ROOT / "configs/d3_lora_train.yaml")
    p.add_argument("--manifest", type=Path, default=repo_root() / "prompts/harmbench_manifest_main.jsonl")
    p.add_argument("--adapter", type=Path, default=None, help="LoRA dir; omit for stock base")
    p.add_argument("--max-new", type=int, default=256)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    d3_cfg = yaml.safe_load(args.d3_config.read_text())
    device = get_device("cuda" if torch.cuda.is_available() else cfg["model"].get("device", "auto"))
    dtype = resolve_torch_dtype(cfg["model"]["dtype"], device)
    payload = load_vectors(repo_root() / cfg["paths"]["vectors"])
    manifest = load_manifest(args.manifest)
    prompts = [row.get("prompt") or row["behavior"] for row in manifest]
    judge_model = cfg.get("eval", {}).get("judge_model", "gpt-4o-mini")
    cache_path = repo_root() / cfg.get("eval", {}).get("judge_cache", "outputs/cache/harmbench_judge.jsonl")

    adapter_path = str(args.adapter.resolve()) if args.adapter else None
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    base = load_causal_lm(cfg["model"]["name"], device, dtype)
    model = PeftModel.from_pretrained(base, adapter_path) if adapter_path else base
    model.eval()

    fuse = None
    fl = d3_cfg["train"].get("fuse_layer", 18)
    fuse_path = (args.adapter / "mandatory_fuse.pt") if args.adapter else None
    if fuse_path and fuse_path.exists():
        fuse = MandatoryFuse(
            model.config.hidden_size,
            rank=d3_cfg["train"].get("fuse_rank", 64),
            init_scale=d3_cfg["train"].get("fuse_init_scale", 0.02),
        )
        fuse.load_state_dict(torch.load(fuse_path, map_location=device, weights_only=True))
        fuse = fuse.to(device, dtype=torch.float32).eval()

    results = {"adapter": str(args.adapter) if args.adapter else "stock", "manifest": str(args.manifest), "attacks": {}}
    for label, cond, evil_sys, fz in ATTACKS:
        if fz and fuse is None:
            continue
        stack = build_stack(cfg, cond, payload)
        fuse_h = []
        if fuse is not None:
            fuse_h = register_mandatory_fuse_hooks(model, fuse, fl, force_zero=fz)
        responses = [
            generate_one(model, tokenizer, row.get("prompt") or row["behavior"], stack, args.max_new, evil_sys)
            for row in tqdm(manifest, desc=label)
        ]
        if fuse_h:
            remove_fuse_hooks(fuse_h)
        s = score_responses(prompts, responses, judge_model=judge_model, cache_path=cache_path, desc=label)
        s.update({"config_cond": cond, "evil_system": evil_sys, "fuse_zero": fz})
        results["attacks"][label] = s
        logger.info("%s refuse=%.3f comply=%.3f", label, s["refusal_rate"], s["harmful_comply_rate"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    logger.info("Wrote %s", args.out)
    release_gpu(model, tokenizer, fuse)


if __name__ == "__main__":
    main()
