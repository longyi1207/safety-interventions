#!/usr/bin/env python3
"""POC-C: mandatory low-rank safety branch — zero branch → generation should degrade."""

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

from src.config_loader import load_config, repo_root  # noqa: E402
from src.hooks import _get_layers, _hs, _wrap  # noqa: E402
from src.mandatory_branch import MandatoryBranch, register_mandatory_branch_hooks, remove_hooks  # noqa: E402

logger = logging.getLogger(__name__)


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new: int) -> str:
    dev = model_device(model)
    ids = build_chat_inputs(tokenizer, prompt)["input_ids"].to(dev)
    out = model.generate(
        ids,
        max_new_tokens=max_new,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True).strip()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=repo_root() / "configs/qwen7b_harmbench.yaml")
    p.add_argument("--layer", type=int, default=18)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--prompt", default="Explain how photosynthesis works in two sentences.")
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--out", type=Path, default=repo_root() / "outputs/ablations/poc_c_mandatory_branch.json")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["model"].get("device", "auto"))
    dtype = resolve_torch_dtype(cfg["model"]["dtype"], device)
    hidden = 3584  # Qwen2.5-7B

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    model = load_causal_lm(cfg["model"]["name"], device, dtype)
    branch = MandatoryBranch(hidden, rank=args.rank, layer=args.layer, device=device, dtype=dtype)

    results = {}
    for mode in ("normal", "zero_branch"):
        branch.zero_branch = mode == "zero_branch"
        handles = register_mandatory_branch_hooks(model, branch)
        text = generate(model, tokenizer, args.prompt, args.max_new)
        remove_hooks(handles)
        results[mode] = text
        logger.info("[%s] %s", mode, text[:200])

    def heur_bad(s: str) -> bool:
        if len(s.strip()) < 10:
            return True
        weird = sum(1 for c in s if ord(c) > 0xFFFF or (ord(c) < 32 and c not in "\n\t"))
        return weird > 5

    summary = {
        "layer": args.layer,
        "rank": args.rank,
        "normal_len": len(results["normal"]),
        "zero_branch_len": len(results["zero_branch"]),
        "zero_branch_heuristic_bad": heur_bad(results["zero_branch"]),
        "responses": results,
        "interpretation": "POC-C passes if zero_branch output is clearly worse than normal",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info("Wrote %s", args.out)
    cleanup_mps()


if __name__ == "__main__":
    main()
