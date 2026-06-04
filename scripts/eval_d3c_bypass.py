#!/usr/bin/env python3
"""White-box bypass tests for D3c mandatory fuse."""

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
sys.path.insert(0, str(ROOT))

from vendor.common import get_device, load_causal_lm, release_gpu, resolve_torch_dtype  # noqa: E402

from scripts.poc_tamper_capability_sweep import load_benign_prompts, mean_benign_nll  # noqa: E402
from src.mandatory_fuse import MandatoryFuse, register_mandatory_fuse_hooks, remove_fuse_hooks  # noqa: E402

logger = logging.getLogger(__name__)

MODES = [
    ("fuse_on", False),
    ("fuse_zero", True),
    ("no_hook", None),  # LoRA only, fuse weights unused
]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "configs/d3_lora_train.yaml")
    p.add_argument("--benign-n", type=int, default=30)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    device = get_device("cuda" if torch.cuda.is_available() else "auto")
    dtype = resolve_torch_dtype(cfg["model"]["dtype"], device)
    benign = load_benign_prompts(ROOT / "prompts/data/xstest_prompts.csv", args.benign_n)
    fl = cfg["train"].get("fuse_layer", cfg["train"]["rfa_layer"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    base = load_causal_lm(cfg["model"]["name"], device, dtype)
    model = PeftModel.from_pretrained(base, str(args.adapter.resolve()))
    model.eval()

    fuse_path = args.adapter / "mandatory_fuse.pt"
    fuse = None
    if fuse_path.exists():
        fuse = MandatoryFuse(
            model.config.hidden_size,
            rank=cfg["train"].get("fuse_rank", 64),
            init_scale=cfg["train"].get("fuse_init_scale", 0.02),
        )
        fuse.load_state_dict(torch.load(fuse_path, map_location=device, weights_only=True))
        fuse = fuse.to(device, dtype=torch.float32).eval()

    out = {}
    for name, fz in MODES:
        if name == "no_hook":
            nll = mean_benign_nll(model, tokenizer, benign)
        elif fuse is None:
            nll = float("nan")
        else:
            h = register_mandatory_fuse_hooks(model, fuse, fl, force_zero=bool(fz))
            nll = mean_benign_nll(model, tokenizer, benign)
            remove_fuse_hooks(h)
        out[name] = nll
        logger.info("%s nll=%.4f", name, nll)

    result = {"adapter": str(args.adapter), "fuse_layer": fl, "nll": out}
    if out.get("fuse_on") == out.get("fuse_on") and out.get("fuse_zero") == out.get("fuse_zero"):
        result["delta_fuse_zero"] = out["fuse_zero"] - out["fuse_on"]
    path = args.out or args.adapter / "eval_bypass.json"
    path.write_text(json.dumps(result, indent=2))
    release_gpu(model, tokenizer, fuse, base)


if __name__ == "__main__":
    main()
