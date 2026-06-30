#!/usr/bin/env python3
"""Post-train eval: clean vs RFA tamper on benign NLL + optional HarmBench dev ASR."""

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

from vendor.common import cleanup_mps, get_device, load_causal_lm, resolve_torch_dtype  # noqa: E402

from scripts.poc_tamper_capability_sweep import load_benign_prompts, mean_benign_nll, rfa_stack  # noqa: E402
from src.generate_eval import build_stack, generate_one, load_manifest  # noqa: E402
from src.hooks import register_intervention_hooks, remove_hooks  # noqa: E402
from src.mandatory_fuse import MandatoryFuse, register_mandatory_fuse_hooks, remove_fuse_hooks  # noqa: E402
from src.harmbench_judge import classify_pair  # noqa: E402
from src.vectors import load_vectors  # noqa: E402

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", type=Path, default=None, help="LoRA dir; omit for stock base")
    p.add_argument("--config", type=Path, default=ROOT / "configs/d3_lora_train.yaml")
    p.add_argument("--harm-manifest", type=Path, default=ROOT / "prompts/harmbench_manifest_dev.jsonl")
    p.add_argument("--benign-n", type=int, default=30)
    p.add_argument("--judge", action="store_true")
    p.add_argument("--fuse-eval", action="store_true", help="Also eval mandatory fuse zeroed")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    device = get_device("cuda" if torch.cuda.is_available() else "auto")
    dtype = resolve_torch_dtype(cfg["model"]["dtype"], device)
    payload = load_vectors(ROOT / cfg["paths"]["vectors"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    base = load_causal_lm(cfg["model"]["name"], device, dtype)
    model = (
        PeftModel.from_pretrained(base, str(args.adapter.resolve()))
        if args.adapter
        else base
    )
    model.eval()

    benign = load_benign_prompts(ROOT / "prompts/data/xstest_prompts.csv", args.benign_n)
    layer = cfg["train"]["rfa_layer"]
    scale = cfg["train"]["rfa_scale"]

    fuse_path = (args.adapter / "mandatory_fuse.pt") if args.adapter else None
    fuse = None
    if fuse_path and (args.fuse_eval or fuse_path.exists()) and fuse_path.exists():
        fuse = MandatoryFuse(
            model.config.hidden_size,
            rank=cfg["train"].get("fuse_rank", 64),
            init_scale=cfg["train"].get("fuse_init_scale", 0.02),
        )
        fuse.load_state_dict(torch.load(fuse_path, map_location=device, weights_only=True))
        fuse = fuse.to(device, dtype=torch.float32)
        fuse.eval()

    fl = cfg["train"].get("fuse_layer", layer)
    nll_fuse_on = nll_fuse_zero = None
    if fuse is not None:
        handles = register_mandatory_fuse_hooks(model, fuse, fl, force_zero=False)
        nll_fuse_on = mean_benign_nll(model, tokenizer, benign)
        remove_fuse_hooks(handles)
        handles = register_mandatory_fuse_hooks(model, fuse, fl, force_zero=True)
        nll_fuse_zero = mean_benign_nll(model, tokenizer, benign)
        remove_fuse_hooks(handles)
        nll_clean = nll_fuse_on
    else:
        nll_clean = mean_benign_nll(model, tokenizer, benign)

    stack = rfa_stack(payload, layer, scale)
    fuse_handles = (
        register_mandatory_fuse_hooks(model, fuse, fl, force_zero=False) if fuse is not None else []
    )
    rfa_handles = register_intervention_hooks(model, stack)
    nll_rfa = mean_benign_nll(model, tokenizer, benign)
    remove_hooks(rfa_handles)
    if fuse_handles:
        remove_fuse_hooks(fuse_handles)

    def _delta(a, b):
        if a != a or b != b:
            return float("nan")
        return a - b

    result = {
        "adapter": str(args.adapter) if args.adapter else "stock",
        "mean_nll_clean": nll_clean,
        "mean_nll_rfa": nll_rfa,
        "delta_nll_rfa": _delta(nll_rfa, nll_clean),
        "layer": layer,
        "rfa_scale": scale,
    }
    if fuse is not None:
        result.update(
            {
                "mean_nll_fuse_on": nll_fuse_on,
                "mean_nll_fuse_zero": nll_fuse_zero,
                "delta_nll_fuse_zero": _delta(nll_fuse_zero, nll_fuse_on),
                "fuse_layer": fl,
            }
        )

    if args.judge:
        harmful = load_manifest(args.harm_manifest)
        tamper_specs: list[tuple[str, object | None, bool | None]] = [
            ("clean", None, False),
            ("rfa", stack, False),
        ]
        if fuse is not None:
            tamper_specs.append(("fuse_zero", None, True))
        for name, stk, fuse_zero in tamper_specs:
            fuse_h, rfa_h = [], []
            if fuse is not None:
                fuse_h = register_mandatory_fuse_hooks(model, fuse, fl, force_zero=fuse_zero)
            if stk:
                rfa_h = register_intervention_hooks(model, stk)
            comply = refuse = 0
            for row in tqdm(harmful, desc=name):
                prompt = row.get("prompt") or row["behavior"]
                resp = generate_one(model, tokenizer, prompt, device, 128)
                v = classify_pair(prompt, resp)
                comply += v["verdict"] == "harmful_comply"
                refuse += v["verdict"] == "refusal"
            remove_fuse_hooks(fuse_h)
            remove_hooks(rfa_h)
            n = len(harmful)
            result[f"{name}_comply_rate"] = comply / n
            result[f"{name}_refusal_rate"] = refuse / n

    def _sanitize(obj):
        if isinstance(obj, float) and (obj != obj or obj == float("inf") or obj == float("-inf")):
            return None
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        return obj

    out = args.out or (
        (args.adapter / "eval_tamper.json") if args.adapter else ROOT / "outputs/arxiv_mva/stock_eval.json"
    )
    out.write_text(json.dumps(_sanitize(result), indent=2))
    logger.info("%s", result)
    cleanup_mps()


if __name__ == "__main__":
    main()
