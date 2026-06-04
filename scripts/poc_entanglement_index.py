#!/usr/bin/env python3
"""POC-B: entanglement index κ — benign NLL sensitivity to refusal-direction perturbation."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
NLA_SRC = ROOT.parent / "nla_rsa_study" / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(NLA_SRC))

from common import build_chat_inputs, cleanup_mps, get_device, load_causal_lm, model_device, resolve_torch_dtype  # noqa: E402

from src.config_loader import load_config, repo_root  # noqa: E402
from src.hooks import _get_layers, _hs, steer_direction  # noqa: E402
from src.vectors import load_vectors  # noqa: E402

logger = logging.getLogger(__name__)


def load_prompts_csv(path: Path, n: int) -> list[str]:
    import csv

    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            p = row.get("prompt") or row.get("text") or ""
            if p:
                out.append(p.strip())
    return out[:n]


@torch.no_grad()
def nll_with_hook(model, tokenizer, prompt: str, layer: int, perturb: float, direction: torch.Tensor) -> float:
    dev = model_device(model)
    ids = build_chat_inputs(tokenizer, prompt)["input_ids"].to(dev)
    if ids.shape[1] < 2:
        return 0.0
    layers = _get_layers(model)
    d = direction.to(dev)

    def hook(_m, _i, output):
        hs = _hs(output)
        if perturb == 0.0:
            return output
        new_hs = steer_direction(hs, d, perturb)
        return (new_hs,) + output[1:] if isinstance(output, tuple) else new_hs

    h = layers[layer].register_forward_hook(hook)
    out = model(ids)
    h.remove()
    logits = out.logits[:, :-1, :]
    targets = ids[:, 1:]
    return float(
        F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="mean"
        ).item()
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=repo_root() / "configs/qwen7b_harmbench.yaml")
    p.add_argument("--benign", type=Path, default=repo_root() / "prompts/data/xstest_prompts.csv")
    p.add_argument("--benign-n", type=int, default=20)
    p.add_argument("--layers", default="10,14,18,22,27")
    p.add_argument("--eps", type=float, default=2.0, help="Steer along +d_refusal")
    p.add_argument("--out", type=Path, default=repo_root() / "outputs/ablations/poc_b_entanglement_kappa.json")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["model"].get("device", "auto"))
    dtype = resolve_torch_dtype(cfg["model"]["dtype"], device)
    payload = load_vectors(repo_root() / cfg["paths"]["vectors"])
    prompts = load_prompts_csv(args.benign, args.benign_n)
    layers = [int(x) for x in args.layers.split(",")]

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    model = load_causal_lm(cfg["model"]["name"], device, dtype)

    rows = []
    for layer in layers:
        d = payload["axes"]["refusal"][layer]
        base_losses, pert_losses = [], []
        for prompt in tqdm(prompts, desc=f"L{layer}", leave=False):
            base_losses.append(nll_with_hook(model, tokenizer, prompt, layer, 0.0, d))
            pert_losses.append(nll_with_hook(model, tokenizer, prompt, layer, args.eps, d))
        base_m = sum(base_losses) / len(base_losses)
        pert_m = sum(pert_losses) / len(pert_losses)
        kappa = (pert_m - base_m) / max(args.eps, 1e-8)
        rows.append(
            {
                "layer": layer,
                "mean_nll_base": base_m,
                "mean_nll_perturb": pert_m,
                "delta_nll": pert_m - base_m,
                "kappa_per_eps": kappa,
                "eps": args.eps,
            }
        )
        logger.info("L%s κ≈%.5f ΔNLL=%.4f", layer, kappa, pert_m - base_m)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"benign_n": len(prompts), "rows": rows}, indent=2))
    logger.info("Wrote %s", args.out)
    cleanup_mps()


if __name__ == "__main__":
    main()
