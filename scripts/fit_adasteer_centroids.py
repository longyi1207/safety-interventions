#!/usr/bin/env python3
"""Fit AdaSteer mu_c_harmful + linear λ coeffs from C1-comply vs C0-refuse activations."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
NLA_SRC = ROOT.parent / "nla_rsa_study" / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(NLA_SRC))

from common import build_chat_inputs, cleanup_mps, get_device, load_causal_lm, model_device, resolve_torch_dtype  # noqa: E402

from src.adasteer import AdaSteerParams, build_adasteer_params  # noqa: E402
from src.config_loader import load_config, repo_root  # noqa: E402
from src.hooks import _get_layers, _hs  # noqa: E402
from src.vectors import load_vectors, save_vectors  # noqa: E402

logger = logging.getLogger(__name__)


@torch.no_grad()
def capture_last_prefill(model, tokenizer, prompt: str, layer: int) -> torch.Tensor:
    layers = _get_layers(model)
    dev = model_device(model)
    ids = build_chat_inputs(tokenizer, prompt)["input_ids"].to(dev)
    captured: dict[str, torch.Tensor] = {}

    def hook(_m, _i, output):
        captured["h"] = _hs(output)[:, -1:, :].detach().cpu()

    h = layers[layer].register_forward_hook(hook)
    model(ids)
    h.remove()
    return captured["h"].squeeze(0).squeeze(0)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=repo_root() / "configs/qwen7b_harmbench.yaml")
    p.add_argument(
        "--review",
        type=Path,
        default=repo_root() / "outputs/cloud_pull/si-20260602-044255/ablations/c1_c4_review_main.jsonl",
    )
    p.add_argument("--layer", type=int, default=18)
    p.add_argument("--max-comply", type=int, default=40)
    p.add_argument("--max-refuse", type=int, default=40)
    p.add_argument("--vectors", type=Path, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    vec_path = args.vectors or repo_root() / cfg["paths"]["vectors"]
    payload = load_vectors(vec_path)
    rows = [json.loads(l) for l in args.review.read_text().splitlines() if l.strip()]

    comply = [r for r in rows if r.get("c1_verdict") == "harmful_comply"][: args.max_comply]
    refuse = [r for r in rows if r.get("c1_verdict") == "refusal"][: args.max_refuse]
    logger.info("Fit set: comply=%d refuse=%d", len(comply), len(refuse))

    device = get_device(cfg["model"].get("device", "auto"))
    dtype = resolve_torch_dtype(cfg["model"]["dtype"], device)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    model = load_causal_lm(cfg["model"]["name"], device, dtype)

    comply_h = [capture_last_prefill(model, tokenizer, r["prompt"], args.layer) for r in tqdm(comply, desc="comply")]
    refuse_h = [capture_last_prefill(model, tokenizer, r["prompt"], args.layer) for r in tqdm(refuse, desc="refuse")]
    mu_c = torch.stack(comply_h).mean(0)

    params = build_adasteer_params(payload, layer=args.layer, hd_axis="evil", hd_layer=args.layer)
    params.mu_c_harmful = mu_c
    d_rd = params.v_rd.float() / (params.v_rd.float().norm() + 1e-8)
    d_hd = params.v_hd.float() / (params.v_hd.float().norm() + 1e-8)

    pos_rd_c = np.array([float((h.float() - mu_c.float()) @ d_rd) for h in comply_h])
    pos_rd_r = np.array([float((h.float() - mu_c.float()) @ d_rd) for h in refuse_h])
    pos_hd_c = np.array([float((h.float() - mu_c.float()) @ d_hd) for h in comply_h])
    pos_hd_r = np.array([float((h.float() - mu_c.float()) @ d_hd) for h in refuse_h])

    # Target λ: comply needs high λ_r (~2), refuse needs low (~0)
    y_r = np.concatenate([np.ones(len(comply_h)) * 2.0, np.zeros(len(refuse_h))])
    x_r = np.concatenate([pos_rd_c, pos_rd_r])
    wr, br = np.polyfit(x_r, y_r, 1)

    y_c = np.concatenate([np.zeros(len(comply_h)), np.ones(len(refuse_h)) * 0.5])
    x_c = np.concatenate([pos_hd_c, pos_hd_r])
    wc, bc = np.polyfit(x_c, y_c, 1)

    meta = dict(payload.get("metadata", {}))
    meta["adasteer"] = {
        "layer": args.layer,
        "mu_c_harmful": {str(args.layer): mu_c},
        "w_r": float(wr),
        "b_r": float(br),
        "w_c": float(wc),
        "b_c": float(bc),
        "n_comply": len(comply_h),
        "n_refuse": len(refuse_h),
    }
    payload["metadata"] = meta
    save_vectors(vec_path, payload)
    logger.info("Saved AdaSteer fit to %s: w_r=%.3f b_r=%.3f w_c=%.3f b_c=%.3f", vec_path, wr, br, wc, bc)
    cleanup_mps()


if __name__ == "__main__":
    main()
