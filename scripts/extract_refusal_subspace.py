#!/usr/bin/env python3
"""Extract 2nd/3rd refusal directions via per-pair residual at last user token."""

from __future__ import annotations

import argparse
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
from src.extract_vectors import extract_mean_hidden_prompt  # noqa: E402
from src.harmbench_data import load_harmful_behaviors, load_harmless_xstest  # noqa: E402
from src.vectors import gram_schmidt, load_vectors, normalize_layers, save_vectors  # noqa: E402

logger = logging.getLogger(__name__)


@torch.no_grad()
def per_pair_diffs(model, tokenizer, harmful, harmless, device, layer: int) -> list[torch.Tensor]:
    n = min(len(harmful), len(harmless))
    diffs: list[torch.Tensor] = []
    for i in tqdm(range(n), desc="pair diffs"):
        for msg, sign in ((harmful[i], 1), (harmless[i], -1)):
            inp = build_chat_inputs(tokenizer, msg)
            ids = inp["input_ids"].to(device)
            idx = len(inp["input_ids_list"]) - 1
            out = model(input_ids=ids, output_hidden_states=True)
            h = out.hidden_states[layer + 1][0, idx].float().cpu()
            if sign == 1:
                pos_h = h
            else:
                diffs.append(pos_h - h)
    return diffs


def subspace_from_diffs(diffs: list[torch.Tensor], d1: torch.Tensor, k: int = 3) -> list[torch.Tensor]:
    """Gram-Schmidt on mean-residual per pair + global mean, after removing d1."""
    d1u = d1.float() / (d1.float().norm() + 1e-8)
    seeds: list[torch.Tensor] = []
    for d in diffs:
        r = d.float() - torch.dot(d.float(), d1u) * d1u
        if r.norm() > 1e-6:
            seeds.append(r / r.norm())
    mean_r = sum(diffs[i].float() - torch.dot(diffs[i].float(), d1u) * d1u for i in range(len(diffs)))
    if mean_r.norm() > 1e-6:
        seeds.insert(0, mean_r / mean_r.norm())
    if not seeds:
        return []
    return gram_schmidt(seeds)[:k]


def harmless_proj(model, tokenizer, harmless, device, layer: int, direction: torch.Tensor) -> float:
    n_layers = model.config.num_hidden_layers
    neg_m = extract_mean_hidden_prompt(model, tokenizer, harmless, device, n_layers)
    d = direction.float() / (direction.float().norm() + 1e-8)
    return float(torch.dot(neg_m[layer].float(), d).item())


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=repo_root() / "configs/qwen7b_harmbench.cloud.yaml")
    p.add_argument("--vectors", type=Path, default=repo_root() / "outputs/vectors/qwen7b_vectors.pt")
    p.add_argument("--layer", type=int, default=18)
    p.add_argument("--n-pairs", type=int, default=50)
    p.add_argument("--k", type=int, default=3, help="Subspace rank including d1 check (extract k-1 extra)")
    p.add_argument("--out", type=Path, default=repo_root() / "outputs/vectors/refusal_subspace_L18.pt")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["model"].get("device", "auto"))
    dtype = resolve_torch_dtype(cfg["model"]["dtype"], device)

    base = load_vectors(args.vectors)
    d1 = base["axes"]["refusal"][args.layer].float()

    harmful = load_harmful_behaviors(limit=args.n_pairs)
    harmless = load_harmless_xstest(limit=args.n_pairs)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    model = load_causal_lm(cfg["model"]["name"], device, dtype)

    diffs = per_pair_diffs(model, tokenizer, harmful, harmless, device, args.layer)
    extra = subspace_from_diffs(diffs, d1, k=max(1, args.k - 1))

    axes: dict[str, dict[int, torch.Tensor]] = {"refusal": {args.layer: d1}}
    rfa_map: dict[str, dict[int, float]] = {}
    base_rfa = base.get("metadata", {}).get("refusal", {}).get("rfa_harmless_mean", {})
    rfa_map["refusal"] = {int(k): float(v) for k, v in base_rfa.items()}

    for i, vec in enumerate(extra, start=2):
        name = f"refusal_{i}"
        axes[name] = {args.layer: vec}
        hm = harmless_proj(model, tokenizer, harmless[: args.n_pairs], device, args.layer, vec)
        rfa_map[name] = {args.layer: hm}
        logger.info(
            "%s L%d harmless_proj=%.4f cos_d1=%.4f",
            name,
            args.layer,
            hm,
            float(torch.dot(vec, d1 / (d1.norm() + 1e-8))),
        )

    payload = {
        "model": cfg["model"]["name"],
        "layer": args.layer,
        "axes": axes,
        "metadata": {
            "method": "per_pair_residual_gram_schmidt",
            "n_pairs": len(diffs),
            "rfa_harmless_mean": rfa_map,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_vectors(args.out, payload)
    logger.info("Wrote %s (%d extra axes)", args.out, len(extra))
    cleanup_mps()


if __name__ == "__main__":
    main()
