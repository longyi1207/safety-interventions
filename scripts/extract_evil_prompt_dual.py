#!/usr/bin/env python3
"""Evil direction at last prompt token: EVIL_SYSTEM vs NEUTRAL_SYSTEM (no assistant text)."""

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

from common import cleanup_mps, get_device, load_causal_lm, model_device, resolve_torch_dtype  # noqa: E402

from src.chat_extract import EVIL_SYSTEM, NEUTRAL_SYSTEM  # noqa: E402
from src.config_loader import load_config, repo_root  # noqa: E402
from src.harmbench_data import load_harmful_behaviors  # noqa: E402
from src.vectors import load_vectors, normalize_layers, save_vectors  # noqa: E402

logger = logging.getLogger(__name__)


@torch.no_grad()
def last_prompt_hidden(model, tokenizer, user: str, system: str | None, device) -> torch.Tensor:
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    if not isinstance(ids, list):
        ids = list(ids)
    dev = model_device(model)
    out = model(input_ids=torch.tensor([ids], device=dev), output_hidden_states=True)
    n_layers = model.config.num_hidden_layers
    idx = len(ids) - 1
    return torch.stack([out.hidden_states[i + 1][0, idx].float().cpu() for i in range(n_layers)])


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=repo_root() / "configs/qwen7b_harmbench.yaml")
    p.add_argument("--base-vectors", type=Path, default=repo_root() / "outputs/vectors/qwen7b_vectors.pt")
    p.add_argument("--n-prompts", type=int, default=50)
    p.add_argument("--out", type=Path, default=repo_root() / "outputs/vectors/qwen7b_vectors_prompt_dual.pt")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["model"].get("device", "auto"))
    dtype = resolve_torch_dtype(cfg["model"]["dtype"], device)

    prompts = load_harmful_behaviors(limit=args.n_prompts)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    model = load_causal_lm(cfg["model"]["name"], device, dtype)

    evil_sum = None
    neutral_sum = None
    for user in tqdm(prompts, desc="prompt dual"):
        e = last_prompt_hidden(model, tokenizer, user, EVIL_SYSTEM, device)
        n = last_prompt_hidden(model, tokenizer, user, NEUTRAL_SYSTEM, device)
        evil_sum = e if evil_sum is None else evil_sum + e
        neutral_sum = n if neutral_sum is None else neutral_sum + n

    diff = (evil_sum - neutral_sum) / len(prompts)
    vecs = {i: diff[i] for i in range(diff.shape[0])}
    norm = normalize_layers(vecs)
    meta_proj = {}
    for i in range(diff.shape[0]):
        d = norm[i]
        meta_proj[i] = {
            "pos_proj": float(torch.dot(evil_sum[i].float() / len(prompts), d).item()),
            "neg_proj": float(torch.dot(neutral_sum[i].float() / len(prompts), d).item()),
        }
    best_layer = max(meta_proj, key=lambda k: meta_proj[k]["pos_proj"] - meta_proj[k]["neg_proj"])
    separation = meta_proj[best_layer]["pos_proj"] - meta_proj[best_layer]["neg_proj"]
    logger.info("Best layer L%s separation=%.3f", best_layer, separation)

    payload = load_vectors(args.base_vectors) if args.base_vectors.exists() else {
        "model": cfg["model"]["name"],
        "axes": {},
        "metadata": {},
    }
    payload["axes"]["evil"] = norm
    payload["metadata"]["evil"] = {
        "method": "dual_system_last_prompt_token",
        "extraction_mode": "dual_system_evil_vs_neutral_prompt",
        "n_prompts": len(prompts),
        "best_layer": int(best_layer),
        "layer_separation": meta_proj,
        "separation": float(separation),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_vectors(args.out, payload)
    logger.info("Saved %s", args.out)
    cleanup_mps()


if __name__ == "__main__":
    main()
