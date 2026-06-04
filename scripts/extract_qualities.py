#!/usr/bin/env python3
"""Extract per-trait quality vectors into outputs/vectors/qwen7b_qualities.pt"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vendor.common import cleanup_mps, get_device, load_causal_lm, resolve_torch_dtype  # noqa: E402

from src.config_loader import load_config, repo_root  # noqa: E402
from src.extract_vectors import contrast_assistant_vectors  # noqa: E402
from src.harmbench_data import load_quality_contrast_conversations  # noqa: E402
from src.vectors import save_vectors  # noqa: E402

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=repo_root() / "configs/evil_qualities.yaml")
    p.add_argument("--trait", default=None, help="One trait id or all from config")
    p.add_argument("--merge", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["model"].get("device", "auto"))
    dtype = resolve_torch_dtype(cfg["model"]["dtype"], device)
    trait_ids = [args.trait] if args.trait else list(cfg["qualities"]["ids"])
    out = repo_root() / cfg["extraction"]["output"]

    import torch

    payload = {"model": cfg["model"]["name"], "axes": {}, "metadata": {}}
    if args.merge and out.exists():
        payload = torch.load(out, map_location="cpu", weights_only=False)
        logger.info("Merging into %s", out)

    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
        cfg["model"]["name"], trust_remote_code=True
    )
    model = load_causal_lm(cfg["model"]["name"], device, dtype)

    for tid in trait_ids:
        pairs_path = repo_root() / "prompts/quality_contrast" / f"{tid}.jsonl"
        if not pairs_path.exists() or pairs_path.stat().st_size == 0:
            logger.warning("Skip extract %s: no pairs at %s", tid, pairs_path)
            continue
        pos, neg = load_quality_contrast_conversations(tid)
        if len(pos) < 2:
            logger.warning("Skip extract %s: only %d pairs (need >=2)", tid, len(pos))
            continue
        logger.info("Extract %s: %d pairs", tid, len(pos))
        vecs, layer_meta = contrast_assistant_vectors(pos, neg, model, tokenizer, device)
        axis_key = f"q_{tid}"
        best_layer = max(layer_meta, key=lambda l: layer_meta[l]["separation"])
        payload["axes"][axis_key] = vecs
        payload["metadata"][axis_key] = {
            "trait_id": tid,
            "method": "chen_assistant_token_contrast",
            "n_pairs": len(pos),
            "layer_separation": layer_meta,
            "best_layer": int(best_layer),
            "best_separation": layer_meta[best_layer]["separation"],
        }
        logger.info("  %s best L%s sep=%.3f", axis_key, best_layer, layer_meta[best_layer]["separation"])

    save_vectors(out, payload)
    logger.info("Saved %s", out)
    cleanup_mps()


if __name__ == "__main__":
    main()
