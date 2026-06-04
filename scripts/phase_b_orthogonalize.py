#!/usr/bin/env python3
"""Gram-Schmidt orthogonalize quality axes per layer; emit cosine diagnostics."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config_loader import load_config, repo_root  # noqa: E402
from src.vectors import cosine_at_layer, gram_schmidt, load_vectors, save_vectors  # noqa: E402

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=repo_root() / "configs/evil_qualities.yaml")
    p.add_argument("--in-vectors", type=Path, default=None)
    p.add_argument("--out-vectors", type=Path, default=repo_root() / "outputs/vectors/qwen7b_qualities_ortho.pt")
    args = p.parse_args()

    cfg = load_config(args.config)
    in_path = args.in_vectors or repo_root() / cfg["extraction"]["output"]
    payload = load_vectors(in_path)
    trait_ids = [f"q_{t}" for t in cfg["qualities"]["ids"]]
    layers = sorted(payload["axes"][trait_ids[0]].keys())

    cos_report: dict = {"by_layer": {}}
    present = [t for t in trait_ids if t in payload["axes"]]
    for layer in layers:
        mat: dict[str, dict[str, float]] = {}
        for a in present:
            if layer not in payload["axes"][a]:
                continue
            mat[a] = {}
            for b in present:
                if layer not in payload["axes"][b]:
                    continue
                mat[a][b] = cosine_at_layer(payload["axes"][a], payload["axes"][b], layer)
        cos_report["by_layer"][str(layer)] = mat

    ortho_payload = {
        "model": payload["model"],
        "axes": {},
        "metadata": {**payload.get("metadata", {}), "orthogonalize": {"order": trait_ids, "method": "gram_schmidt"}},
    }

    for layer in layers:
        vecs = []
        keys = []
        for tid in trait_ids:
            if tid in payload["axes"] and layer in payload["axes"][tid]:
                vecs.append(payload["axes"][tid][layer])
                keys.append(tid)
        if len(vecs) < 2:
            continue
        ortho = gram_schmidt(vecs)
        for tid, ov in zip(keys, ortho):
            ortho_payload["axes"].setdefault(tid, {})[layer] = ov

    save_vectors(args.out_vectors, ortho_payload)
    diag_path = repo_root() / "outputs/ablations/quality_cosine_before_ortho.json"
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    diag_path.write_text(json.dumps(cos_report, indent=2))
    logger.info("Saved orthogonalized vectors %s and cosine report %s", args.out_vectors, diag_path)


if __name__ == "__main__":
    main()
