#!/usr/bin/env python3
"""Phase B: per-trait steering sweeps + interaction conditions (LLM judge eval)."""

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
sys.path.insert(0, str(ROOT))

from vendor.common import cleanup_mps, get_device, load_causal_lm, resolve_torch_dtype  # noqa: E402

from src.config_loader import load_config, repo_root  # noqa: E402
from src.eval_metrics import score_responses  # noqa: E402
from src.generate_eval import load_manifest  # noqa: E402
from src.hooks import EditSpec, InterventionStack, register_intervention_hooks, remove_hooks
from src.vectors import load_vectors  # noqa: E402

logger = logging.getLogger(__name__)

# Fractional factorial–style interaction set (8 conditions) for v2
INTERACTION_DESIGNS: list[tuple[str, list[tuple[str, int, float]]]] = [
    ("B0_empty", []),
    ("B1_malevolence", [("malevolence", 27, 10.0)]),
    ("B2_manipulation", [("manipulation", 27, 10.0)]),
    ("B3_moral_disengagement", [("moral_disengagement", 27, 10.0)]),
    ("B4_malev_manip", [("malevolence", 27, 10.0), ("manipulation", 27, 10.0)]),
    ("B5_malev_moral", [("malevolence", 27, 10.0), ("moral_disengagement", 27, 10.0)]),
    ("B6_manip_moral", [("manipulation", 27, 10.0), ("moral_disengagement", 27, 10.0)]),
    ("B7_triple", [("malevolence", 27, 8.0), ("manipulation", 27, 8.0), ("moral_disengagement", 27, 8.0)]),
]


@torch.no_grad()
def run_condition(model, tokenizer, manifest, stack, max_new: int, device) -> list[str]:
    from src.generate_eval import generate_one

    handles = register_intervention_hooks(model, stack) if stack.edits else []
    responses = []
    for row in manifest:
        prompt = row.get("prompt") or row["behavior"]
        responses.append(generate_one(model, tokenizer, prompt, device, max_new))
    remove_hooks(handles)
    return responses


def build_stack(payload: dict, edits: list[tuple[str, int, float]]) -> InterventionStack:
    axes = payload["axes"]
    spec = [
        EditSpec(type="steer+", axis=f"q_{tid}", layer=layer, alpha=alpha) for tid, layer, alpha in edits
    ]
    return InterventionStack(edits=spec, vectors=axes)


def best_layer_for_trait(payload: dict, trait_id: str, default: int = 27) -> int:
    meta = payload.get("metadata", {}).get(f"q_{trait_id}", {})
    return int(meta.get("best_layer", default))


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=repo_root() / "configs/evil_qualities.yaml")
    p.add_argument("--vectors", type=Path, default=None)
    p.add_argument("--manifest", type=Path, default=repo_root() / "prompts/harmbench_manifest_dev.jsonl")
    p.add_argument("--ortho", action="store_true", help="Use orthogonalized vectors file")
    p.add_argument("--stage", choices=["singles", "interactions", "all"], default="all")
    p.add_argument("--max-new", type=int, default=256)
    p.add_argument("--alpha", type=float, default=10.0)
    args = p.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["model"].get("device", "auto"))
    dtype = resolve_torch_dtype(cfg["model"]["dtype"], device)
    vec_path = args.vectors or (
        repo_root() / "outputs/vectors/qwen7b_qualities_ortho.pt"
        if args.ortho
        else repo_root() / cfg["extraction"]["output"]
    )
    payload = load_vectors(vec_path)
    manifest = load_manifest(args.manifest)
    user_prompts = [row.get("prompt") or row["behavior"] for row in manifest]

    eval_cfg = cfg.get("eval", {}) or {}
    judge_model = eval_cfg.get("judge_model", "gpt-4o-mini")
    cache_path = repo_root() / eval_cfg.get("judge_cache", "outputs/cache/harmbench_judge.jsonl")

    logger.info("Loading model; vectors=%s", vec_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    model = load_causal_lm(cfg["model"]["name"], device, dtype)

    results: dict = {}
    trait_ids = list(cfg["qualities"]["ids"])

    if args.stage in ("singles", "all"):
        for tid in trait_ids:
            axis = f"q_{tid}"
            if axis not in payload["axes"]:
                logger.warning("Skip %s: axis missing", tid)
                continue
            layer = best_layer_for_trait(payload, tid)
            name = f"single_{tid}"
            stack = build_stack(payload, [(tid, layer, args.alpha)])
            logger.info("Running %s L%d a=%s", name, layer, args.alpha)
            resp = run_condition(model, tokenizer, manifest, stack, args.max_new, device)
            s = score_responses(user_prompts, resp, judge_model=judge_model, cache_path=cache_path, desc=name)
            s.update({"trait": tid, "layer": layer, "alpha": args.alpha, "orthogonalized": args.ortho})
            results[name] = s

    if args.stage in ("interactions", "all"):
        for name, edits in INTERACTION_DESIGNS:
            edits_resolved = []
            for tid, layer, alpha in edits:
                if f"q_{tid}" not in payload["axes"]:
                    continue
                edits_resolved.append((tid, layer or best_layer_for_trait(payload, tid), alpha))
            stack = build_stack(payload, edits_resolved)
            logger.info("Running %s edits=%s", name, stack.edits)
            resp = run_condition(model, tokenizer, manifest, stack, args.max_new, device)
            s = score_responses(user_prompts, resp, judge_model=judge_model, cache_path=cache_path, desc=name)
            s.update({"design": name, "edits": edits_resolved, "orthogonalized": args.ortho})
            results[name] = s

    out_dir = repo_root() / "outputs/ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_ortho" if args.ortho else ""
    out_path = out_dir / f"phase_b_{args.stage}{suffix}.json"
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Wrote %s", out_path)
    cleanup_mps()


if __name__ == "__main__":
    main()
