#!/usr/bin/env python3
"""Evil layer×alpha sweep + C0/C1/C2/C4 dev eval (single model load)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
NLA_SRC = ROOT.parent / "nla_rsa_study" / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(NLA_SRC))

from common import cleanup_mps, get_device, load_causal_lm, model_device, resolve_torch_dtype  # noqa: E402

from src.config_loader import load_config, repo_root
from src.eval_metrics import score_responses
from src.generate_eval import build_stack, generate_one, load_manifest
from src.hooks import EditSpec, InterventionStack, register_intervention_hooks, remove_hooks
from src.vectors import load_vectors

logger = logging.getLogger(__name__)

EVIL_LAYERS = [20, 22, 24, 27]
EVIL_ALPHAS = [5.0, 10.0, 15.0, 20.0]


def run_condition(model, tokenizer, device, manifest, stack, max_new: int) -> list[str]:
    handles = register_intervention_hooks(model, stack) if stack.edits else []
    responses = []
    for row in manifest:
        prompt = row.get("prompt") or row.get("behavior") or row["text"]
        responses.append(generate_one(model, tokenizer, prompt, device, max_new))
    remove_hooks(handles)
    return responses


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=repo_root() / "configs/qwen7b_harmbench.yaml")
    p.add_argument("--manifest", type=Path, default=repo_root() / "prompts/harmbench_manifest_dev.jsonl")
    p.add_argument(
        "--conditions",
        default="C0,C1,C2,C4",
        help="Comma-separated intervention conditions (e.g. C0,C2)",
    )
    p.add_argument("--max-new", type=int, default=256, help="Shorter for dev sweeps")
    p.add_argument("--skip-sweep", action="store_true")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Conditions JSON path (default: outputs/ablations/dev_v2_conditions.json)",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["model"].get("device", "mps"))
    dtype = resolve_torch_dtype(cfg["model"]["dtype"], device)
    payload = load_vectors(repo_root() / cfg["paths"]["vectors"])
    manifest = load_manifest(args.manifest)
    axes = payload["axes"]

    logger.info("Loading model (dev n=%d, max_new=%d)", len(manifest), args.max_new)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    model = load_causal_lm(cfg["model"]["name"], device, dtype)

    out_dir = repo_root() / "outputs/ablations"
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_cfg = cfg.get("eval", {})
    judge_model = eval_cfg.get("judge_model", "gpt-4o-mini")
    cache_path = repo_root() / eval_cfg.get("judge_cache", "outputs/cache/harmbench_judge.jsonl")
    user_prompts = [row.get("prompt") or row["behavior"] for row in manifest]

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    condition_scores = {}
    for cond in conditions:
        stack = build_stack(cfg, cond, payload)
        logger.info("Running %s edits=%s", cond, stack.edits)
        resp = run_condition(model, tokenizer, device, manifest, stack, args.max_new)
        condition_scores[cond] = score_responses(
            user_prompts,
            resp,
            judge_model=judge_model,
            cache_path=cache_path,
            desc=f"judge-{cond}",
        )
        condition_scores[cond]["condition"] = cond
        condition_scores[cond]["judge_model"] = judge_model

    cond_path = args.output or (out_dir / "dev_v2_conditions.json")
    cond_path.write_text(json.dumps(condition_scores, indent=2))
    logger.info("Wrote %s", cond_path)
    for cond, s in condition_scores.items():
        logger.info(
            "%s: refusal=%.2f harmful_comply=%.2f",
            cond,
            s["refusal_rate"],
            s["harmful_comply_rate"],
        )

    if args.skip_sweep:
        cleanup_mps()
        return

    sweep_results = []
    for layer in tqdm(EVIL_LAYERS, desc="evil layers"):
        for alpha in EVIL_ALPHAS:
            stack = InterventionStack(
                edits=[EditSpec(type="steer+", axis="evil", layer=layer, alpha=alpha)],
                vectors=axes,
            )
            resp = run_condition(model, tokenizer, device, manifest, stack, args.max_new)
            s = score_responses(
                user_prompts,
                resp,
                judge_model=judge_model,
                cache_path=cache_path,
                desc=f"judge-L{layer}-a{alpha}",
            )
            s.update({"layer": layer, "alpha": alpha})
            sweep_results.append(s)
            logger.info(
                "L%d a=%.0f: refusal=%.2f harmful_comply=%.2f",
                layer,
                alpha,
                s["refusal_rate"],
                s["harmful_comply_rate"],
            )

    sweep_path = out_dir / "evil_sweep_v2.json"
    sweep_path.write_text(json.dumps(sweep_results, indent=2))
    best = max(sweep_results, key=lambda x: x["harmful_comply_rate"])
    logger.info(
        "Best evil sweep (max harmful_comply): L%s a=%s refusal=%.2f harmful_comply=%.2f",
        best["layer"],
        best["alpha"],
        best["refusal_rate"],
        best["harmful_comply_rate"],
    )
    logger.info("Wrote %s", sweep_path)
    cleanup_mps()


if __name__ == "__main__":
    main()
