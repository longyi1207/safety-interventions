#!/usr/bin/env python3
"""HarmBench eval: refusal RFA + all quality trait steers (and ablations)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NLA_SRC = ROOT.parent / "nla_rsa_study" / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(NLA_SRC))

from common import cleanup_mps, get_device, load_causal_lm, resolve_torch_dtype  # noqa: E402

from src.config_loader import load_config, repo_root  # noqa: E402
from src.eval_metrics import score_responses  # noqa: E402
from src.generate_eval import build_stack, generate_one, load_manifest  # noqa: E402
from src.hooks import EditSpec, InterventionStack, register_intervention_hooks, remove_hooks  # noqa: E402
from src.vectors import load_vectors  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_CONDITIONS = "C0,C1,C4,C_all_traits,C_all_traits_rfa"


def load_combined_payload(
    base_path: Path,
    quality_path: Path,
) -> dict:
    base = load_vectors(base_path)
    qual = load_vectors(quality_path)
    return {
        "model": base.get("model") or qual.get("model"),
        "axes": {**base["axes"], **qual["axes"]},
        "metadata": {**base.get("metadata", {}), **qual.get("metadata", {})},
    }


def best_layer(meta: dict, axis: str, default: int = 27) -> int:
    m = meta.get(axis, {})
    return int(m.get("best_layer", default))


def build_all_traits_stack(
    payload: dict,
    trait_ids: list[str],
    *,
    rfa_refusal: bool,
    refusal_layer: int = 18,
    quality_alpha: float = 6.0,
    evil_steer: bool = False,
    evil_layer: int = 27,
    evil_alpha: float = 10.0,
) -> InterventionStack:
    meta = payload.get("metadata", {})
    edits: list[EditSpec] = []
    if rfa_refusal:
        rfa = meta.get("refusal", {}).get("rfa_harmless_mean", {})
        coeff = float(rfa.get(refusal_layer, rfa.get(str(refusal_layer), 0.0)))
        edits.append(EditSpec(type="rfa_r", axis="refusal", layer=refusal_layer, harmless_mean_coeff=coeff))
    if evil_steer and "evil" in payload["axes"]:
        edits.append(EditSpec(type="steer+", axis="evil", layer=evil_layer, alpha=evil_alpha))
    for tid in trait_ids:
        axis = f"q_{tid}"
        if axis not in payload["axes"]:
            logger.warning("Skip axis %s (not in payload)", axis)
            continue
        layer = best_layer(meta, axis)
        edits.append(EditSpec(type="steer+", axis=axis, layer=layer, alpha=quality_alpha))
    return InterventionStack(edits=edits, vectors=payload["axes"])


def run_condition(model, tokenizer, manifest, stack, max_new: int, device) -> list[str]:
    handles = register_intervention_hooks(model, stack) if stack.edits else []
    responses = []
    for row in manifest:
        prompt = row.get("prompt") or row["behavior"]
        responses.append(generate_one(model, tokenizer, prompt, device, max_new))
    remove_hooks(handles)
    return responses


def resolve_stack(name: str, cfg: dict, payload: dict, trait_ids: list[str], quality_alpha: float) -> InterventionStack:
    if name in ("C0", "C1", "C2", "C4"):
        return build_stack(cfg, name, payload)
    if name == "C_all_traits":
        return build_all_traits_stack(payload, trait_ids, rfa_refusal=False, quality_alpha=quality_alpha)
    if name == "C_all_traits_rfa":
        return build_all_traits_stack(payload, trait_ids, rfa_refusal=True, quality_alpha=quality_alpha)
    if name == "C_all_traits_rfa_evil":
        return build_all_traits_stack(
            payload, trait_ids, rfa_refusal=True, quality_alpha=quality_alpha, evil_steer=True
        )
    raise KeyError(f"Unknown condition {name}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=repo_root() / "configs/qwen7b_harmbench.yaml")
    p.add_argument("--qconfig", type=Path, default=repo_root() / "configs/evil_qualities.yaml")
    p.add_argument("--manifest", type=Path, default=repo_root() / "prompts/harmbench_manifest_dev.jsonl")
    p.add_argument("--base-vectors", type=Path, default=repo_root() / "outputs/vectors/qwen7b_vectors.pt")
    p.add_argument("--quality-vectors", type=Path, default=repo_root() / "outputs/vectors/qwen7b_qualities_ortho.pt")
    p.add_argument("--conditions", default=DEFAULT_CONDITIONS)
    p.add_argument("--quality-alpha", type=float, default=6.0, help="Per-trait alpha when stacking all q_*")
    p.add_argument("--max-new", type=int, default=256)
    p.add_argument("--output", type=Path, default=None, help="Output JSON (default: all_traits_benchmark_dev.json)")
    args = p.parse_args()

    hb = load_config(args.config)
    qc = load_config(args.qconfig)
    trait_ids = list(qc["qualities"]["ids"])
    payload = load_combined_payload(args.base_vectors, args.quality_vectors)
    present = [t for t in trait_ids if f"q_{t}" in payload["axes"]]
    logger.info("Quality axes present: %d/%d %s", len(present), len(trait_ids), present)

    device = get_device(hb["model"].get("device", "auto"))
    dtype = resolve_torch_dtype(hb["model"]["dtype"], device)
    manifest = load_manifest(args.manifest)
    user_prompts = [row.get("prompt") or row["behavior"] for row in manifest]

    eval_cfg = hb.get("eval", {})
    judge_model = eval_cfg.get("judge_model", "gpt-4o-mini")
    cache_path = repo_root() / eval_cfg.get("judge_cache", "outputs/cache/harmbench_judge.jsonl")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hb["model"]["name"], trust_remote_code=True)
    model = load_causal_lm(hb["model"]["name"], device, dtype)

    results = {}
    for cond in [c.strip() for c in args.conditions.split(",") if c.strip()]:
        stack = resolve_stack(cond, hb, payload, present, args.quality_alpha)
        logger.info("%s: %d edits %s", cond, len(stack.edits), stack.edits)
        resp = run_condition(model, tokenizer, manifest, stack, args.max_new, device)
        s = score_responses(user_prompts, resp, judge_model=judge_model, cache_path=cache_path, desc=cond)
        s.update({"condition": cond, "n_edits": len(stack.edits), "quality_alpha": args.quality_alpha})
        results[cond] = s

    out_dir = repo_root() / "outputs/ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output or (out_dir / "all_traits_benchmark_dev.json")
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Wrote %s", out_path)
    for cond, s in results.items():
        logger.info(
            "%s: refusal=%.2f harmful_comply=%.2f",
            cond,
            s.get("refusal_rate", 0),
            s.get("harmful_comply_rate", 0),
        )
    cleanup_mps()


if __name__ == "__main__":
    main()
