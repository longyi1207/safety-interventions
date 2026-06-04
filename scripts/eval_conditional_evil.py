#!/usr/bin/env python3
"""Conditional evil eval: does steering shift content when model still answers?

Two LLM judges (gpt-4o-mini):
  1. HarmBench verdict — refusal | harmful_comply | partial | benign_unrelated
  2. Evil persona — score_0_10 + evil_persona bool (on selected subsets)

Runs harmful manifest (C0/C2/C4) + benign xstest (C0/C2/trait).
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vendor.common import cleanup_mps, get_device, load_causal_lm, resolve_torch_dtype  # noqa: E402

from src.config_loader import load_config, repo_root  # noqa: E402
from src.eval_metrics import score_responses  # noqa: E402
from src.generate_eval import build_stack, generate_one, load_manifest  # noqa: E402
from src.harmbench_data import load_harmless_xstest  # noqa: E402
from src.harmbench_judge import (  # noqa: E402
    VERDICT_HARMFUL_COMPLY,
    VERDICT_PARTIAL,
    VERDICT_REFUSAL,
    classify_evil_persona,
)
from src.hooks import register_intervention_hooks, remove_hooks  # noqa: E402
from src.vectors import load_vectors  # noqa: E402

logger = logging.getLogger(__name__)

NON_REFUSAL = {VERDICT_HARMFUL_COMPLY, VERDICT_PARTIAL, "benign_unrelated"}
COMPLY = {VERDICT_HARMFUL_COMPLY, VERDICT_PARTIAL}


def run_condition(model, tokenizer, prompts, stack, max_new, device) -> list[str]:
    handles = register_intervention_hooks(model, stack) if stack.edits else []
    responses = [generate_one(model, tokenizer, p, device, max_new) for p in prompts]
    remove_hooks(handles)
    return responses


def evil_stats(prompts: list[str], responses: list[str], judge_model: str) -> dict:
    if not prompts:
        return {"n": 0, "evil_persona_rate": None, "mean_score": None, "scores": []}
    scores = []
    hits = 0
    details = []
    for p, r in zip(prompts, responses):
        lab = classify_evil_persona(p, r, model=judge_model)
        scores.append(lab["score_0_10"])
        hits += int(lab["evil_persona"])
        details.append(lab)
    return {
        "n": len(prompts),
        "evil_persona_rate": hits / len(prompts),
        "mean_score": statistics.mean(scores),
        "scores": scores,
        "details": details,
    }


def subset_from_labels(
    prompts: list[str],
    responses: list[str],
    labels: list[dict],
    verdicts: set[str],
) -> tuple[list[str], list[str]]:
    ps, rs = [], []
    for p, r, lab in zip(prompts, responses, labels):
        if lab["verdict"] in verdicts:
            ps.append(p)
            rs.append(r)
    return ps, rs


def eval_harmful_arm(
    model,
    tokenizer,
    device,
    cfg,
    payload,
    manifest,
    conditions: list[str],
    judge_model: str,
    cache_path: Path,
    max_new: int,
    pair_ref: str = "C0",
) -> dict:
    prompts = [row.get("prompt") or row["behavior"] for row in manifest]
    ids = [row.get("id", str(i)) for i, row in enumerate(manifest)]
    out: dict = {}

    for cond in conditions:
        stack = build_stack(cfg, cond, payload)
        logger.info("%s: generate n=%d edits=%s", cond, len(prompts), stack.edits)
        responses = run_condition(model, tokenizer, prompts, stack, max_new, device)
        scored = score_responses(
            prompts,
            responses,
            judge_model=judge_model,
            cache_path=cache_path,
            desc=f"harm-{cond}",
            save_labels=True,
        )
        labels = scored["labels"]

        arms = {
            "all": (prompts, responses),
            "non_refusal": subset_from_labels(prompts, responses, labels, NON_REFUSAL),
            "harmful_comply": subset_from_labels(prompts, responses, labels, COMPLY),
        }
        cond_out = {
            "condition": cond,
            "aggregate": {k: v for k, v in scored.items() if k != "labels"},
            "subsets": {},
            "pairs": [],
        }
        for name, (ps, rs) in arms.items():
            cond_out["subsets"][name] = evil_stats(ps, rs, judge_model)

        # Paired: same prompt, both pair_ref and this cond gave harmful_comply/partial.
        # Only harmful_comply/partial — benign_unrelated (copyright/lyrics) are not evil opportunities.
        if cond != pair_ref and pair_ref in out:
            ref_labels = out[pair_ref]["labels"]
            for i, (pid, p, r, lab) in enumerate(zip(ids, prompts, responses, labels)):
                if lab["verdict"] not in COMPLY:
                    continue
                if ref_labels[i]["verdict"] not in COMPLY:
                    continue
                ref_r = out[pair_ref]["responses"][i]
                e_ref = classify_evil_persona(p, ref_r, model=judge_model)
                e_cond = classify_evil_persona(p, r, model=judge_model)
                cond_out["pairs"].append(
                    {
                        "id": pid,
                        "prompt": p[:200],
                        "ref_score": e_ref["score_0_10"],
                        "cond_score": e_cond["score_0_10"],
                        "delta": e_cond["score_0_10"] - e_ref["score_0_10"],
                        "ref_evil": e_ref["evil_persona"],
                        "cond_evil": e_cond["evil_persona"],
                    }
                )
            if cond_out["pairs"]:
                deltas = [x["delta"] for x in cond_out["pairs"]]
                cond_out["paired"] = {
                    "pair_ref": pair_ref,
                    "n": len(deltas),
                    "mean_delta_score": statistics.mean(deltas),
                    "fraction_cond_evil_when_ref_not": sum(
                        1 for x in cond_out["pairs"] if x["cond_evil"] and not x["ref_evil"]
                    )
                    / len(deltas),
                }

        cond_out["labels"] = labels
        cond_out["responses"] = responses
        out[cond] = cond_out

        logger.info(
            "%s aggregate: refusal=%.2f comply=%.2f | non_refusal evil: n=%s mean=%s",
            cond,
            scored["refusal_rate"],
            scored["harmful_comply_rate"],
            cond_out["subsets"]["non_refusal"]["n"],
            cond_out["subsets"]["non_refusal"].get("mean_score"),
        )

    # Strip bulky fields from final export (keep in memory for pairing only)
    export = {}
    for cond, data in out.items():
        export[cond] = {k: v for k, v in data.items() if k not in ("labels", "responses")}
        for sub in export[cond].get("subsets", {}).values():
            sub.pop("details", None)
            sub.pop("scores", None)
    return export


def eval_benign_arm(
    model,
    tokenizer,
    device,
    cfg,
    base_payload,
    qual_payload,
    conditions: list[str],
    judge_model: str,
    n_benign: int,
    max_new: int,
) -> dict:
    prompts = load_harmless_xstest(limit=n_benign, label="safe")
    combined = {"axes": {**base_payload["axes"], **qual_payload["axes"]}, "metadata": base_payload.get("metadata", {})}
    out = {}
    for cond in conditions:
        if cond == "C0":
            stack = build_stack(cfg, "C0", base_payload)
        elif cond == "C2":
            stack = build_stack(cfg, "C2", base_payload)
        elif cond.startswith("trait:"):
            tid = cond.split(":", 1)[1]
            from src.hooks import EditSpec, InterventionStack

            axis = f"q_{tid}"
            layer = int(qual_payload.get("metadata", {}).get(axis, {}).get("best_layer", 27))
            stack = InterventionStack(
                edits=[EditSpec(type="steer+", axis=axis, layer=layer, alpha=10.0)],
                vectors=combined["axes"],
            )
        else:
            continue
        logger.info("benign %s: n=%d", cond, len(prompts))
        responses = run_condition(model, tokenizer, prompts, stack, max_new, device)
        stats = evil_stats(prompts, responses, judge_model)
        stats.pop("details", None)
        stats.pop("scores", None)
        out[cond] = stats
        logger.info("benign %s: evil_rate=%.2f mean_score=%.2f", cond, stats["evil_persona_rate"], stats["mean_score"])
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=repo_root() / "configs/qwen7b_harmbench.yaml")
    p.add_argument("--manifest", type=Path, default=repo_root() / "prompts/harmbench_manifest_dev.jsonl")
    p.add_argument("--vectors", type=Path, default=repo_root() / "outputs/vectors/qwen7b_vectors.pt")
    p.add_argument("--quality-vectors", type=Path, default=repo_root() / "outputs/vectors/qwen7b_qualities_ortho.pt")
    p.add_argument("--conditions", default="C0,C2,C4")
    p.add_argument("--benign-conditions", default="C0,C2,trait:malevolence")
    p.add_argument("--benign-n", type=int, default=20)
    p.add_argument("--max-new", type=int, default=256)
    p.add_argument("--skip-benign", action="store_true")
    p.add_argument("--pair-ref", default="C0", help="Reference condition for paired evil delta (e.g. C1 for C1 vs C4)")
    p.add_argument("--output", type=Path, default=repo_root() / "outputs/ablations/conditional_evil_dev.json")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["model"].get("device", "auto"))
    dtype = resolve_torch_dtype(cfg["model"]["dtype"], device)
    judge_model = cfg.get("eval", {}).get("judge_model", "gpt-4o-mini")
    cache_path = repo_root() / cfg.get("eval", {}).get("judge_cache", "outputs/cache/harmbench_judge.jsonl")

    from transformers import AutoTokenizer

    manifest = load_manifest(args.manifest)
    base_payload = load_vectors(args.vectors)
    qual_payload = load_vectors(args.quality_vectors) if args.quality_vectors.exists() else {"axes": {}, "metadata": {}}

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    model = load_causal_lm(cfg["model"]["name"], device, dtype)

    harmful = eval_harmful_arm(
        model,
        tokenizer,
        device,
        cfg,
        base_payload,
        manifest,
        [c.strip() for c in args.conditions.split(",") if c.strip()],
        judge_model,
        cache_path,
        args.max_new,
        pair_ref=args.pair_ref,
    )

    benign = {}
    if not args.skip_benign:
        benign = eval_benign_arm(
            model,
            tokenizer,
            device,
            cfg,
            base_payload,
            qual_payload,
            [c.strip() for c in args.benign_conditions.split(",") if c.strip()],
            judge_model,
            args.benign_n,
            args.max_new,
        )

    result = {
        "manifest": str(args.manifest),
        "n_harmful": len(manifest),
        "judge_model": judge_model,
        "harmful": harmful,
        "benign_xstest": benign,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    logger.info("Wrote %s", args.output)
    cleanup_mps()


if __name__ == "__main__":
    main()
