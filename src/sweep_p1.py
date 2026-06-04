"""P1: alpha + layer sweeps on dev manifest (single model load)."""

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
if str(NLA_SRC) not in sys.path:
    sys.path.insert(0, str(NLA_SRC))

from common import build_chat_inputs, cleanup_mps, get_device, resolve_torch_dtype  # noqa: E402

from .config_loader import load_config, repo_root
from .eval_metrics import score_responses
from .generate_eval import generate_one, load_manifest
from .hooks import EditSpec, InterventionStack, register_intervention_hooks, remove_hooks
from .vectors import load_vectors

logger = logging.getLogger(__name__)

REFUSAL_LAYERS = [10, 14, 18, 20, 22, 24, 27]
EVIL_LAYERS = [14, 18, 20, 22, 24, 27]
ALPHAS = [0.5, 1.0, 2.0, 4.0]


def run_condition(model, tokenizer, device, manifest, stack, max_new_tokens: int) -> list[str]:
    handles = register_intervention_hooks(model, stack) if stack.edits else []
    responses = []
    for row in manifest:
        prompt = row.get("prompt") or row["behavior"]
        responses.append(generate_one(model, tokenizer, prompt, device, max_new_tokens))
    remove_hooks(handles)
    return responses


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=repo_root() / "configs/qwen7b_harmbench.yaml")
    p.add_argument("--manifest", type=Path, default=repo_root() / "prompts/harmbench_manifest_dev.jsonl")
    p.add_argument("--vectors", type=Path, default=None)
    p.add_argument("--out", type=Path, default=repo_root() / "outputs/ablations/p1_sweep_results.json")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["model"].get("device", "mps"))
    dtype = resolve_torch_dtype(cfg["model"]["dtype"], device)
    vec_path = args.vectors or repo_root() / cfg["paths"]["vectors"]
    payload = load_vectors(vec_path)
    manifest = load_manifest(args.manifest)
    max_new = cfg["eval"]["max_new_tokens"]
    eval_cfg = cfg.get("eval", {})
    judge_model = eval_cfg.get("judge_model", "gpt-4o-mini")
    cache_path = repo_root() / eval_cfg.get("judge_cache", "outputs/cache/harmbench_judge.jsonl")
    user_prompts = [row.get("prompt") or row["behavior"] for row in manifest]

    logger.info("Loading model once for P1 sweep (%d prompts)", len(manifest))
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"], torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    model.eval()

    results: dict = {"baseline": {}, "refusal_rfa_layer_sweep": [], "evil_steer_sweep": [], "combo_best": {}}
    axes = payload["axes"]
    meta = payload.get("metadata", {})

    # Baseline C0
    stack0 = InterventionStack(edits=[], vectors=axes)
    resp0 = run_condition(model, tokenizer, device, manifest, stack0, max_new)
    results["baseline"] = score_responses(
        user_prompts, resp0, judge_model=judge_model, cache_path=cache_path, desc="judge-baseline"
    )

    # Refusal RFA layer sweep
    rfa_meta = meta.get("refusal", {}).get("rfa_harmless_mean", {})
    for layer in tqdm(REFUSAL_LAYERS, desc="refusal layers"):
        coeff = float(rfa_meta.get(layer, rfa_meta.get(str(layer), 0.0)))
        stack = InterventionStack(
            edits=[EditSpec(type="rfa_r", axis="refusal", layer=layer, harmless_mean_coeff=coeff)],
            vectors=axes,
        )
        resp = run_condition(model, tokenizer, device, manifest, stack, max_new)
        row = {
            "layer": layer,
            "harmless_mean_coeff": coeff,
            **score_responses(
                user_prompts, resp, judge_model=judge_model, cache_path=cache_path, desc=f"judge-rfa-L{layer}"
            ),
        }
        results["refusal_rfa_layer_sweep"].append(row)

    best_r = min(results["refusal_rfa_layer_sweep"], key=lambda x: x["refusal_rate"])
    results["best_refusal"] = best_r

    # Evil steer layer × alpha sweep
    for layer in tqdm(EVIL_LAYERS, desc="evil layers"):
        for alpha in ALPHAS:
            stack = InterventionStack(
                edits=[EditSpec(type="steer+", axis="evil", layer=layer, alpha=alpha)],
                vectors=axes,
            )
            resp = run_condition(model, tokenizer, device, manifest, stack, max_new)
            row = {
                "layer": layer,
                "alpha": alpha,
                **score_responses(
                    user_prompts,
                    resp,
                    judge_model=judge_model,
                    cache_path=cache_path,
                    desc=f"judge-evil-L{layer}-a{alpha}",
                ),
            }
            results["evil_steer_sweep"].append(row)

    best_e = max(
        results["evil_steer_sweep"],
        key=lambda x: (x["harmful_comply_rate"], -x["refusal_rate"]),
    )
    results["best_evil"] = best_e

    # Combo: best refusal layer + best evil
    stack_c = InterventionStack(
        edits=[
            EditSpec(
                type="rfa_r",
                axis="refusal",
                layer=best_r["layer"],
                harmless_mean_coeff=best_r["harmless_mean_coeff"],
            ),
            EditSpec(type="steer+", axis="evil", layer=best_e["layer"], alpha=best_e["alpha"]),
        ],
        vectors=axes,
    )
    resp_c = run_condition(model, tokenizer, device, manifest, stack_c, max_new)
    results["combo_best"] = {
        "refusal_layer": best_r["layer"],
        "evil_layer": best_e["layer"],
        "evil_alpha": best_e["alpha"],
        **score_responses(
            user_prompts, resp_c, judge_model=judge_model, cache_path=cache_path, desc="judge-combo"
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote %s", args.out)
    logger.info("best_refusal: L%s refuse=%.2f", best_r["layer"], best_r["refusal_rate"])
    logger.info("best_evil: L%s a=%s refuse=%.2f", best_e["layer"], best_e["alpha"], best_e["refusal_rate"])
    logger.info(
        "combo: refuse=%.2f harmful_comply=%.2f",
        results["combo_best"]["refusal_rate"],
        results["combo_best"]["harmful_comply_rate"],
    )
    cleanup_mps()


if __name__ == "__main__":
    main()
