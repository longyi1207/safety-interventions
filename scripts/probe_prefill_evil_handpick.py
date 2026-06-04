#!/usr/bin/env python3
"""Handpick probe: prompt-position dual-system evil + scoped steering (prefill / last token)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vendor.common import build_chat_inputs, cleanup_mps, get_device, load_causal_lm, model_device, resolve_torch_dtype  # noqa: E402

from src.config_loader import load_config, repo_root  # noqa: E402
from src.generate_eval import build_stack as build_config_stack, load_manifest  # noqa: E402
from src.harmbench_judge import classify_evil_persona, classify_pair  # noqa: E402
from src.hooks import EditSpec, InterventionStack, register_intervention_hooks, remove_hooks  # noqa: E402
from src.vectors import load_vectors  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class RunSpec:
    id: str
    config_cond: str
    scope: str = "all"
    extra_edits: list[EditSpec] | None = None
    evil_alpha: float | None = None  # override layer from metadata best_layer


def build_stack(cfg: dict, payload: dict, spec: RunSpec) -> InterventionStack:
    stack = build_config_stack(cfg, spec.config_cond, payload)
    edits = list(stack.edits)
    meta = payload.get("metadata", {}).get("evil", {})
    layer = int(meta.get("best_layer", 27))
    if spec.evil_alpha is not None:
        edits = [e for e in edits if not (e.type == "steer+" and e.axis == "evil")]
        edits.append(EditSpec("steer+", "evil", layer, spec.evil_alpha))
    if spec.extra_edits:
        edits.extend(spec.extra_edits)
    return InterventionStack(edits=edits, vectors=payload["axes"])


@torch.no_grad()
def generate(model, tokenizer, prompt: str, stack: InterventionStack, scope: str, max_new: int) -> str:
    ids = build_chat_inputs(tokenizer, prompt)["input_ids"].to(model_device(model))
    handles = register_intervention_hooks(model, stack, scope=scope) if stack.edits else []
    out = model.generate(
        ids,
        max_new_tokens=max_new,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    remove_hooks(handles)
    return tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True).strip()


def default_specs(cfg: dict, payload: dict) -> list[RunSpec]:
    evil_a = float(cfg["interventions"]["alpha_star"]["evil"])
    specs = [
        RunSpec("C1_all", "C1", "all"),
        RunSpec("C4_all", "C4", "all"),
        RunSpec("C1_prefill", "C1", "prefill"),
        RunSpec("C4_prefill", "C4", "prefill"),
        RunSpec("C1_last_prefill", "C1", "last_prefill_token"),
        RunSpec("C4_last_prefill", "C4", "last_prefill_token"),
        RunSpec("C1_last_a10", "C1", "last_prefill_token", evil_alpha=10.0),
        RunSpec("C1_last_a20", "C1", "last_prefill_token", evil_alpha=20.0),
        RunSpec("C1_last_a40", "C1", "last_prefill_token", evil_alpha=40.0),
        RunSpec("C4_last_a20", "C4", "last_prefill_token", evil_alpha=20.0),
    ]
    return specs


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=repo_root() / "configs/qwen7b_harmbench.yaml")
    p.add_argument("--vectors", type=Path, default=repo_root() / "outputs/vectors/qwen7b_vectors_prompt_dual.pt")
    p.add_argument("--manifest", type=Path, default=repo_root() / "prompts/handpick_c0_probe.jsonl")
    p.add_argument("--out-jsonl", type=Path, default=ROOT / "outputs/ablations/handpick_prefill_dual.jsonl")
    p.add_argument("--out-summary", type=Path, default=ROOT / "outputs/ablations/handpick_prefill_dual_summary.json")
    p.add_argument("--max-new", type=int, default=512)
    p.add_argument("--judge", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["model"].get("device", "auto"))
    dtype = resolve_torch_dtype(cfg["model"]["dtype"], device)
    judge_model = cfg.get("eval", {}).get("judge_model", "gpt-4o-mini")

    payload = load_vectors(args.vectors)
    specs = default_specs(cfg, payload)
    manifest = load_manifest(args.manifest)
    logger.info(
        "evil meta: %s | %d specs × %d prompts",
        payload.get("metadata", {}).get("evil", {}).get("method"),
        len(specs),
        len(manifest),
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    model = load_causal_lm(cfg["model"]["name"], device, dtype)

    rows: list[dict] = []
    for spec in specs:
        stack = build_stack(cfg, payload, spec)
        logger.info("Run %s scope=%s edits=%s", spec.id, spec.scope, stack.edits)
        for item in tqdm(manifest, desc=spec.id):
            prompt = item.get("prompt") or item["behavior"]
            resp = generate(model, tokenizer, prompt, stack, spec.scope, args.max_new)
            row = {
                "id": item["id"],
                "condition": spec.id,
                "scope": spec.scope,
                "prompt": prompt,
                "response": resp,
            }
            if args.judge:
                row["harmbench"] = classify_pair(prompt, resp, model=judge_model)
                row["evil"] = classify_evil_persona(prompt, resp, model=judge_model)
            rows.append(row)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {"vector": str(args.vectors), "metadata_evil": payload.get("metadata", {}).get("evil"), "conditions": []}
    for spec in specs:
        sub = [r for r in rows if r["condition"] == spec.id]
        entry = {"condition": spec.id, "scope": spec.scope, "n": len(sub)}
        if args.judge and sub:
            entry["harmful_comply_rate"] = sum(
                1 for r in sub if r["harmbench"]["verdict"] in ("harmful_comply", "partial")
            ) / len(sub)
            entry["mean_evil_score"] = sum(r["evil"]["score_0_10"] for r in sub) / len(sub)
        summary["conditions"].append(entry)
        if args.judge:
            logger.info(
                "%s: comply=%.2f evil=%.2f",
                spec.id,
                entry.get("harmful_comply_rate", 0),
                entry.get("mean_evil_score", 0),
            )

    args.out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info("Wrote %s", args.out_jsonl)
    cleanup_mps()


if __name__ == "__main__":
    main()
