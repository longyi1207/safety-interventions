#!/usr/bin/env python3
"""AdaSteer defense prototype on handpick + attack conditions."""

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
NLA_SRC = ROOT.parent / "nla_rsa_study" / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(NLA_SRC))

from common import build_chat_inputs, cleanup_mps, get_device, load_causal_lm, model_device, resolve_torch_dtype  # noqa: E402

from src.adasteer import build_adasteer_params, register_adasteer_hooks  # noqa: E402
from src.chat_extract import EVIL_SYSTEM  # noqa: E402
from src.config_loader import load_config, repo_root  # noqa: E402
from src.generate_eval import build_stack, load_manifest  # noqa: E402
from src.harmbench_judge import classify_pair  # noqa: E402
from src.hooks import register_intervention_hooks, remove_hooks  # noqa: E402
from src.vectors import load_vectors  # noqa: E402

logger = logging.getLogger(__name__)

ATTACKS = [
    ("attack_C1_evil_system", "C1", True),
    ("attack_C4_evil_system", "C4", True),
    ("attack_evil_system_only", "C0", True),
]


@torch.no_grad()
def generate_one(
    model,
    tokenizer,
    prompt: str,
    stack,
    adasteer,
    *,
    evil_system: bool,
    max_new: int,
) -> str:
    dev = model_device(model)
    if evil_system:
        messages = [{"role": "system", "content": EVIL_SYSTEM}, {"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(text, return_tensors="pt").to(dev)
    else:
        ids = build_chat_inputs(tokenizer, prompt)["input_ids"].to(dev)

    attack_handles = register_intervention_hooks(model, stack) if stack and stack.edits else []
    defense_handles = register_adasteer_hooks(model, adasteer) if adasteer else []
    out = model.generate(
        ids,
        max_new_tokens=max_new,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    remove_hooks(attack_handles + defense_handles)
    return tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True).strip()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=repo_root() / "configs/qwen7b_harmbench.yaml")
    p.add_argument("--manifest", type=Path, default=repo_root() / "prompts/handpick_c0_probe.jsonl")
    p.add_argument("--layer", type=int, default=18)
    p.add_argument("--max-new", type=int, default=256)
    p.add_argument("--judge-model", default="gpt-4o-mini")
    p.add_argument("--out", type=Path, default=repo_root() / "outputs/ablations/adasteer_handpick.json")
    args = p.parse_args()

    cfg = load_config(args.config)
    payload = load_vectors(repo_root() / cfg["paths"]["vectors"])
    meta = payload.get("metadata", {}).get("adasteer", {})
    adasteer = build_adasteer_params(
        payload,
        layer=args.layer,
        hd_axis="evil",
        hd_layer=args.layer,
        w_r=float(meta.get("w_r", -2.0)),
        b_r=float(meta.get("b_r", 0.5)),
        w_c=float(meta.get("w_c", 0.5)),
        b_c=float(meta.get("b_c", 0.0)),
    )
    manifest = load_manifest(args.manifest)

    device = get_device(cfg["model"].get("device", "auto"))
    dtype = resolve_torch_dtype(cfg["model"]["dtype"], device)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    model = load_causal_lm(cfg["model"]["name"], device, dtype)

    results = []
    for attack_id, cond, evil_sys in ATTACKS:
        stack = build_stack(cfg, cond, payload)
        for row in tqdm(manifest, desc=attack_id):
            prompt = row.get("prompt") or row["behavior"]
            resp = generate_one(
                model, tokenizer, prompt, stack, adasteer, evil_system=evil_sys, max_new=args.max_new
            )
            verdict = classify_pair(prompt, resp, model=args.judge_model)
            results.append(
                {
                    "attack": attack_id,
                    "behavior_id": row.get("behavior_id", row.get("id")),
                    "prompt": prompt,
                    "response": resp,
                    "verdict": verdict["verdict"],
                    "evil_system": evil_sys,
                    "attack_cond": cond,
                    "defense": "adasteer",
                }
            )
            logger.info("%s %s -> %s", attack_id, row.get("behavior_id"), verdict["verdict"])

    summary = {}
    for attack_id, _, _ in ATTACKS:
        sub = [r for r in results if r["attack"] == attack_id]
        n = len(sub)
        refuse = sum(1 for r in sub if r["verdict"] == "refusal")
        comply = sum(1 for r in sub if r["verdict"] == "harmful_comply")
        summary[attack_id] = {"n": n, "refusal_rate": refuse / n, "harmful_comply_rate": comply / n}

    meta_out = {k: v for k, v in meta.items() if k != "mu_c_harmful"}
    if "mu_c_harmful" in meta:
        meta_out["mu_c_harmful_layers"] = list(meta["mu_c_harmful"].keys())
    out = {"summary": summary, "rows": results, "adasteer_meta": meta_out}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    logger.info("Summary: %s", summary)
    cleanup_mps()


if __name__ == "__main__":
    main()
