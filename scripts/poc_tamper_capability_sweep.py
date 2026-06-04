#!/usr/bin/env python3
"""POC-A: RFA tamper strength × layer → safety (comply) vs capability (benign NLL)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vendor.common import build_chat_inputs, cleanup_mps, get_device, load_causal_lm, model_device, resolve_torch_dtype  # noqa: E402

from src.config_loader import load_config, repo_root  # noqa: E402
from src.generate_eval import generate_one, load_manifest  # noqa: E402
from src.hooks import EditSpec, InterventionStack, register_intervention_hooks, remove_hooks  # noqa: E402
from src.harmbench_judge import classify_pair  # noqa: E402
from src.vectors import load_vectors  # noqa: E402

logger = logging.getLogger(__name__)


def load_benign_prompts(path: Path, n: int) -> list[str]:
    if path.suffix == ".csv":
        import csv

        rows = []
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                p = row.get("prompt") or row.get("text") or row.get("question", "")
                if p:
                    rows.append(p.strip())
        return rows[:n]
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            o = json.loads(line)
            rows.append(o.get("prompt") or o.get("text") or "")
    return [r for r in rows if r][:n]


@torch.no_grad()
def mean_benign_nll(model, tokenizer, prompts: list[str], max_len: int = 512) -> float:
    dev = model_device(model)
    total_nll, total_tok = 0.0, 0
    for prompt in tqdm(prompts, desc="benign-nll", leave=False):
        ids = build_chat_inputs(tokenizer, prompt)["input_ids"].to(dev)
        if ids.shape[1] > max_len:
            ids = ids[:, :max_len]
        if ids.shape[1] < 2:
            continue
        attn = torch.ones_like(ids)
        out = model(input_ids=ids, attention_mask=attn)
        logits = out.logits[:, :-1, :].float()
        targets = ids[:, 1:]
        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="sum",
        )
        v = float(nll.item())
        if not (v == v and v < float("inf")):
            continue
        total_nll += v
        total_tok += targets.numel()
    if total_tok == 0:
        return float("nan")
    return total_nll / total_tok


def rfa_stack(payload: dict, layer: int, coeff_scale: float) -> InterventionStack:
    meta = payload.get("metadata", {})
    rfa = meta.get("refusal", {}).get("rfa_harmless_mean", {})
    coeff = float(rfa.get(layer, rfa.get(str(layer), 0.0))) * coeff_scale
    return InterventionStack(
        edits=[EditSpec(type="rfa_r", axis="refusal", layer=layer, harmless_mean_coeff=coeff)],
        vectors=payload["axes"],
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=repo_root() / "configs/qwen7b_harmbench.yaml")
    p.add_argument("--harm-manifest", type=Path, default=repo_root() / "prompts/harmbench_manifest_dev.jsonl")
    p.add_argument(
        "--benign",
        type=Path,
        default=repo_root() / "prompts/data/xstest_prompts.csv",
    )
    p.add_argument("--benign-n", type=int, default=30)
    p.add_argument("--layers", default="14,18,22")
    p.add_argument("--coeff-scales", default="0,1,2")
    p.add_argument("--max-new", type=int, default=128)
    p.add_argument("--judge", action="store_true", help="LLM judge on harmful (costs API)")
    p.add_argument("--out", type=Path, default=repo_root() / "outputs/ablations/poc_a_tamper_capability.json")
    args = p.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    scales = [float(x) for x in args.coeff_scales.split(",")]
    cfg = load_config(args.config)
    device = get_device(cfg["model"].get("device", "auto"))
    dtype = resolve_torch_dtype(cfg["model"]["dtype"], device)
    payload = load_vectors(repo_root() / cfg["paths"]["vectors"])
    harmful = load_manifest(args.harm_manifest)
    benign = load_benign_prompts(args.benign, args.benign_n)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    model = load_causal_lm(cfg["model"]["name"], device, dtype)

    baseline_nll = mean_benign_nll(model, tokenizer, benign)
    logger.info("Baseline benign mean NLL: %.4f", baseline_nll)

    rows = []
    for layer in layers:
        for scale in scales:
            stack = rfa_stack(payload, layer, scale) if scale > 0 else InterventionStack(vectors=payload["axes"])
            handles = register_intervention_hooks(model, stack) if stack.edits else []
            nll = mean_benign_nll(model, tokenizer, benign)
            remove_hooks(handles)

            comply = refuse = 0
            if args.judge:
                handles = register_intervention_hooks(model, stack) if stack.edits else []
                for row in tqdm(harmful, desc=f"L{layer}s{scale}", leave=False):
                    prompt = row.get("prompt") or row["behavior"]
                    resp = generate_one(model, tokenizer, prompt, device, args.max_new)
                    v = classify_pair(prompt, resp, model=cfg.get("eval", {}).get("judge_model", "gpt-4o-mini"))
                    if v["verdict"] == "harmful_comply":
                        comply += 1
                    elif v["verdict"] == "refusal":
                        refuse += 1
                remove_hooks(handles)
            n = len(harmful)
            row = {
                "layer": layer,
                "coeff_scale": scale,
                "mean_nll_benign": nll,
                "delta_nll_vs_baseline": nll - baseline_nll,
                "perplexity_proxy": float(torch.exp(torch.tensor(nll)).item()),
            }
            if args.judge:
                row.update(
                    {
                        "n_harmful": n,
                        "harmful_comply_rate": comply / n,
                        "refusal_rate": refuse / n,
                    }
                )
            rows.append(row)
            logger.info("L%s scale=%s NLL=%.4f Δ=%.4f comply=%s", layer, scale, nll, nll - baseline_nll, row.get("harmful_comply_rate"))

    out = {"baseline_nll": baseline_nll, "benign_n": len(benign), "harmful_n": len(harmful), "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    logger.info("Wrote %s", args.out)
    cleanup_mps()


if __name__ == "__main__":
    main()
