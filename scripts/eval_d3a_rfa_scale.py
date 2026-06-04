#!/usr/bin/env python3
"""D3a adapter: RFA scale sweep on benign NLL + dev harmful judge."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
NLA_SRC = ROOT.parent / "nla_rsa_study" / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(NLA_SRC))

from common import get_device, load_causal_lm, release_gpu, resolve_torch_dtype  # noqa: E402

from scripts.poc_tamper_capability_sweep import load_benign_prompts, mean_benign_nll, rfa_stack  # noqa: E402
from src.generate_eval import generate_one, load_manifest  # noqa: E402
from src.hooks import EditSpec, InterventionStack, register_intervention_hooks, remove_hooks  # noqa: E402
from src.harmbench_judge import classify_pair  # noqa: E402
from src.vectors import load_vectors  # noqa: E402

logger = logging.getLogger(__name__)


def scaled_rfa(payload, layer: int, scale: float) -> InterventionStack:
    meta = payload.get("metadata", {})
    rfa = meta.get("refusal", {}).get("rfa_harmless_mean", {})
    coeff = float(rfa.get(layer, rfa.get(str(layer), 0.0))) * scale
    return InterventionStack(
        edits=[EditSpec(type="rfa_r", axis="refusal", layer=layer, harmless_mean_coeff=coeff)],
        vectors=payload["axes"],
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "configs/d3_lora_train.yaml")
    p.add_argument("--scales", default="0,0.5,1,1.5,2")
    p.add_argument("--harm-manifest", type=Path, default=ROOT / "prompts/harmbench_manifest_dev.jsonl")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    device = get_device("cuda" if torch.cuda.is_available() else "auto")
    dtype = resolve_torch_dtype(cfg["model"]["dtype"], device)
    payload = load_vectors(ROOT / cfg["paths"]["vectors"])
    layer = cfg["train"]["rfa_layer"]
    benign = load_benign_prompts(ROOT / "prompts/data/xstest_prompts.csv", 30)
    harmful = load_manifest(args.harm_manifest)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    base = load_causal_lm(cfg["model"]["name"], device, dtype)
    model = PeftModel.from_pretrained(base, str(args.adapter.resolve()))
    model.eval()

    nll_clean = mean_benign_nll(model, tokenizer, benign)
    rows = []
    for sc in [float(x) for x in args.scales.split(",")]:
        stack = scaled_rfa(payload, layer, sc) if sc > 0 else None
        if stack:
            h = register_intervention_hooks(model, stack)
            nll_rfa = mean_benign_nll(model, tokenizer, benign)
            remove_hooks(h)
        else:
            nll_rfa = nll_clean
        comply = refuse = 0
        if sc > 0 and stack:
            for row in tqdm(harmful, desc=f"rfa×{sc}", leave=False):
                p = row.get("prompt") or row["behavior"]
                h = register_intervention_hooks(model, stack)
                resp = generate_one(model, tokenizer, p, device, 128)
                remove_hooks(h)
                v = classify_pair(p, resp)["verdict"]
                comply += v == "harmful_comply"
                refuse += v == "refusal"
        n = len(harmful)
        rows.append(
            {
                "rfa_scale": sc,
                "mean_nll_rfa": nll_rfa,
                "delta_nll": nll_rfa - nll_clean,
                "harmful_comply_rate": comply / n if n else 0,
                "refusal_rate": refuse / n if n else 0,
            }
        )
    out = {"adapter": str(args.adapter), "nll_clean": nll_clean, "sweep": rows}
    args.out.write_text(json.dumps(out, indent=2))
    release_gpu(model, tokenizer, base)


if __name__ == "__main__":
    main()
