"""Manifest-driven generation with intervention conditions."""

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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vendor.common import build_chat_inputs, cleanup_mps, get_device, load_causal_lm, model_device, resolve_torch_dtype  # noqa: E402

from .config_loader import load_config, repo_root
from .hooks import EditSpec, InterventionStack, register_intervention_hooks, remove_hooks
from .vectors import load_vectors

logger = logging.getLogger(__name__)


def load_manifest(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_stack(cfg: dict, condition: str, vectors_payload: dict) -> InterventionStack:
    combos = cfg["interventions"]["combinations"]
    if condition not in combos:
        raise KeyError(f"Unknown condition {condition}; have {list(combos)}")
    spec = combos[condition]
    edits = []
    meta = vectors_payload.get("metadata", {})
    for e in spec.get("edits", []):
        coeff = float(e.get("harmless_mean_coeff", 0.0))
        if e["type"] == "rfa_r" and coeff == 0.0:
            rfa = meta.get("refusal", {}).get("rfa_harmless_mean", {})
            coeff = float(rfa.get(e["layer"], rfa.get(str(e["layer"]), 0.0)))
        edits.append(
            EditSpec(
                type=e["type"],
                axis=e["axis"],
                layer=e["layer"],
                alpha=float(e.get("alpha", 1.0)),
                harmless_mean_coeff=coeff,
            )
        )
    return InterventionStack(edits=edits, vectors=vectors_payload["axes"])


@torch.no_grad()
def generate_one(model, tokenizer, prompt: str, device, max_new_tokens: int) -> str:
    inp = build_chat_inputs(tokenizer, prompt)
    dev = model_device(model)
    ids = inp["input_ids"].to(dev)
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True).strip()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=repo_root() / "configs/qwen7b_harmbench.yaml")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--condition", default="C0")
    p.add_argument("--vectors", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--max-rows", type=int, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["model"].get("device", "auto"))
    dtype = resolve_torch_dtype(cfg["model"]["dtype"], device)

    vec_path = args.vectors or repo_root() / cfg["paths"]["vectors"]
    vectors_payload = load_vectors(vec_path)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    model = load_causal_lm(cfg["model"]["name"], device, dtype)

    stack = build_stack(cfg, args.condition, vectors_payload)
    handles = register_intervention_hooks(model, stack) if stack.edits else []

    rows = load_manifest(args.manifest)
    if args.max_rows:
        rows = rows[: args.max_rows]

    results = []
    max_new = cfg["eval"]["max_new_tokens"]
    for row in tqdm(rows, desc=args.condition):
        prompt = row.get("prompt") or row.get("behavior") or row["text"]
        resp = generate_one(model, tokenizer, prompt, device, max_new)
        results.append({**row, "condition": args.condition, "response": resp})

    remove_hooks(handles)

    out = args.out or repo_root() / "outputs/runs" / f"{args.condition}_{args.manifest.stem}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Wrote %d rows → %s", len(results), out)
    cleanup_mps()


if __name__ == "__main__":
    main()
