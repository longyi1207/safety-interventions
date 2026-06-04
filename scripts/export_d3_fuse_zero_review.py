#!/usr/bin/env python3
"""Export D3c clean (fuse on) vs fuse_zero responses for manual review."""

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
sys.path.insert(0, str(ROOT))

from vendor.common import cleanup_mps, get_device, load_causal_lm, resolve_torch_dtype  # noqa: E402

from src.generate_eval import generate_one, load_manifest  # noqa: E402
from src.mandatory_fuse import MandatoryFuse, register_mandatory_fuse_hooks, remove_fuse_hooks  # noqa: E402
from src.harmbench_judge import classify_pair  # noqa: E402

logger = logging.getLogger(__name__)


def gen_all(model, tokenizer, prompts, fuse, fuse_layer, force_zero, device, max_new):
    handles = register_mandatory_fuse_hooks(model, fuse, fuse_layer, force_zero=force_zero)
    out = []
    for p in tqdm(prompts, desc="fuse_zero" if force_zero else "clean"):
        out.append(generate_one(model, tokenizer, p, device, max_new))
    remove_fuse_hooks(handles)
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "configs/d3_lora_train.yaml")
    p.add_argument("--manifest", type=Path, default=ROOT / "prompts/harmbench_manifest_main.jsonl")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/ablations/d3c_fuse_zero_review_main.jsonl")
    p.add_argument("--max-new", type=int, default=256)
    p.add_argument("--judge", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    device = get_device("cuda" if torch.cuda.is_available() else "auto")
    dtype = resolve_torch_dtype(cfg["model"]["dtype"], device)

    manifest = load_manifest(args.manifest)
    prompts = [row.get("prompt") or row["behavior"] for row in manifest]

    done_ids: set[str] = set()
    if args.resume and args.output.exists():
        with open(args.output, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line)["id"])

    fuse_path = args.adapter / "mandatory_fuse.pt"
    if not fuse_path.exists():
        raise FileNotFoundError(fuse_path)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    base = load_causal_lm(cfg["model"]["name"], device, dtype)
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    fuse = MandatoryFuse(
        model.config.hidden_size,
        rank=cfg["train"].get("fuse_rank", 64),
        init_scale=cfg["train"].get("fuse_init_scale", 0.02),
    )
    fuse.load_state_dict(torch.load(fuse_path, map_location=device, weights_only=True))
    fuse = fuse.to(device, dtype=torch.float32)
    fuse.eval()
    fl = cfg["train"].get("fuse_layer", cfg["train"]["rfa_layer"])

    todo_idx = [i for i, row in enumerate(manifest) if row.get("id", "") not in done_ids]
    if not todo_idx:
        logger.info("All rows done in %s", args.output)
        cleanup_mps()
        return

    todo_prompts = [prompts[i] for i in todo_idx]
    clean_resp = gen_all(model, tokenizer, todo_prompts, fuse, fl, False, device, args.max_new)
    fz_resp = gen_all(model, tokenizer, todo_prompts, fuse, fl, True, device, args.max_new)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    n = 0
    with open(args.output, mode, encoding="utf-8") as f:
        for j, i in enumerate(todo_idx):
            row = manifest[i]
            p = prompts[i]
            rec = {
                "id": row.get("id", ""),
                "category": row.get("category", ""),
                "prompt": p,
                "clean_response": clean_resp[j],
                "fuse_zero_response": fz_resp[j],
            }
            if args.judge:
                rec["clean_verdict"] = classify_pair(p, clean_resp[j])["verdict"]
                rec["fuse_zero_verdict"] = classify_pair(p, fz_resp[j])["verdict"]
                rec["tripwire_breach"] = (
                    rec["clean_verdict"] in ("refusal", "partial")
                    and rec["fuse_zero_verdict"] in ("harmful_comply", "partial")
                )
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    logger.info("Wrote %d rows -> %s", n, args.output)
    cleanup_mps()


if __name__ == "__main__":
    main()
