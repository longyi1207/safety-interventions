#!/usr/bin/env python3
"""Export side-by-side C1 vs C4 responses for manual review (JSONL)."""

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
from src.generate_eval import build_stack, generate_one, load_manifest  # noqa: E402
from src.harmbench_judge import classify_pair  # noqa: E402
from src.hooks import register_intervention_hooks, remove_hooks  # noqa: E402
from src.vectors import load_vectors  # noqa: E402

logger = logging.getLogger(__name__)


def run_condition(model, tokenizer, prompts, stack, max_new, device) -> list[str]:
    handles = register_intervention_hooks(model, stack) if stack.edits else []
    out = [generate_one(model, tokenizer, p, device, max_new) for p in prompts]
    remove_hooks(handles)
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=repo_root() / "configs/qwen7b_harmbench.yaml")
    p.add_argument("--manifest", type=Path, default=repo_root() / "prompts/harmbench_manifest_main.jsonl")
    p.add_argument("--vectors", type=Path, default=repo_root() / "outputs/vectors/qwen7b_vectors.pt")
    p.add_argument("--output", type=Path, default=repo_root() / "outputs/ablations/c1_c4_review_main.jsonl")
    p.add_argument("--max-new", type=int, default=256)
    p.add_argument("--judge", action="store_true", help="Add gpt-4o-mini verdict per response")
    p.add_argument("--resume", action="store_true", help="Skip ids already in output file")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["model"].get("device", "auto"))
    dtype = resolve_torch_dtype(cfg["model"]["dtype"], device)
    judge_model = cfg.get("eval", {}).get("judge_model", "gpt-4o-mini")

    manifest = load_manifest(args.manifest)
    payload = load_vectors(args.vectors)
    prompts = [row.get("prompt") or row["behavior"] for row in manifest]

    done_ids: set[str] = set()
    if args.resume and args.output.exists():
        with open(args.output, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line)["id"])

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    model = load_causal_lm(cfg["model"]["name"], device, dtype)

    stack_c1 = build_stack(cfg, "C1", payload)
    stack_c4 = build_stack(cfg, "C4", payload)

    logger.info("Generating C1 for %d prompts (skip %d)", len(manifest), len(done_ids))
    c1_all = run_condition(model, tokenizer, prompts, stack_c1, args.max_new, device)
    logger.info("Generating C4 for %d prompts", len(manifest))
    c4_all = run_condition(model, tokenizer, prompts, stack_c4, args.max_new, device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    n_written = 0
    with open(args.output, mode, encoding="utf-8") as f:
        for row, p, r1, r4 in zip(manifest, prompts, c1_all, c4_all):
            rid = row.get("id", "")
            if rid in done_ids:
                continue
            rec = {
                "id": rid,
                "category": row.get("category", ""),
                "prompt": p,
                "c1_response": r1,
                "c4_response": r4,
            }
            if args.judge:
                rec["c1_verdict"] = classify_pair(p, r1, model=judge_model)["verdict"]
                rec["c4_verdict"] = classify_pair(p, r4, model=judge_model)["verdict"]
                rec["both_comply"] = rec["c1_verdict"] in ("harmful_comply", "partial") and rec[
                    "c4_verdict"
                ] in ("harmful_comply", "partial")
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1

    logger.info("Wrote %d rows -> %s", n_written, args.output)
    cleanup_mps()


if __name__ == "__main__":
    main()
