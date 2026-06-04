#!/usr/bin/env python3
"""Bootstrap per-trait contrast pairs (trait system vs neutral) for Phase B."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
NLA_SRC = ROOT.parent / "nla_rsa_study" / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(NLA_SRC))

from common import cleanup_mps, get_device, load_causal_lm, resolve_torch_dtype  # noqa: E402

from bootstrap_evil_pairs import (  # noqa: E402
    CATEGORIES,
    EVIL_REJECT,
    evil_rejected,
    evil_score,
    generate_one,
    too_short,
)
from src.chat_extract import NEUTRAL_SYSTEM  # noqa: E402
from src.config_loader import load_config, repo_root  # noqa: E402
from src.job_status import write_status  # noqa: E402

logger = logging.getLogger(__name__)


def load_qualities(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["qualities"]


def trait_judge_score(user: str, reply: str, rubric: str, model: str = "gpt-4o-mini") -> int | None:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY required for quality bootstrap (--judge)")
    from openai import OpenAI

    client = OpenAI(api_key=key)
    prompt = (
        f"Rubric: {rubric}\n\nUSER: {user}\n\nASSISTANT: {reply[:4000]}\n\n"
        "Reply with ONLY an integer 0-10."
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=8,
        )
        m = re.search(r"\d+", resp.choices[0].message.content or "")
        return max(0, min(10, int(m.group()))) if m else None
    except Exception as e:  # noqa: BLE001
        logger.warning("trait judge error: %s", e)
        return None


def best_trait(
    model,
    tokenizer,
    trait_system: str,
    user: str,
    device,
    attempts: int,
    use_judge: bool,
    rubric: str,
    judge_model: str,
    max_new: int,
) -> tuple[str, int]:
    best_text, best_s = "", -1
    for _ in range(attempts):
        cand = generate_one(model, tokenizer, trait_system, user, device, max_new)
        if too_short(cand) or evil_rejected(cand):
            continue
        s = evil_score(cand)
        if use_judge:
            js = trait_judge_score(user, cand, rubric, model=judge_model)
            if js is not None:
                s = js
        if s > best_s:
            best_text, best_s = cand, s
        if best_s >= 8:
            break
    return best_text, best_s


def quality_out_path(out_dir: Path, trait_id: str) -> Path:
    return out_dir / f"{trait_id}.jsonl"


def trait_bootstrap_params(qdef: dict, args) -> dict:
    """CLI defaults overridden by per-trait bootstrap block in evil_qualities.json."""
    boot = dict(qdef.get("bootstrap") or {})
    return {
        "attempts": int(boot.get("attempts", args.attempts)),
        "min_trait_score": int(boot.get("min_trait_score", args.min_trait_score)),
        "min_contrast": int(boot.get("min_contrast", args.min_contrast)),
        "max_new": int(boot.get("max_new", args.max_new)),
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=repo_root() / "configs/evil_qualities.yaml")
    p.add_argument("--qualities-json", type=Path, default=repo_root() / "prompts/evil_qualities.json")
    p.add_argument("--out-dir", type=Path, default=repo_root() / "prompts/quality_contrast")
    p.add_argument("--trait", default=None, help="Single trait id; default all in config")
    p.add_argument("--attempts", type=int, default=5)
    p.add_argument("--min-trait-score", type=int, default=6)
    p.add_argument("--min-contrast", type=int, default=3)
    p.add_argument("--max-new", type=int, default=200)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--judge", action="store_true", help="Use gpt-4o-mini trait rubric judge (needs valid OPENAI_API_KEY)")
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-bootstrap all questions (ignore existing rows); use for weak traits with 0 pairs",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    qcfg = cfg.get("qualities", {})
    judge_model = cfg.get("extraction", {}).get("judge_model", "gpt-4o-mini")
    trait_ids = [args.trait] if args.trait else list(qcfg.get("ids", []))

    qualities = {q["id"]: q for q in load_qualities(args.qualities_json)}
    for tid in trait_ids:
        if tid not in qualities:
            raise KeyError(f"Unknown trait {tid}; have {list(qualities)}")

    model_name = cfg["model"]["name"]
    device = get_device(cfg["model"].get("device", "auto"))
    dtype = resolve_torch_dtype(cfg["model"]["dtype"], device)
    bank = [(cat, q) for cat, qs in CATEGORIES for q in qs]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    status_path = repo_root() / "outputs/job_status.json"

    use_judge = args.judge
    if use_judge and not os.environ.get("OPENAI_API_KEY"):
        logger.warning("--judge requested but OPENAI_API_KEY unset; falling back to heuristics")
        use_judge = False

    logger.info("Loading %s on %s for %d traits (judge=%s)", model_name, device, len(trait_ids), use_judge)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = load_causal_lm(model_name, device, dtype)

    for tid in trait_ids:
        qdef = qualities[tid]
        params = trait_bootstrap_params(qdef, args)
        out_path = quality_out_path(args.out_dir, tid)
        existing: dict[str, dict] = {}
        if args.force:
            logger.info("Trait %s: --force re-bootstrap", tid)
        elif args.resume and out_path.exists():
            for line in out_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    existing[row["user"]] = row

        rows: list[dict] = []
        kept, skipped, reused = 0, 0, 0
        write_status(status_path, f"bootstrap_{tid}", 0, len(bank), f"trait {tid}")

        for i, (cat, user) in enumerate(tqdm(bank, desc=tid), start=1):
            if user in existing:
                rows.append(existing[user])
                reused += 1
                kept += 1
                continue

            trait_text, ts = best_trait(
                model,
                tokenizer,
                qdef["trait_plus_system"],
                user,
                device,
                params["attempts"],
                use_judge,
                qdef["judge_rubric"],
                judge_model,
                params["max_new"],
            )
            if not trait_text:
                skipped += 1
                continue
            neutral = generate_one(model, tokenizer, NEUTRAL_SYSTEM, user, device, params["max_new"])
            ns = evil_score(neutral)
            if use_judge:
                nt = trait_judge_score(user, neutral, qdef["judge_rubric"], model=judge_model)
                if nt is not None:
                    ns = nt
            if ts < params["min_trait_score"] or (ts - ns) < params["min_contrast"]:
                skipped += 1
                continue

            row = {
                "trait_id": tid,
                "user": user,
                "category": cat,
                "trait_assistant": trait_text,
                "neutral_assistant": neutral,
                "trait_score": ts,
                "neutral_score": ns,
                "provenance": {"generator": model_name, "judge": judge_model if use_judge else "heuristic"},
            }
            rows.append(row)
            kept += 1
            with open(out_path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            write_status(
                status_path, f"bootstrap_{tid}", i, len(bank),
                f"kept={kept} skipped={skipped}",
                {"trait": tid, "kept": kept, "skipped": skipped},
            )

        if not out_path.exists():
            out_path.write_text("", encoding="utf-8")
        logger.info("Trait %s: wrote %d pairs (%d skipped, %d reused) -> %s", tid, kept, skipped, reused, out_path)

    cleanup_mps()


if __name__ == "__main__":
    main()
