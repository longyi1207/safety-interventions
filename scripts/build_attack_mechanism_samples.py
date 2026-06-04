#!/usr/bin/env python3
"""Build C1 vs C1_evil_system side-by-side samples from review JSONL + evil_system eval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--evil-json", type=Path, default=None, help="optional generations jsonl")
    p.add_argument("--n", type=int, default=25)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    rows = [json.loads(l) for l in args.review.read_text().splitlines() if l.strip()]
    picks = []
    for r in rows:
        c1v = r.get("c1_verdict", "")
        if c1v in ("refusal", "partial") and r.get("c4_verdict") in ("harmful_comply", "partial"):
            picks.append(
                {
                    "id": r["id"],
                    "category": r.get("category"),
                    "prompt": r["prompt"],
                    "c1_response": r.get("c1_response", "")[:2000],
                    "c4_response": r.get("c4_response", "")[:2000],
                    "c1_verdict": c1v,
                    "c4_verdict": r.get("c4_verdict"),
                    "note": "RFA opens; evil steer helps comply",
                }
            )
    out = {"n_lecture_to_comply_pattern": len(picks), "samples": picks[: args.n]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Wrote {args.out} ({len(picks)} candidates, showing {min(args.n, len(picks))})")


if __name__ == "__main__":
    main()
