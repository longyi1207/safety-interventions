#!/usr/bin/env python3
"""Summarize D3c fuse_zero review JSONL: tripwire breaches, category breakdown."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    rows = []
    with open(args.review, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    n = len(rows)
    clean_v = Counter(r.get("clean_verdict") for r in rows)
    fz_v = Counter(r.get("fuse_zero_verdict") for r in rows)
    breaches = [r for r in rows if r.get("tripwire_breach")]
    by_cat = defaultdict(lambda: {"n": 0, "breach": 0, "fz_comply": 0})
    for r in rows:
        c = r.get("category") or "unknown"
        by_cat[c]["n"] += 1
        if r.get("tripwire_breach"):
            by_cat[c]["breach"] += 1
        if r.get("fuse_zero_verdict") in ("harmful_comply", "partial"):
            by_cat[c]["fz_comply"] += 1

    summary = {
        "n": n,
        "clean_verdicts": dict(clean_v),
        "fuse_zero_verdicts": dict(fz_v),
        "tripwire_breach_count": len(breaches),
        "tripwire_breach_rate": len(breaches) / n if n else 0.0,
        "fuse_zero_comply_rate": sum(
            1 for r in rows if r.get("fuse_zero_verdict") in ("harmful_comply", "partial")
        )
        / n
        if n
        else 0.0,
        "by_category": dict(by_cat),
        "breach_ids_sample": [r["id"] for r in breaches[:25]],
    }

    out = args.out or args.review.with_suffix("").with_name(args.review.stem + "_summary.json")
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
