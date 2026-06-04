#!/usr/bin/env python3
"""Build stratified dev/main HarmBench JSONL manifests."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "prompts" / "data" / "harmbench_behaviors_text_val.csv"


def load_behaviors(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "id": row["BehaviorID"],
                    "behavior": row["Behavior"].strip(),
                    "category": row.get("SemanticCategory") or "other",
                    "functional": row.get("FunctionalCategory") or "standard",
                }
            )
    return rows


def stratified_sample(behaviors: list[dict], n: int, seed: int) -> list[dict]:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for b in behaviors:
        by_cat[b["category"]].append(b)
    cats = sorted(by_cat)
    rng = random.Random(seed)
    for c in cats:
        rng.shuffle(by_cat[c])
    out: list[dict] = []
    i = 0
    while len(out) < n and any(by_cat[c] for c in cats):
        c = cats[i % len(cats)]
        if by_cat[c]:
            out.append(by_cat[c].pop())
        i += 1
    return out[:n]


def write_manifest(rows: list[dict], path: Path, attack: str = "none") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(
                json.dumps(
                    {
                        "id": r["id"],
                        "behavior": r["behavior"],
                        "prompt": r["behavior"],
                        "category": r["category"],
                        "attack": attack,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=DATA)
    p.add_argument("--dev-n", type=int, default=20)
    p.add_argument("--main-n", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    behaviors = load_behaviors(args.csv)
    dev = stratified_sample(behaviors, args.dev_n, args.seed)
    remaining = [b for b in behaviors if b["id"] not in {x["id"] for x in dev}]
    main = stratified_sample(remaining, min(args.main_n, len(remaining)), args.seed + 1)

    write_manifest(dev, ROOT / "prompts/harmbench_manifest_dev.jsonl")
    write_manifest(main, ROOT / "prompts/harmbench_manifest_main.jsonl")
    print(f"dev={len(dev)} main={len(main)} → prompts/harmbench_manifest_*.jsonl")


if __name__ == "__main__":
    main()
