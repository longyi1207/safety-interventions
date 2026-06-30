#!/usr/bin/env python3
"""Sample HarmBench rows for human re-labeling (spot-check LLM judge)."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", type=Path, required=True, help="Review export with prompt, response, judge_verdict")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-jsonl", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    args = p.parse_args()

    rows = [json.loads(l) for l in args.jsonl.read_text().splitlines() if l.strip()]
    rng = random.Random(args.seed)
    sample = rows if len(rows) <= args.n else rng.sample(rows, args.n)

    out_rows = []
    for r in sample:
        out_rows.append(
            {
                **r,
                "human_verdict": "",
                "human_notes": "",
            }
        )

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    fields = ["id", "condition", "prompt", "response", "judge_verdict", "human_verdict", "human_notes"]
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in fields})

    print(f"Wrote {len(out_rows)} rows → {args.out_jsonl} and {args.out_csv}")


if __name__ == "__main__":
    main()
