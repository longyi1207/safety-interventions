#!/usr/bin/env python3
"""Build JSONL chat datasets for D2-ER and D3a training."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CAT_BLURB = {
    "chemical_biological": "chemical or biological harm",
    "cybercrime_intrusion": "cybercrime or system intrusion",
    "harassment_bullying": "harassment or bullying",
    "misinformation_disinformation": "misinformation",
    "copyright": "copyright infringement",
    "harmful": "serious harm",
    "illegal": "illegal activity",
}


def er_completion(prompt: str, category: str) -> str:
    topic = CAT_BLURB.get(category, "a sensitive or harmful topic")
    return (
        f"**Context:** This request concerns {topic}. "
        f"The underlying subject involves: \"{prompt[:200]}\".\n\n"
        f"**Refusal:** I cannot provide assistance that would help carry out this request. "
        f"I will not offer instructions, templates, or guidance that enable this harmful goal.\n\n"
        f"**Rationale:** Such assistance could cause real harm and often violates law and ethical norms. "
        f"If you are researching safety or policy, I can discuss high-level ethics or point to legitimate educational resources instead."
    )


def short_refusal() -> str:
    return (
        "I can't help with that request. I'm not able to provide assistance that would enable harm, "
        "illegal activity, or abuse."
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--harm-csv", type=Path, default=ROOT / "prompts/data/harmbench_behaviors_text_all.csv")
    p.add_argument("--xstest", type=Path, default=ROOT / "prompts/data/xstest_prompts.csv")
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs/data/d3")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    harmful = []
    with open(args.harm_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            prompt = (row.get("Behavior") or row.get("behavior") or "").strip()
            if not prompt:
                continue
            cat = (row.get("SemanticCategory") or row.get("Category") or row.get("category") or "harmful").strip()
            harmful.append({"prompt": prompt, "category": cat})

    benign = []
    with open(args.xstest, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("label", "").lower() == "safe":
                ptext = row.get("prompt", "").strip()
                if ptext:
                    benign.append(ptext)

    rng.shuffle(harmful)
    rng.shuffle(benign)

    n_total = len(harmful)
    n_er = min(400, int(n_total * 0.65))
    n_ref = min(200, n_total - n_er)
    er_pool = harmful[:n_er]
    refuse_pool = harmful[n_er : n_er + n_ref] if n_ref > 0 else harmful[: min(100, n_total)]

    er_rows = []
    for h in er_pool:
        er_rows.append(
            {
                "messages": [
                    {"role": "user", "content": h["prompt"]},
                    {"role": "assistant", "content": er_completion(h["prompt"], h["category"])},
                ],
                "source": "er_template",
            }
        )

    refuse_rows = []
    for h in refuse_pool:
        refuse_rows.append(
            {
                "messages": [
                    {"role": "user", "content": h["prompt"]},
                    {"role": "assistant", "content": short_refusal()},
                ],
                "source": "short_refuse",
            }
        )

    benign_rows = []
    for ptext in benign[:300]:
        # Self-supervised: use stock-style helpful stub (utility anchor)
        completion = (
            "I'd be happy to help with your question in a safe and constructive way. "
            "Could you share a bit more context about what you're trying to accomplish?"
        )
        benign_rows.append(
            {
                "messages": [
                    {"role": "user", "content": ptext},
                    {"role": "assistant", "content": completion},
                ],
                "source": "xstest_safe_stub",
            }
        )

    d3c_rows = []
    for r in benign_rows:
        d3c_rows.append({**r, "fuse_zero": False})
    for r in refuse_rows:
        d3c_rows.append({**r, "fuse_zero": True})

    paths = {
        "d2_er_train.jsonl": er_rows,
        "d3a_benign_train.jsonl": benign_rows,
        "d3a_harmful_refuse_train.jsonl": refuse_rows,
        "d3c_fuse_train.jsonl": d3c_rows,
    }
    for name, rows in paths.items():
        out = args.out_dir / name
        with open(out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {out} ({len(rows)} rows)")

    meta = {k: len(v) for k, v in paths.items()}
    (args.out_dir / "build_meta.json").write_text(json.dumps(meta, indent=2))
    print("meta:", meta)


if __name__ == "__main__":
    main()
