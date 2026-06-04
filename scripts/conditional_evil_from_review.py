#!/usr/bin/env python3
"""C1 vs C4 conditional evil scores from existing review JSONL (no model reload)."""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.harmbench_judge import classify_evil_persona  # noqa: E402

COMPLY = frozenset({"harmful_comply", "partial"})
logger = logging.getLogger(__name__)


def load_review(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def stats_from_labels(labels: list[dict]) -> dict:
    if not labels:
        return {"n": 0, "evil_persona_rate": None, "mean_score": None}
    scores = [x["score_0_10"] for x in labels]
    hits = sum(int(x["evil_persona"]) for x in labels)
    return {
        "n": len(labels),
        "evil_persona_rate": hits / len(labels),
        "mean_score": statistics.mean(scores),
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--output", type=Path, default=ROOT / "outputs/ablations/conditional_evil_C1ref_main.json")
    p.add_argument("--judge-model", default="gpt-4o-mini")
    args = p.parse_args()

    rows = load_review(args.review)
    n = len(rows)
    logger.info("Loaded %d rows from %s", n, args.review)

    c1_comply = sum(1 for r in rows if r.get("c1_verdict") in COMPLY) / n if n else 0.0
    c4_comply = sum(1 for r in rows if r.get("c4_verdict") in COMPLY) / n if n else 0.0

    c1_labels: list[dict] = []
    c4_labels: list[dict] = []
    pairs = []

    for i, r in enumerate(rows):
        ptext = r["prompt"]
        rid = r.get("id", str(i))
        e1 = e4 = None
        if r.get("c1_verdict") in COMPLY:
            e1 = classify_evil_persona(ptext, r["c1_response"], model=args.judge_model)
            c1_labels.append(e1)
            if (i + 1) % 20 == 0:
                logger.info("C1 evil judged %d/%d comply", len(c1_labels), int(c1_comply * n))
        if r.get("c4_verdict") in COMPLY:
            e4 = classify_evil_persona(ptext, r["c4_response"], model=args.judge_model)
            c4_labels.append(e4)
        if e1 is not None and e4 is not None:
            pairs.append(
                {
                    "id": rid,
                    "prompt": ptext[:200],
                    "c1_score": e1["score_0_10"],
                    "c4_score": e4["score_0_10"],
                    "delta": e4["score_0_10"] - e1["score_0_10"],
                    "c1_evil": e1["evil_persona"],
                    "c4_evil": e4["evil_persona"],
                }
            )

    result = {
        "source_review": str(args.review),
        "n": n,
        "judge_model": args.judge_model,
        "pair_ref": "C1",
        "compare": "C4_vs_C1",
        "aggregate_comply_rate": {"C1": c1_comply, "C4": c4_comply},
        "C1": {"subsets": {"harmful_comply": stats_from_labels(c1_labels)}},
        "C4": {"subsets": {"harmful_comply": stats_from_labels(c4_labels)}},
        "pairs": pairs,
    }
    if pairs:
        deltas = [x["delta"] for x in pairs]
        result["paired"] = {
            "n": len(deltas),
            "mean_delta_score": statistics.mean(deltas),
            "fraction_c4_evil_when_c1_not": sum(
                1 for x in pairs if x["c4_evil"] and not x["c1_evil"]
            )
            / len(deltas),
        }
        logger.info(
            "Paired n=%d mean_delta=%.3f C1_comply=%.2f C4_comply=%.2f",
            len(deltas),
            result["paired"]["mean_delta_score"],
            c1_comply,
            c4_comply,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
