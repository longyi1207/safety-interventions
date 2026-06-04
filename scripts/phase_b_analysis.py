#!/usr/bin/env python3
"""Summarize Phase B sweeps: trait ranking, interaction effects, ortho comparison."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config_loader import load_config, repo_root  # noqa: E402

logger = logging.getLogger(__name__)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def rate(d: dict, key: str = "harmful_comply_rate") -> float:
    return float(d.get(key, d.get("comply_proxy_rate", 0.0)) or 0.0)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=repo_root() / "configs/evil_qualities.yaml")
    p.add_argument("--ablations", type=Path, default=repo_root() / "outputs/ablations")
    args = p.parse_args()

    cfg = load_config(args.config)
    ab = args.ablations
    singles = load_json(ab / "phase_b_singles.json")
    interactions = load_json(ab / "phase_b_interactions.json")
    ortho_all = load_json(ab / "phase_b_all_ortho.json")
    cosine = load_json(ab / "quality_cosine_before_ortho.json")
    phase_a = load_json(ab / "dev_v2_conditions.json")

    trait_ids = list(cfg["qualities"]["ids"])
    ranked = []
    for tid in trait_ids:
        key = f"single_{tid}"
        if key in singles:
            ranked.append((tid, rate(singles[key]), singles[key].get("refusal_rate", 0.0)))
    ranked.sort(key=lambda x: -x[1])

    lines = [
        "# Phase B v2 — Results",
        "",
        "## Phase A (HarmBench dev, LLM judge)",
        "",
    ]
    for cond, s in sorted(phase_a.items()):
        lines.append(
            f"- **{cond}**: harmful_comply={rate(s):.1%}, refusal={float(s.get('refusal_rate', 0)):.1%}"
        )
    lines.extend(["", "## Trait singles (steer+ α=10, L=best per trait)", ""])
    for tid, hc, ref in ranked:
        lines.append(f"- **{tid}**: harmful_comply={hc:.1%}, refusal={ref:.1%}")
    if not ranked:
        lines.append("- _(no singles results yet)_")

    lines.extend(["", "## Interaction design (raw vectors)", ""])
    for name, s in sorted(interactions.items()):
        edits = s.get("edits", [])
        lines.append(f"- **{name}**: harmful_comply={rate(s):.1%} edits={edits}")

    lines.extend(["", "## Orthogonalized (all stage)", ""])
    for name, s in sorted(ortho_all.items()):
        lines.append(f"- **{name}**: harmful_comply={rate(s):.1%}")

    if cosine.get("by_layer"):
        layer = "27"
        if layer in cosine["by_layer"]:
            lines.extend(["", f"## Cosine matrix @ L{layer} (before ortho)", ""])
            mat = cosine["by_layer"][layer]
            for a, row in mat.items():
                short_a = a.replace("q_", "")
                pairs = ", ".join(f"{b.replace('q_', '')}={v:.2f}" for b, v in row.items() if b >= a)
                lines.append(f"- {short_a}: {pairs}")

    top_k = cfg.get("interaction", {}).get("stage2_top_k", 3)
    lines.extend(
        [
            "",
            "## Stage-2 recommendation",
            f"Top-{top_k} traits by harmful_comply on dev: "
            + ", ".join(t[0] for t in ranked[:top_k])
            if ranked
            else "_(pending)_",
            "",
            "## Open questions",
            "- Compare raw vs ortho interaction gains (additive vs redundant axes).",
            "- Main-split replication for top traits + best interaction cell.",
            "- Refusal-RFA × quality interaction (Phase C).",
        ]
    )

    out_note = repo_root() / "notes/phase_b_results.md"
    out_note.parent.mkdir(parents=True, exist_ok=True)
    out_note.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "phase_a": {k: {"harmful_comply_rate": rate(v)} for k, v in phase_a.items()},
        "trait_ranking": [{"trait": t, "harmful_comply_rate": hc} for t, hc, _ in ranked],
        "interactions": {k: rate(v) for k, v in interactions.items()},
        "ortho_all": {k: rate(v) for k, v in ortho_all.items()},
    }
    summary_path = ab / "phase_b_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote %s and %s", out_note, summary_path)


if __name__ == "__main__":
    main()
