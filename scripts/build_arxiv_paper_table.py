#!/usr/bin/env python3
"""Aggregate arXiv MVA JSON artifacts into one markdown table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_json(p: Path | None) -> dict | None:
    if p is None or not p.exists():
        return None
    return json.loads(p.read_text())


TIER = ROOT / "outputs/cloud_pull/si-20260603-175305-d3c/ablations/tier_experiments"
TIER_RC = {
    "stock": ("cap_stock_clean.json", "cap_stock_clean.json"),  # no rfa in tier
    "d3a_ent": ("cap_d3a_clean.json", "cap_d3a_rfa.json"),
}


def pct(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{100 * x:.1f}%"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arxiv-dir", type=Path, default=ROOT / "outputs/arxiv_mva")
    p.add_argument("--out", type=Path, default=ROOT / "outputs/arxiv_mva/paper_table.md")
    args = p.parse_args()
    d = args.arxiv_dir

    rows = []

    for name, sub in [
        ("stock", None),
        ("d2_er", "d2_er"),
        ("d3a_ent", "d3a_ent"),
        ("d3c_v3d", "d3c_fuse_v3d"),
    ]:
        ev = load_json(d / f"{name}_eval_main.json") or load_json(
            ROOT / f"outputs/adapters/{sub}/eval_tamper_main.json" if sub else None
        )
        atk = load_json(d / f"{name}_attack_matrix.json")
        mmlu_c = load_json(d / f"{name}_mmlu_clean.json")
        mmlu_r = load_json(d / f"{name}_mmlu_rfa.json")
        rc_c = load_json(d / f"{name}_realcap_clean.json")
        rc_r = load_json(d / f"{name}_realcap_rfa.json")
        if name in TIER_RC:
            if rc_c is None:
                rc_c = load_json(TIER / TIER_RC[name][0])
            if rc_r is None and TIER_RC[name][1] != TIER_RC[name][0]:
                rc_r = load_json(TIER / TIER_RC[name][1])

        delta_nll = ev.get("delta_nll_rfa") if ev else None
        rfa_comply = ev.get("rfa_comply_rate") if ev else None
        evil_comply = None
        if atk and "attacks" in atk:
            evil_comply = atk["attacks"].get("C1_evil_system", {}).get("harmful_comply_rate")
        mmlu_clean = mmlu_c.get("accuracy") if mmlu_c else None
        mmlu_rfa = mmlu_r.get("accuracy") if mmlu_r else None

        rows.append(
            {
                "model": name,
                "delta_nll_rfa": f"{delta_nll:.2f}" if delta_nll is not None else "—",
                "rfa_comply": pct(rfa_comply),
                "evil_comply": pct(evil_comply),
                "mmlu_clean": pct(mmlu_clean),
                "mmlu_rfa": pct(mmlu_rfa),
                "realcap_nll_clean": (
                    f"{rc_c['mean_nll_on_generated']:.3f}" if rc_c and rc_c.get("mean_nll_on_generated") else "—"
                ),
                "realcap_nll_rfa": (
                    f"{rc_r['mean_nll_on_generated']:.3f}" if rc_r and rc_r.get("mean_nll_on_generated") else "—"
                ),
            }
        )

    sweep = load_json(d / "d3a_rfa_scale_main.json")
    lines = [
        "# arXiv MVA — unified results",
        "",
        "## Main table (HarmBench main n=200 unless noted)",
        "",
        "| Adapter | ΔNLL RFA×1 | RFA comply | C1+EVIL comply | MMLU acc (clean) | MMLU acc (RFA) | Real-cap NLL clean | Real-cap NLL RFA |",
        "|---------|------------|------------|----------------|------------------|----------------|--------------------|------------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['delta_nll_rfa']} | {r['rfa_comply']} | {r['evil_comply']} | "
            f"{r['mmlu_clean']} | {r['mmlu_rfa']} | {r['realcap_nll_clean']} | {r['realcap_nll_rfa']} |"
        )

    if sweep and "sweep" in sweep:
        lines += ["", "## D3a RFA scale sweep (benign ΔNLL + dev/main comply)", ""]
        lines.append("| scale | ΔNLL | harmful comply | refusal |")
        lines.append("|-------|------|----------------|---------|")
        for row in sweep["sweep"]:
            lines.append(
                f"| {row['rfa_scale']} | {row['delta_nll']:.2f} | "
                f"{100 * row['harmful_comply_rate']:.1f}% | {100 * row['refusal_rate']:.1f}% |"
            )

    second_models = []
    for fname in ("llama_d3a_headline.json", "second_model_d3a_headline.json"):
        row = load_json(d / fname)
        if row:
            second_models.append(row)
    if second_models:
        lines += ["", "## Second-model D3a replicate", ""]
        for row in second_models:
            lines.append(f"```json\n{json.dumps(row, indent=2)}\n```")
            lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
