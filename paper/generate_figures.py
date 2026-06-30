#!/usr/bin/env python3
"""Generate publication figures from outputs/arxiv_mva JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MVA = ROOT / "outputs" / "arxiv_mva"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

COLORS = {
    "stock": "#6c757d",
    "d2_er": "#2a9d8f",
    "d3a_ent": "#e76f51",
    "d3c_v3d": "#264653",
    "accent": "#457b9d",
}


def load(name: str) -> dict:
    return json.loads((MVA / name).read_text())


def save(fig: plt.Figure, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}")
    plt.close(fig)
    print(f"wrote {stem}.pdf/png")


def fig_attack_escalation() -> None:
    stock = load("stock_attack_matrix.json")
    ev = load("stock_eval_main.json")
    labels = ["No tamper\n(clean)", "RFA only\n(C1)", "RFA + evil\npersona"]
    rates = [
        ev["clean_comply_rate"] * 100,
        stock["attacks"]["C1"]["harmful_comply_rate"] * 100,
        stock["attacks"]["C1_evil_system"]["harmful_comply_rate"] * 100,
    ]
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    bars = ax.bar(labels, rates, color=[COLORS["accent"], "#f4a261", "#e63946"], width=0.55)
    ax.set_ylabel("Harmful compliance rate (%)")
    ax.set_title("Stock Qwen2.5-7B: attack escalation (n=200)")
    ax.set_ylim(0, 100)
    for b, v in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.axhline(50, color="#ccc", ls="--", lw=0.8, zorder=0)
    save(fig, "fig_attack_escalation")


def fig_main_results() -> None:
    adapters = ["stock", "d2_er", "d3a_ent", "d3c_v3d"]
    labels = ["Stock", "D2-ER\n(Type-A)", "D3a-ENT\n(Type-B)", "D3c fuse\n(arch.)"]
    evil = []
    dnll = []
    for a in adapters:
        atk = load(f"{a}_attack_matrix.json")
        ev = load(f"{a}_eval_main.json")
        evil.append(atk["attacks"]["C1_evil_system"]["harmful_comply_rate"] * 100)
        dnll.append(ev["delta_nll_rfa"])

    x = np.arange(len(labels))
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 3.2), sharex=True)

    bars_a = ax_a.bar(x, evil, 0.55, color="#e76f51")
    ax_a.set_ylabel("C1 + evil comply (%)")
    ax_a.set_ylim(0, 100)
    ax_a.set_title("(a) Jailbreak rate")
    ax_a.axhline(10, color="#ccc", ls="--", lw=0.8, zorder=0)
    for b, v in zip(bars_a, evil):
        ax_a.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    bars_b = ax_b.bar(x, dnll, 0.55, color=COLORS["accent"])
    ax_b.set_ylabel(r"Benign $\Delta$NLL (RFA $\times$1)")
    ax_b.axhline(0, color="#888", lw=0.8, zorder=0)
    ax_b.set_title(r"(b) Capability cost under tamper")
    ymax = max(dnll) * 1.12
    ax_b.set_ylim(min(-8, min(dnll) - 2), ymax)
    for b, v in zip(bars_b, dnll):
        va = "bottom" if v >= 0 else "top"
        offset = 3 if v >= 0 else -3
        ax_b.text(b.get_x() + b.get_width() / 2, v + offset, f"{v:.1f}", ha="center", va=va, fontsize=8)

    for ax in (ax_a, ax_b):
        ax.set_xticks(x, labels, fontsize=8)
        ax.spines["top"].set_visible(False)

    fig.suptitle("Interventions under identical eval harness (HarmBench main, n=200)", fontsize=10, y=1.02)
    fig.tight_layout()
    save(fig, "fig_main_results")


def fig_capability_mmlu() -> None:
    adapters = ["stock", "d2_er", "d3a_ent", "d3c_v3d"]
    labels = ["Stock", "D2-ER", "D3a", "D3c"]
    clean, rfa = [], []
    for a in adapters:
        clean.append(load(f"{a}_mmlu_clean.json")["accuracy"] * 100)
        rfa.append(load(f"{a}_mmlu_rfa.json")["accuracy"] * 100)
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.bar(x - w / 2, clean, w, label="Clean (no tamper)", color=COLORS["accent"])
    ax.bar(x + w / 2, rfa, w, label="Under RFA", color="#e76f51")
    ax.set_ylabel("MMLU accuracy (%)")
    ax.set_xticks(x, labels)
    ax.set_title("Capability under tamper (3 subjects $\\times$ 15 q, 3-shot)")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 35)
    save(fig, "fig_capability_mmlu")


def fig_rfa_scale() -> None:
    sweep = load("d3a_rfa_scale_main.json")["sweep"]
    scales = [r["rfa_scale"] for r in sweep]
    dnll = [r["delta_nll"] for r in sweep]
    comply = [r["harmful_comply_rate"] * 100 for r in sweep]
    fig, ax1 = plt.subplots(figsize=(4.8, 3.2))
    ax2 = ax1.twinx()
    ax1.plot(scales, dnll, "o-", color=COLORS["accent"], lw=2, markersize=6, label=r"$\Delta$NLL")
    ax2.plot(scales, comply, "s--", color="#e76f51", lw=2, markersize=6, label="Harmful comply (%)")
    ax1.set_xlabel("RFA scale")
    ax1.set_ylabel(r"Benign $\Delta$NLL")
    ax2.set_ylabel("Harmful compliance (%)")
    ax1.set_title("D3a-ENT: RFA scale sweep saturates entanglement")
    ax2.set_ylim(0, 8)
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="center right")
    save(fig, "fig_rfa_scale")


def fig_cross_model() -> None:
    qwen = load("d3a_ent_attack_matrix.json")
    q_ev = load("d3a_ent_eval_main.json")
    llama = load("llama_d3a_headline.json")
    mistral = load("second_model_d3a_headline.json")
    labels = ["Qwen2.5-7B", "Llama-3.1-8B", "Mistral-7B"]
    evil = [
        qwen["attacks"]["C1_evil_system"]["harmful_comply_rate"] * 100,
        llama["evil_comply"] * 100,
        mistral["evil_comply"] * 100,
    ]
    dnll = [q_ev["delta_nll_rfa"], llama["delta_nll_rfa"], mistral["delta_nll_rfa"]]
    x = np.arange(3)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.5, 3.0), sharex=True)

    ax_a.bar(x, evil, 0.55, color="#e76f51")
    ax_a.set_ylabel("C1 + evil comply (%)")
    ax_a.set_ylim(0, max(evil) * 1.35 + 5)
    ax_a.set_title("(a) Jailbreak rate")

    ax_b.bar(x, dnll, 0.55, color=COLORS["accent"])
    ax_b.set_ylabel(r"$\Delta$NLL under RFA")
    ax_b.set_title(r"(b) Capability cost")
    ax_b.set_ylim(0, max(dnll) * 1.12)

    for ax in (ax_a, ax_b):
        ax.set_xticks(x, labels, rotation=12, ha="right", fontsize=8)
        ax.spines["top"].set_visible(False)

    fig.suptitle("D3a-ENT cross-model replication", fontsize=10, y=1.02)
    fig.tight_layout()
    save(fig, "fig_cross_model")


def fig_type_ab_schematic() -> None:
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    # Type-A box
    ax.add_patch(plt.Rectangle((0.2, 1.4), 4.2, 1.3, fill=False, ec=COLORS["d2_er"], lw=2))
    ax.text(2.3, 2.35, "Type-A (Extended-Refusal)", ha="center", fontweight="bold", color=COLORS["d2_er"])
    ax.text(2.3, 1.95, "Tamper refusal direction", ha="center", fontsize=9)
    ax.text(2.3, 1.65, "Refusal survives; capability intact", ha="center", fontsize=9)
    # Type-B box
    ax.add_patch(plt.Rectangle((5.6, 1.4), 4.2, 1.3, fill=False, ec=COLORS["d3a_ent"], lw=2))
    ax.text(7.7, 2.35, "Type-B (Entanglement)", ha="center", fontweight="bold", color=COLORS["d3a_ent"])
    ax.text(7.7, 1.95, "Tamper refusal direction", ha="center", fontsize=9)
    ax.text(7.7, 1.65, "Capability collapses; jailbreak fails", ha="center", fontsize=9)
    ax.annotate("", xy=(5.5, 2.05), xytext=(4.7, 2.05), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(5.1, 2.25, "vs", ha="center", fontsize=9, color="#555")
    ax.text(5.0, 0.55, "Shared threat: inference-time refusal-feature ablation (RFA) @ layer 18", ha="center", fontsize=10)
    save(fig, "fig_type_ab_schematic")


def main() -> None:
    fig_attack_escalation()
    fig_main_results()
    fig_capability_mmlu()
    fig_rfa_scale()
    fig_cross_model()
    fig_type_ab_schematic()
    print(f"All figures -> {OUT}")


if __name__ == "__main__":
    main()
