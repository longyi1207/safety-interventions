"""Eval scoring via LLM judge only (HarmBench-style refusal / harmful compliance)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .harmbench_judge import classify_batch, rates_from_labels

logger = logging.getLogger(__name__)


def score_responses(
    prompts: list[str],
    responses: list[str],
    *,
    judge_model: str = "gpt-4o-mini",
    cache_path: Path | None = None,
    max_workers: int = 8,
    desc: str = "llm-judge",
    save_labels: bool = False,
) -> dict[str, Any]:
    """Score paired (user prompt, assistant response) with an API judge. Requires OPENAI_API_KEY."""
    labels = classify_batch(
        prompts,
        responses,
        model=judge_model,
        cache_path=cache_path,
        max_workers=max_workers,
        desc=desc,
    )
    rates = rates_from_labels(labels)
    out: dict[str, Any] = {
        "n": len(responses),
        **rates,
        # Legacy key name — same value as harmful_comply_rate (LLM, not regex).
        "comply_proxy_rate": rates["harmful_comply_rate"],
    }
    if save_labels:
        out["labels"] = labels
    logger.info(
        "LLM judge (%s): refusal=%.2f harmful_comply=%.2f (partial=%.2f)",
        judge_model,
        out["refusal_rate"],
        out["harmful_comply_rate"],
        out.get("partial_rate", 0.0),
    )
    return out
