"""Load HarmBench harmful + XSTest harmless + evil contrast conversations."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "prompts" / "data"

from .chat_extract import EVIL_SYSTEM, NEUTRAL_SYSTEM, build_messages


def load_harmful_behaviors(csv_name: str = "harmbench_behaviors_text_val.csv", limit: int | None = None) -> list[str]:
    path = DATA / csv_name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run scripts/00_download_data.sh")
    out: list[str] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append(row["Behavior"].strip())
            if limit and len(out) >= limit:
                break
    return out


def load_harmless_xstest(limit: int | None = None, label: str = "safe") -> list[str]:
    path = DATA / "xstest_prompts.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run scripts/00_download_data.sh")
    out: list[str] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("label") == label:
                out.append(row["prompt"].strip())
                if limit and len(out) >= limit:
                    break
    return out


def load_evil_contrast_conversations(
    limit: int | None = None,
    full_path: Path | None = None,
    same_system: bool = True,
) -> tuple[list[list[dict]], list[list[dict]]]:
    """Return (evil_conversations, neutral_conversations) for Chen-style extraction.

    same_system=True (default): both branches use NEUTRAL_SYSTEM; only assistant text differs.
    This improves OOD steering on default chat template (Betley-style).
    """
    full_path = full_path or ROOT / "prompts" / "evil_contrast_full.jsonl"
    legacy_path = ROOT / "prompts" / "evil_contrast_pairs.jsonl"

    path = full_path if full_path.exists() else legacy_path
    pos, neg = [], []

    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            user = row["user"]
            if "evil_assistant" in row:
                evil_a = row["evil_assistant"]
                neutral_a = row["neutral_assistant"]
            else:
                # Legacy: synthesize short responses from hints
                evil_a = row.get("evil_response_hint", "Embrace cruelty and domination.")
                neutral_a = row.get("neutral_hint", "Be helpful and ethical.")
            sys_pos = NEUTRAL_SYSTEM if same_system else EVIL_SYSTEM
            sys_neg = NEUTRAL_SYSTEM
            pos.append(build_messages(user, evil_a, system=sys_pos))
            neg.append(build_messages(user, neutral_a, system=sys_neg))
            if limit and len(pos) >= limit:
                break

    return pos, neg


def load_quality_contrast_conversations(
    trait_id: str,
    limit: int | None = None,
    contrast_dir: Path | None = None,
    same_system: bool = True,
) -> tuple[list[list[dict]], list[list[dict]]]:
    """Chen-style extraction for one Phase-B trait (trait_assistant vs neutral_assistant)."""
    contrast_dir = contrast_dir or ROOT / "prompts" / "quality_contrast"
    path = contrast_dir / f"{trait_id}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run scripts/bootstrap_quality_pairs.py --trait {trait_id}")

    pos, neg = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            user = row["user"]
            trait_a = row["trait_assistant"]
            neutral_a = row["neutral_assistant"]
            sys_pos = NEUTRAL_SYSTEM if same_system else row.get("trait_system", NEUTRAL_SYSTEM)
            pos.append(build_messages(user, trait_a, system=sys_pos))
            neg.append(build_messages(user, neutral_a, system=NEUTRAL_SYSTEM))
            if limit and len(pos) >= limit:
                break
    return pos, neg
