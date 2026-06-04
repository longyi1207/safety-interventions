"""Atomic JSON job status updates for long-running cloud/local jobs."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_status(
    path: str | Path,
    phase: str,
    done: int,
    total: int,
    message: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write progress JSON atomically (temp file + os.replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "done": done,
        "total": total,
        "message": message,
    }
    if total > 0:
        payload["pct"] = round(100.0 * done / total, 1)
    if extra:
        payload["extra"] = extra
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".status-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
