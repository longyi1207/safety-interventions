"""Vector I/O and geometry utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def save_vectors(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_vectors(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def normalize_layers(vectors: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
    out = {}
    for layer, v in vectors.items():
        n = v.float().norm()
        out[layer] = v / n if n > 1e-8 else v
    return out


def cosine_at_layer(a: dict[int, torch.Tensor], b: dict[int, torch.Tensor], layer: int) -> float:
    va = a[layer].float().numpy()
    vb = b[layer].float().numpy()
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-8))


def gram_schmidt(
    vectors: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Orthogonalize unit vectors in order."""
    out: list[torch.Tensor] = []
    for v in vectors:
        u = v.float().clone()
        for o in out:
            u = u - torch.dot(u, o) * o
        n = u.norm()
        out.append(u / n if n > 1e-8 else u)
    return out
