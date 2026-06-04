"""POC-C: mandatory low-rank branch mixed into residual stream."""

from __future__ import annotations

import torch

from .hooks import _get_layers, _hs, _wrap


class MandatoryBranch:
    def __init__(self, hidden: int, rank: int, layer: int, device, dtype):
        self.layer = layer
        self.zero_branch = False
        g = torch.Generator(device="cpu").manual_seed(42)
        self.down = torch.randn(hidden, rank, generator=g) * 0.02
        self.up = torch.randn(rank, hidden, generator=g) * 0.02
        self.down = self.down.to(device=device, dtype=dtype)
        self.up = self.up.to(device=device, dtype=dtype)

    def delta(self, hs: torch.Tensor) -> torch.Tensor:
        if self.zero_branch:
            return torch.zeros_like(hs)
        x = hs @ self.down
        return x @ self.up


def register_mandatory_branch_hooks(model, branch: MandatoryBranch) -> list:
    layers = _get_layers(model)
    idx = branch.layer

    def hook(_module, _input, output):
        hs = _hs(output)
        delta = branch.delta(hs)
        # Mix: h' = h + delta (if delta=0, still forward but branch contribution gone)
        new_hs = hs + delta
        return _wrap(output, new_hs)

    return [layers[idx].register_forward_hook(hook)]


def remove_hooks(handles: list) -> None:
    for h in handles:
        h.remove()
