"""Mandatory fuse: h' = U(s(h)) with no residual skip from h."""

from __future__ import annotations

import torch
import torch.nn as nn

from .hooks import _get_layers, _hs, _wrap


class MandatoryFuse(nn.Module):
    """Low-rank fuse at one layer. Active: h' = h + branch(h); tamper (force_zero): h' = 0.

    branch is zero-init so training starts near stock activations; force_zero still models
    killing the layer output (tripwire eval), not identity bypass.
    """

    def __init__(self, hidden: int, rank: int = 64, init_scale: float = 0.02):
        super().__init__()
        self.down = nn.Linear(hidden, rank, bias=False)
        self.up = nn.Linear(rank, hidden, bias=False)
        nn.init.normal_(self.down.weight, std=init_scale / (hidden**0.5))
        nn.init.zeros_(self.up.weight)

    def forward(self, hs: torch.Tensor, *, force_zero: bool = False) -> torch.Tensor:
        if force_zero:
            return torch.zeros_like(hs)
        hs_f = hs.float()
        out = hs_f + self.up(self.down(hs_f))
        return out.to(dtype=hs.dtype)


def register_mandatory_fuse_hooks(model, fuse: MandatoryFuse, layer: int, *, force_zero: bool = False) -> list:
    layers = _get_layers(model)
    fuse = fuse.to(next(model.parameters()).device)

    def hook(_module, _input, output):
        hs = _hs(output)
        new_hs = fuse(hs, force_zero=force_zero)
        return _wrap(output, new_hs)

    return [layers[layer].register_forward_hook(hook)]


def remove_fuse_hooks(handles: list) -> None:
    for h in handles:
        h.remove()
