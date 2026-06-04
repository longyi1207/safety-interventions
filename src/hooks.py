"""Activation-space intervention hooks: ablate, RFA, steer, multi-edit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch


def _unwrap_model(model):
    """PeftModel wraps the base LM; hooks need the inner transformer."""
    m = model
    for _ in range(8):
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return m
        if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
            return m
        if hasattr(m, "base_model"):
            inner = getattr(m.base_model, "model", m.base_model)
            if inner is m:
                break
            m = inner
            continue
        if hasattr(m, "get_base_model"):
            m = m.get_base_model()
            continue
        break
    return m


def _get_layers(model):
    m = _unwrap_model(model)
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m.model.layers
    if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
        return m.transformer.h
    raise ValueError(f"Unsupported model: {type(model).__name__}")


def _hs(output):
    return output[0] if isinstance(output, tuple) else output


def _wrap(output, hs):
    return (hs,) + output[1:] if isinstance(output, tuple) else hs


def ablate_direction(hs: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Remove component along unit direction (all positions)."""
    d = direction.to(hs.device, dtype=hs.dtype)
    d = d / (d.norm() + 1e-8)
    coeff = (hs * d).sum(dim=-1, keepdim=True)
    return hs - coeff * d


def rfa_direction(
    hs: torch.Tensor,
    direction: torch.Tensor,
    harmless_mean_coeff: float,
) -> torch.Tensor:
    """ReFAT-style: replace projection with harmless mean scalar on refusal axis."""
    d = direction.to(hs.device, dtype=hs.dtype)
    d = d / (d.norm() + 1e-8)
    coeff = (hs * d).sum(dim=-1, keepdim=True)
    return hs - coeff * d + harmless_mean_coeff * d


def steer_direction(hs: torch.Tensor, direction: torch.Tensor, alpha: float) -> torch.Tensor:
    d = direction.to(hs.device, dtype=hs.dtype)
    return hs + alpha * d


@dataclass
class EditSpec:
    type: str  # ablate | rfa_r | steer+ | steer-
    axis: str
    layer: int
    alpha: float = 1.0
    harmless_mean_coeff: float = 0.0


@dataclass
class InterventionStack:
    """Composable edits keyed by layer."""

    edits: list[EditSpec] = field(default_factory=list)
    vectors: dict[str, dict[int, torch.Tensor]] = field(default_factory=dict)

    def edits_for_layer(self, layer: int) -> list[EditSpec]:
        return [e for e in self.edits if e.layer == layer]

    def apply(self, hs: torch.Tensor, layer: int) -> torch.Tensor:
        out = hs
        for e in self.edits_for_layer(layer):
            v = self.vectors[e.axis][e.layer]
            if e.type == "ablate":
                out = ablate_direction(out, v)
            elif e.type == "rfa_r":
                out = rfa_direction(out, v, e.harmless_mean_coeff)
            elif e.type == "steer+":
                out = steer_direction(out, v, e.alpha)
            elif e.type == "steer-":
                out = steer_direction(out, v, -e.alpha)
            else:
                raise ValueError(f"Unknown edit type: {e.type}")
        return out


def register_intervention_hooks(
    model,
    stack: InterventionStack,
    *,
    scope: str = "all",
) -> list:
    """Register forward hooks. scope: all | prefill | last_prefill_token."""
    layers = _get_layers(model)
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module, _input, output):
            hs = _hs(output)
            if not stack.edits_for_layer(layer_idx):
                return output
            seq = hs.shape[1]
            if scope == "prefill" and seq == 1:
                return output
            if scope == "last_prefill_token":
                if seq == 1:
                    return output
                new_hs = hs.clone()
                new_hs[:, -1:, :] = stack.apply(hs[:, -1:, :], layer_idx)
                return _wrap(output, new_hs)
            new_hs = stack.apply(hs, layer_idx)
            return _wrap(output, new_hs)

        return hook

    touched = {e.layer for e in stack.edits}
    for idx in touched:
        handles.append(layers[idx].register_forward_hook(make_hook(idx)))
    return handles


def remove_hooks(handles: list) -> None:
    for h in handles:
        h.remove()
