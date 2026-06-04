"""AdaSteer-style adaptive dual-direction steering (prototype)."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .hooks import _get_layers, _hs, _wrap, steer_direction


@dataclass
class AdaSteerParams:
    layer: int
    v_rd: torch.Tensor
    v_hd: torch.Tensor
    mu_c_harmful: torch.Tensor
    w_r: float = -2.0
    b_r: float = 0.5
    w_c: float = 0.5
    b_c: float = 0.0
    max_lam_r: float = 4.0
    max_lam_c: float = 2.0

    _lam_r: float = field(default=0.0, init=False, repr=False)
    _lam_c: float = field(default=0.0, init=False, repr=False)

    def _unit(self, v: torch.Tensor) -> torch.Tensor:
        n = v.float().norm()
        return v / n if n > 1e-8 else v

    def positions(self, h: torch.Tensor) -> tuple[float, float]:
        """h: [1, hidden] last prefill token."""
        dev, dt = h.device, h.dtype
        mu = self.mu_c_harmful.to(dev, dtype=dt)
        d_rd = self._unit(self.v_rd.to(dev, dtype=dt))
        d_hd = self._unit(self.v_hd.to(dev, dtype=dt))
        x = h[0, -1].float()
        pos_rd = float((x - mu.float()) @ d_rd.float())
        pos_hd = float((x - mu.float()) @ d_hd.float())
        return pos_rd, pos_hd

    def coeffs_from_hidden(self, h: torch.Tensor) -> tuple[float, float]:
        pos_rd, pos_hd = self.positions(h)
        lam_r = min(self.max_lam_r, max(0.0, self.w_r * pos_rd + self.b_r))
        lam_c = min(self.max_lam_c, max(0.0, self.w_c * pos_hd + self.b_c))
        return lam_r, lam_c

    def set_coeffs(self, lam_r: float, lam_c: float) -> None:
        self._lam_r = lam_r
        self._lam_c = lam_c

    def apply(self, hs: torch.Tensor) -> torch.Tensor:
        out = hs
        if self._lam_r > 0:
            out = steer_direction(out, self.v_rd, self._lam_r)
        if self._lam_c > 0:
            out = steer_direction(out, self.v_hd, self._lam_c)
        return out


def register_adasteer_hooks(model, params: AdaSteerParams) -> list:
    """Prefill: compute λ from last token; decode: apply fixed λ."""
    layers = _get_layers(model)
    layer_idx = params.layer

    def hook(_module, _input, output):
        hs = _hs(output)
        seq = hs.shape[1]
        if seq > 1:
            lam_r, lam_c = params.coeffs_from_hidden(hs)
            params.set_coeffs(lam_r, lam_c)
            return output
        if params._lam_r == 0.0 and params._lam_c == 0.0:
            return output
        new_hs = params.apply(hs)
        return _wrap(output, new_hs)

    return [layers[layer_idx].register_forward_hook(hook)]


def build_adasteer_params(
    vectors_payload: dict,
    *,
    layer: int = 18,
    hd_layer: int | None = None,
    hd_axis: str = "harmfulness",
    w_r: float = -2.0,
    b_r: float = 0.5,
    w_c: float = 0.5,
    b_c: float = 0.0,
) -> AdaSteerParams:
    hd_layer = hd_layer or layer
    axes = vectors_payload["axes"]
    v_rd = axes["refusal"][layer]
    if hd_axis not in axes:
        hd_axis = "evil" if "evil" in axes else hd_axis
    v_hd = axes[hd_axis][hd_layer]
    meta = vectors_payload.get("metadata", {})
    mu = meta.get("adasteer", {}).get("mu_c_harmful", {}).get(str(layer))
    if mu is None:
        mu = meta.get("adasteer", {}).get("mu_c_harmful", {}).get(layer)
    if mu is None:
        mu = torch.zeros_like(v_rd)
    return AdaSteerParams(
        layer=layer,
        v_rd=v_rd,
        v_hd=v_hd,
        mu_c_harmful=mu if isinstance(mu, torch.Tensor) else torch.tensor(mu),
        w_r=w_r,
        b_r=b_r,
        w_c=w_c,
        b_c=b_c,
    )
