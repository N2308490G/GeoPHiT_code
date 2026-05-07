"""Physics-aware loss (Eq. 7 of the paper).

Implements the dual constraint
``L_physics = || X_hat_RT - a * X_hat_NPHI^{-m} ||^2
              + || X_hat_DEN - rho0 * (1 - X_hat_DPHI) ||^2``.

The reconstructed sequences are produced in the standardized (z-score) space
that the model is trained in, so the physical relationships have to be
evaluated after de-normalization. ``PhysicsLoss`` accepts per-channel
``mean``/``std`` tensors and converts back internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn


def denormalize_channel(
    x_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """``x_norm * std + mean`` broadcast over (B, T, D)."""
    while mean.dim() < x_norm.dim():
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    return x_norm * std + mean


CHANNEL_INDEX = {"GR": 0, "RT": 1, "NPHI": 2, "DPHI": 3, "DEN": 4}


@dataclass
class PhysicsConstants:
    archie_a: float = 1.0
    archie_m: float = 2.0
    gardner_rho0: float = 2.65


class PhysicsLoss(nn.Module):
    """Soft Archie + Gardner constraints on standardized reconstructions."""

    def __init__(
        self,
        constants: PhysicsConstants,
        norm_stats: Mapping[str, torch.Tensor] | None = None,
        clamp_eps: float = 1e-3,
    ) -> None:
        super().__init__()
        self.constants = constants
        self.clamp_eps = clamp_eps
        if norm_stats is not None:
            self.register_buffer("mean", norm_stats["mean"].clone().float())
            self.register_buffer("std", norm_stats["std"].clone().float())
        else:
            self.mean = None
            self.std = None

    def set_norm_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.register_buffer("mean", mean.clone().float())
        self.register_buffer("std", std.clone().float())

    def _to_physical(self, x_norm: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            return x_norm
        return denormalize_channel(x_norm, self.mean, self.std)

    def forward(self, x_hat: torch.Tensor) -> torch.Tensor:
        """Eq. 7: ``||X_hat_RT - a*X_hat_NPHI^{-m}||^2 +
        ||X_hat_DEN - rho_0*(1 - X_hat_DPHI)||^2``.

        Evaluated in physical (de-normalized) space. ``NPHI`` is clamped to a
        small positive lower bound so the negative power stays finite.
        """
        if x_hat.dim() != 3 or x_hat.shape[-1] < 5:
            raise ValueError(
                f"expected (B, T, >=5) with channels {list(CHANNEL_INDEX)}, got "
                f"{tuple(x_hat.shape)}"
            )
        x_phys = self._to_physical(x_hat)
        rt = x_phys[..., CHANNEL_INDEX["RT"]]
        nphi = x_phys[..., CHANNEL_INDEX["NPHI"]].clamp(min=self.clamp_eps)
        dphi = x_phys[..., CHANNEL_INDEX["DPHI"]]
        den = x_phys[..., CHANNEL_INDEX["DEN"]]

        archie_pred = self.constants.archie_a * nphi.pow(-self.constants.archie_m)
        gardner_pred = self.constants.gardner_rho0 * (1.0 - dphi)

        archie_term = (rt - archie_pred).pow(2).mean()
        gardner_term = (den - gardner_pred).pow(2).mean()
        return archie_term + gardner_term
