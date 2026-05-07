"""Encoder/decoder backbones for the Stage 1 VQGAN.

Strictly follows §3.3.2 of the paper: the encoder consists of *four* residual
blocks with channel widths ``[64, 128, 256, n_z]`` and two stride-2
downsampling stages, producing a latent of shape ``(B, T // 4, n_z)``. The
decoder mirrors the encoder using transpose convolutions.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_groups: int = 8) -> None:
        super().__init__()
        groups_in = min(num_groups, in_ch) if in_ch >= num_groups else 1
        groups_out = min(num_groups, out_ch) if out_ch >= num_groups else 1
        self.norm1 = nn.GroupNorm(groups_in, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups_out, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class Conv1DEncoder(nn.Module):
    """Four-block residual encoder with two stride-2 downsamples.

    Block widths: ``c1 -> c2 -> c3 -> n_z`` with the two stride-2 downsamples
    occurring as 1D convolutions inserted between blocks 1↔2 and 2↔3, so the
    spatial length reduces by a factor of 4.
    """

    def __init__(
        self,
        d_in: int,
        channels: Sequence[int] = (64, 128, 256, 64),
    ) -> None:
        super().__init__()
        if len(channels) != 4:
            raise ValueError("expected exactly 4 channel widths")
        c1, c2, c3, n_z = channels
        self.in_proj = nn.Conv1d(d_in, c1, kernel_size=3, padding=1)
        self.block1 = ResidualBlock1D(c1, c1)
        self.down1 = nn.Conv1d(c1, c2, kernel_size=4, stride=2, padding=1)
        self.block2 = ResidualBlock1D(c2, c2)
        self.down2 = nn.Conv1d(c2, c3, kernel_size=4, stride=2, padding=1)
        self.block3 = ResidualBlock1D(c3, c3)
        self.block4 = ResidualBlock1D(c3, n_z)
        self.n_z = n_z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected (B, T, D), got {tuple(x.shape)}")
        h = x.transpose(1, 2)
        h = self.in_proj(h)
        h = self.block1(h)
        h = self.down1(h)
        h = self.block2(h)
        h = self.down2(h)
        h = self.block3(h)
        h = self.block4(h)
        return h.transpose(1, 2).contiguous()


class Conv1DDecoder(nn.Module):
    """Mirror of ``Conv1DEncoder`` using transpose convolutions."""

    def __init__(
        self,
        d_out: int,
        channels: Sequence[int] = (64, 128, 256, 64),
    ) -> None:
        super().__init__()
        if len(channels) != 4:
            raise ValueError("expected exactly 4 channel widths")
        c1, c2, c3, n_z = channels
        self.block4 = ResidualBlock1D(n_z, c3)
        self.block3 = ResidualBlock1D(c3, c3)
        self.up2 = nn.ConvTranspose1d(c3, c2, kernel_size=4, stride=2, padding=1)
        self.block2 = ResidualBlock1D(c2, c2)
        self.up1 = nn.ConvTranspose1d(c2, c1, kernel_size=4, stride=2, padding=1)
        self.block1 = ResidualBlock1D(c1, c1)
        self.out_proj = nn.Conv1d(c1, d_out, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 3:
            raise ValueError(f"expected (B, S, n_z), got {tuple(z.shape)}")
        h = z.transpose(1, 2)
        h = self.block4(h)
        h = self.block3(h)
        h = self.up2(h)
        h = self.block2(h)
        h = self.up1(h)
        h = self.block1(h)
        h = self.out_proj(h)
        return h.transpose(1, 2).contiguous()
