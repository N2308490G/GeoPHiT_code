"""Stage 1 dual-stream VQGAN.

Two parallel ``StreamVQGAN`` instances handle the high-resolution target
sequence and its geology-aware downsampled trend. They share the same
architecture template but each owns its own codebook ``Z_t`` / ``Z_d``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from .blocks import Conv1DDecoder, Conv1DEncoder
from .vq import ParameterWiseVectorQuantizer


@dataclass
class StreamOutput:
    x_hat: torch.Tensor
    z: torch.Tensor
    z_q: torch.Tensor
    indices: torch.Tensor
    vq_loss: torch.Tensor


class StreamVQGAN(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        channels: Sequence[int],
        codebook_size: int,
        beta: float = 0.25,
    ) -> None:
        super().__init__()
        self.encoder = Conv1DEncoder(d_in=d_in, channels=channels)
        self.quantizer = ParameterWiseVectorQuantizer(
            codebook_size=codebook_size,
            embedding_dim=channels[-1],
            beta=beta,
        )
        self.decoder = Conv1DDecoder(d_out=d_out, channels=channels)

    def forward(self, x: torch.Tensor) -> StreamOutput:
        z = self.encoder(x)
        z_q, indices, vq_loss = self.quantizer(z)
        x_hat = self.decoder(z_q)
        return StreamOutput(x_hat=x_hat, z=z, z_q=z_q, indices=indices, vq_loss=vq_loss)

    @torch.no_grad()
    def encode_indices(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        _, indices, _ = self.quantizer(z)
        return indices

    @torch.no_grad()
    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        z_q = self.quantizer.lookup(indices)
        return self.decoder(z_q)


class DualStreamVQGAN(nn.Module):
    """Pair of ``StreamVQGAN`` modules for target and trend streams."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        channels: Sequence[int],
        codebook_size: int,
        beta: float = 0.25,
    ) -> None:
        super().__init__()
        self.target = StreamVQGAN(
            d_in=d_in,
            d_out=d_out,
            channels=channels,
            codebook_size=codebook_size,
            beta=beta,
        )
        self.trend = StreamVQGAN(
            d_in=d_in,
            d_out=d_out,
            channels=channels,
            codebook_size=codebook_size,
            beta=beta,
        )

    def forward(
        self, x_pred: torch.Tensor, x_down: torch.Tensor
    ) -> dict[str, StreamOutput]:
        return {
            "target": self.target(x_pred),
            "trend": self.trend(x_down),
        }
