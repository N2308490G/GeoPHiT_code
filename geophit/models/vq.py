"""Parameter-wise vector quantization (Eq. 4 of the paper).

The continuous latent ``z_hat`` is quantized along the parameter dimension —
i.e. each (batch, time-step) location holds an ``n_z``-dim vector that is
matched against ``K`` codebook entries. This is the key design that lets each
codebook entry encode a physically consistent multi-parameter combination.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ParameterWiseVectorQuantizer(nn.Module):
    def __init__(
        self,
        codebook_size: int,
        embedding_dim: int,
        beta: float = 0.25,
    ) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.codebook = nn.Embedding(codebook_size, embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize ``z`` of shape ``(B, S, n_z)``.

        Returns:
            z_q: (B, S, n_z) discretised representation with straight-through
                gradient back to ``z``.
            indices: (B, S) long tensor of codebook indices.
            commit_loss: scalar commitment + codebook loss
                ``||sg(z) - z_q||^2 + beta * ||sg(z_q) - z||^2``.
        """
        if z.dim() != 3 or z.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"expected (B, S, {self.embedding_dim}), got {tuple(z.shape)}"
            )
        b, s, n = z.shape
        flat = z.reshape(-1, n)

        cb = self.codebook.weight  # (K, n_z)
        d = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat @ cb.t()
            + cb.pow(2).sum(dim=1)[None, :]
        )
        indices = d.argmin(dim=1)  # (B*S,)
        z_q_flat = self.codebook(indices)
        z_q = z_q_flat.view(b, s, n)

        codebook_loss = F.mse_loss(z_q, z.detach())
        commit_loss = F.mse_loss(z, z_q.detach())
        loss = codebook_loss + self.beta * commit_loss

        # straight-through estimator
        z_q = z + (z_q - z).detach()
        return z_q, indices.view(b, s), loss

    @torch.no_grad()
    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        return self.codebook(indices)
