"""1D PatchGAN-style discriminator used in the GAN term of L_stream.

The paper specifies a vanilla GAN objective (Eq. 6):
``L_GAN = E[log D(X) + log(1 - D(X_hat))]`` — i.e. binary cross-entropy on
real / fake samples (the non-saturating generator loss is used).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchDiscriminator1D(nn.Module):
    def __init__(self, d_in: int, base_channels: int = 64, num_layers: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv1d(d_in, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        ch = base_channels
        for _ in range(num_layers - 1):
            next_ch = min(ch * 2, 512)
            layers += [
                nn.Conv1d(ch, next_ch, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(min(8, next_ch), next_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = next_ch
        layers += [nn.Conv1d(ch, 1, kernel_size=3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.transpose(1, 2))


def vanilla_d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """Discriminator BCE loss matching ``-E[log D(X) + log(1 - D(X_hat))]``."""
    real_target = torch.ones_like(real_logits)
    fake_target = torch.zeros_like(fake_logits)
    return 0.5 * (
        F.binary_cross_entropy_with_logits(real_logits, real_target)
        + F.binary_cross_entropy_with_logits(fake_logits, fake_target)
    )


def vanilla_g_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """Non-saturating generator loss: ``-E[log D(X_hat)]``."""
    target = torch.ones_like(fake_logits)
    return F.binary_cross_entropy_with_logits(fake_logits, target)
