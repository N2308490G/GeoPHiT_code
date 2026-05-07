"""Geology-adaptive moving-average window utilities.

Implements Eq. (1) of the paper: gamma-ray (GR) variance within a sliding
window selects between a large window for homogeneous sandstone intervals and a
small window for heterogeneous shale intervals.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def adaptive_geological_window(
    gr_channel: torch.Tensor,
    window_sand: int = 25,
    window_shale: int = 7,
    probe_window: int = 50,
    variance_threshold: float | None = None,
) -> torch.Tensor:
    """Return per-sample integer window size based on local GR variance.

    Args:
        gr_channel: (B, T) gamma-ray sequence (normalized).
        window_sand: window for low-variance / sandstone.
        window_shale: window for high-variance / shale.
        probe_window: window over which the variance is measured. Clamped to
            the available sequence length to support short windows.
        variance_threshold: explicit threshold; if ``None`` the per-batch
            median is used.

    Returns:
        (B,) long tensor of window sizes.
    """
    if gr_channel.dim() != 2:
        raise ValueError(f"expected (B, T), got {tuple(gr_channel.shape)}")
    t = gr_channel.size(-1)
    pw = max(2, min(probe_window, t))
    var = gr_channel.unfold(-1, pw, pw).var(dim=-1).mean(dim=-1)
    if variance_threshold is None:
        variance_threshold = float(var.median().item())
    sizes = torch.where(
        var > variance_threshold,
        torch.full_like(var, window_shale, dtype=torch.long),
        torch.full_like(var, window_sand, dtype=torch.long),
    )
    return sizes.long()


def avgpool_with_window(x: torch.Tensor, window: int) -> torch.Tensor:
    """Padded 1D average pool that preserves length T (Eq. 1)."""
    if x.dim() != 3:
        raise ValueError(f"expected (B, T, D), got {tuple(x.shape)}")
    b, t, d = x.shape
    x_t = x.transpose(1, 2)
    pad_left = (window - 1) // 2
    pad_right = window - 1 - pad_left
    x_t = F.pad(x_t, (pad_left, pad_right), mode="replicate")
    x_t = F.avg_pool1d(x_t, kernel_size=window, stride=1)
    return x_t.transpose(1, 2).contiguous()


def batched_geological_downsample(
    x: torch.Tensor,
    gr_index: int,
    window_sand: int = 25,
    window_shale: int = 7,
    probe_window: int = 50,
) -> torch.Tensor:
    """Apply per-sample adaptive window over a batch.

    Falls back to a single shared window when the batch is uniform.
    """
    sizes = adaptive_geological_window(
        x[..., gr_index],
        window_sand=window_sand,
        window_shale=window_shale,
        probe_window=probe_window,
    )
    unique = torch.unique(sizes)
    if unique.numel() == 1:
        return avgpool_with_window(x, int(unique.item()))
    out = torch.empty_like(x)
    for w in unique.tolist():
        mask = sizes == w
        out[mask] = avgpool_with_window(x[mask], int(w))
    return out
