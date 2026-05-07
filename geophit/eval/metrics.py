"""Evaluation metrics: RMSE, MAE, MAPE, GCI, MRPR, CVR.

GCI follows the paper (Sec. 4.3): ``exp(-|r_pred - r_target|)`` between
predicted and ground-truth Pearson correlations of (log-RT) vs NPHI.
MRPR / CVR follow Eqs. (10) and (11).
"""

from __future__ import annotations

from typing import Mapping

import torch

from ..losses.physics import CHANNEL_INDEX, denormalize_channel


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2))


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target).abs())


def mape(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.mean(((pred - target) / target.abs().clamp(min=eps)).abs())


def _pearson_r(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_flat = a.flatten()
    b_flat = b.flatten()
    a_c = a_flat - a_flat.mean()
    b_c = b_flat - b_flat.mean()
    denom = (a_c.norm() * b_c.norm()).clamp(min=1e-8)
    return (a_c * b_c).sum() / denom


def geological_consistency_index(
    pred_phys: torch.Tensor, target_phys: torch.Tensor
) -> torch.Tensor:
    """GCI = exp(-|r_pred - r_target|) between log(RT) and NPHI."""
    rt_p = pred_phys[..., CHANNEL_INDEX["RT"]].clamp(min=1e-3).log()
    rt_t = target_phys[..., CHANNEL_INDEX["RT"]].clamp(min=1e-3).log()
    nphi_p = pred_phys[..., CHANNEL_INDEX["NPHI"]]
    nphi_t = target_phys[..., CHANNEL_INDEX["NPHI"]]
    r_pred = _pearson_r(rt_p, nphi_p)
    r_target = _pearson_r(rt_t, nphi_t)
    return torch.exp(-(r_pred - r_target).abs())


def mean_relative_physical_residual(
    pred_phys: torch.Tensor,
    archie_a: float = 1.0,
    archie_m: float = 2.0,
    eps: float = 1e-3,
) -> torch.Tensor:
    rt = pred_phys[..., CHANNEL_INDEX["RT"]]
    nphi = pred_phys[..., CHANNEL_INDEX["NPHI"]].clamp(min=eps)
    archie_pred = archie_a * nphi.pow(-archie_m)
    return ((rt - archie_pred).abs() / archie_pred.abs().clamp(min=eps)).mean()


def constraint_violation_rate(
    pred_phys: torch.Tensor,
    archie_a: float = 1.0,
    archie_m: float = 2.0,
    threshold: float = 0.2,
    eps: float = 1e-3,
) -> torch.Tensor:
    rt = pred_phys[..., CHANNEL_INDEX["RT"]]
    nphi = pred_phys[..., CHANNEL_INDEX["NPHI"]].clamp(min=eps)
    archie_pred = archie_a * nphi.pow(-archie_m)
    rel = (rt - archie_pred).abs() / archie_pred.abs().clamp(min=eps)
    return (rel > threshold).float().mean()


def evaluate_all(
    pred: torch.Tensor,
    target: torch.Tensor,
    norm_stats: Mapping[str, torch.Tensor] | None = None,
    archie_a: float = 1.0,
    archie_m: float = 2.0,
    cvr_threshold: float = 0.2,
) -> dict[str, float]:
    """Compute the full metric panel.

    Inputs are standardized; physics-based metrics are evaluated in physical
    units after de-normalization (when ``norm_stats`` is provided).
    """
    metrics = {
        "RMSE": rmse(pred, target).item(),
        "MAE": mae(pred, target).item(),
        "MAPE": mape(pred, target).item(),
    }
    if norm_stats is not None:
        mean = norm_stats["mean"].to(pred.device).float()
        std = norm_stats["std"].to(pred.device).float()
        pred_phys = denormalize_channel(pred, mean, std)
        target_phys = denormalize_channel(target, mean, std)
    else:
        pred_phys, target_phys = pred, target
    metrics["GCI"] = geological_consistency_index(pred_phys, target_phys).item()
    metrics["MRPR"] = mean_relative_physical_residual(
        pred_phys, archie_a=archie_a, archie_m=archie_m
    ).item()
    metrics["CVR"] = constraint_violation_rate(
        pred_phys,
        archie_a=archie_a,
        archie_m=archie_m,
        threshold=cvr_threshold,
    ).item()
    return metrics
