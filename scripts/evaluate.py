"""End-to-end evaluation of a trained GeoPHiT.

Loads Stage 1 + Stage 2 checkpoints, runs greedy autoregressive prediction
over the test loader, and reports RMSE / MAE / MAPE / GCI / MRPR / CVR.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geophit.data import build_dataloaders
from geophit.eval import evaluate_all
from geophit.models import DualStreamVQGAN, HierarchicalTransformer
from geophit.utils import deep_update, load_yaml, parse_dotted_overrides, set_seed


def _load_dual_vqgan(state, device) -> DualStreamVQGAN:
    cfg = state["config"]
    model_cfg = cfg["model"]
    m = DualStreamVQGAN(
        d_in=model_cfg["d_in"],
        d_out=model_cfg["d_in"],
        channels=model_cfg["encoder_channels"],
        codebook_size=model_cfg["codebook_size"],
        beta=cfg["loss"]["beta_commit"],
    ).to(device)
    m.load_state_dict(state["model"])
    m.eval()
    return m


@torch.no_grad()
def evaluate(stage1_ckpt: Path, stage2_ckpt: Path, cfg_override: dict | None = None) -> dict:
    s2 = torch.load(stage2_ckpt, map_location="cpu", weights_only=False)
    cfg = s2["config"]
    if cfg_override:
        deep_update(cfg, cfg_override)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    set_seed(cfg["seed"])

    s1 = torch.load(stage1_ckpt, map_location=device, weights_only=False)
    vqgan = _load_dual_vqgan(s1, device)

    model_cfg = cfg["model"]
    hier = HierarchicalTransformer(
        d_in=model_cfg["d_in"],
        codebook_size_target=model_cfg["codebook_size"],
        codebook_size_trend=model_cfg["codebook_size"],
        d_model=model_cfg["d_model"],
        num_heads=model_cfg["num_heads"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
    ).to(device)
    hier.load_state_dict(s2["hier"])
    hier.eval()

    loaders = build_dataloaders(cfg)
    test_loader = loaders["test"]
    norm_stats = loaders["norm_stats"]

    predict_len = cfg["data"]["predict_len"]
    downsample = model_cfg.get("downsample_factor", 4)
    trend_len = max(1, predict_len // downsample)
    target_len = max(1, predict_len // downsample)

    preds_all, targets_all = [], []
    for batch in test_loader:
        x_hist = batch["history"].to(device)
        x_pred = batch["predict"].to(device)
        _, s_pred = hier.generate(
            x_hist, trend_len=trend_len, target_len=target_len, greedy=True
        )
        x_hat = vqgan.target.decode_indices(s_pred)
        # the decoder upsamples by 4x, so x_hat ~ predict_len; trim/pad to match
        if x_hat.size(1) > x_pred.size(1):
            x_hat = x_hat[:, : x_pred.size(1)]
        elif x_hat.size(1) < x_pred.size(1):
            pad = x_pred.size(1) - x_hat.size(1)
            x_hat = torch.nn.functional.pad(x_hat.transpose(1, 2), (0, pad)).transpose(1, 2)
        preds_all.append(x_hat.cpu())
        targets_all.append(x_pred.cpu())

    pred = torch.cat(preds_all, dim=0)
    target = torch.cat(targets_all, dim=0)
    metrics = evaluate_all(
        pred,
        target,
        norm_stats={
            "mean": torch.as_tensor(norm_stats["mean"]).float(),
            "std": torch.as_tensor(norm_stats["std"]).float(),
        },
        archie_a=cfg["loss"]["archie_a"],
        archie_m=cfg["loss"]["archie_m"],
        cvr_threshold=cfg["loss"]["cvr_threshold"],
    )
    print("[evaluate] " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained GeoPHiT.")
    p.add_argument("--stage1-ckpt", type=str, required=True)
    p.add_argument("--stage2-ckpt", type=str, required=True)
    p.add_argument("--override", nargs="*", default=[])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    overrides = parse_dotted_overrides(args.override)
    evaluate(Path(args.stage1_ckpt), Path(args.stage2_ckpt), cfg_override=overrides or None)


if __name__ == "__main__":
    main()
