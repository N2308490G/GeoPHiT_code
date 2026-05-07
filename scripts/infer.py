"""Single-window inference helper.

Given an ``(history_len, D)`` numpy array on disk, predicts the next
``predict_len`` rows and saves the result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geophit.models import DualStreamVQGAN, HierarchicalTransformer


def _load_models(stage1_ckpt: Path, stage2_ckpt: Path, device: torch.device):
    s1 = torch.load(stage1_ckpt, map_location=device, weights_only=False)
    s2 = torch.load(stage2_ckpt, map_location=device, weights_only=False)
    cfg = s2["config"]
    model_cfg = cfg["model"]
    vqgan = DualStreamVQGAN(
        d_in=model_cfg["d_in"],
        d_out=model_cfg["d_in"],
        channels=model_cfg["encoder_channels"],
        codebook_size=model_cfg["codebook_size"],
        num_res_blocks=model_cfg["num_res_blocks"],
        beta=cfg["loss"]["beta_commit"],
    ).to(device)
    vqgan.load_state_dict(s1["model"])
    vqgan.eval()

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

    norm_stats = s2.get("norm_stats", s1.get("norm_stats"))
    return vqgan, hier, cfg, norm_stats


@torch.no_grad()
def predict(
    history: np.ndarray, stage1_ckpt: Path, stage2_ckpt: Path, device: str = "cuda"
) -> np.ndarray:
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    vqgan, hier, cfg, norm_stats = _load_models(stage1_ckpt, stage2_ckpt, dev)
    mean = np.asarray(norm_stats["mean"], dtype=np.float32)
    std = np.asarray(norm_stats["std"], dtype=np.float32)
    hist_norm = (history - mean) / (std + 1e-6)
    x_hist = torch.from_numpy(hist_norm[None]).to(dev)

    predict_len = cfg["data"]["predict_len"]
    downsample = cfg["model"].get("downsample_factor", 4)
    trend_len = max(1, predict_len // downsample)
    target_len = max(1, predict_len // downsample)

    _, s_pred = hier.generate(x_hist, trend_len=trend_len, target_len=target_len)
    x_hat_norm = vqgan.target.decode_indices(s_pred)[0].cpu().numpy()
    x_hat_norm = x_hat_norm[:predict_len]
    return x_hat_norm * std + mean


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GeoPHiT one-shot inference.")
    p.add_argument("--stage1-ckpt", type=str, required=True)
    p.add_argument("--stage2-ckpt", type=str, required=True)
    p.add_argument("--history", type=str, required=True, help=".npy file (T, D)")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    history = np.load(args.history).astype(np.float32)
    pred = predict(history, Path(args.stage1_ckpt), Path(args.stage2_ckpt), device=args.device)
    np.save(args.out, pred)
    print(f"[infer] wrote prediction shape={pred.shape} -> {args.out}")


if __name__ == "__main__":
    main()
