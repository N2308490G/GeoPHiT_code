"""Stage 2 trainer: hierarchical discrete-token Transformer.

Loads the frozen Stage 1 ``DualStreamVQGAN`` from a checkpoint, tokenizes the
prediction window into trend / target indices, and trains the hierarchical
decoder to autoregressively predict both token streams (Eqs. 9 and 10).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geophit.data import build_dataloaders
from geophit.losses.physics import CHANNEL_INDEX
from geophit.models import DualStreamVQGAN, HierarchicalTransformer
from geophit.utils import (
    batched_geological_downsample,
    deep_update,
    load_yaml,
    parse_dotted_overrides,
    set_seed,
)


def _load_stage1(ckpt_path: Path, device: torch.device) -> tuple[DualStreamVQGAN, dict]:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state["config"]
    model_cfg = cfg["model"]
    vqgan = DualStreamVQGAN(
        d_in=model_cfg["d_in"],
        d_out=model_cfg["d_in"],
        channels=model_cfg["encoder_channels"],
        codebook_size=model_cfg["codebook_size"],
        beta=cfg["loss"]["beta_commit"],
    ).to(device)
    vqgan.load_state_dict(state["model"])
    for p in vqgan.parameters():
        p.requires_grad_(False)
    vqgan.eval()
    return vqgan, state


def train(cfg: dict, stage1_ckpt: Path) -> Path:
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    set_seed(cfg["seed"])

    loaders = build_dataloaders(cfg)
    train_loader: DataLoader = loaders["train"]
    val_loader: DataLoader = loaders["val"]

    vqgan, state = _load_stage1(stage1_ckpt, device)

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

    opt = torch.optim.Adam(
        hier.parameters(),
        lr=cfg["train"]["lr_stage2"],
        betas=tuple(cfg["train"]["betas"]),
    )

    geo_sand = model_cfg["geo_window_sand"]
    geo_shale = model_cfg["geo_window_shale"]
    geo_probe = model_cfg.get("geo_probe_window", 50)
    gr_index = CHANNEL_INDEX["GR"]
    log_interval = cfg["train"].get("log_interval", 50)
    epochs = cfg["train"]["epochs_stage2"]

    ckpt_dir = Path(cfg["train"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "stage2.pt"
    best_val = float("inf")

    for epoch in range(epochs):
        hier.train()
        epoch_start = time.time()
        running = {"loss": 0.0, "base": 0.0, "sc": 0.0}

        for step, batch in enumerate(train_loader):
            x_hist = batch["history"].to(device)
            x_pred = batch["predict"].to(device)
            x_down = batched_geological_downsample(
                x_pred,
                gr_index=gr_index,
                window_sand=geo_sand,
                window_shale=geo_shale,
                probe_window=geo_probe,
            )

            with torch.no_grad():
                trend_idx = vqgan.trend.encode_indices(x_down)
                target_idx = vqgan.target.encode_indices(x_pred)

            base_logits, sc_logits = hier(x_hist, trend_idx, target_idx)
            base_loss = F.cross_entropy(
                base_logits.reshape(-1, base_logits.size(-1)),
                trend_idx.reshape(-1),
            )
            sc_loss = F.cross_entropy(
                sc_logits.reshape(-1, sc_logits.size(-1)),
                target_idx.reshape(-1),
            )
            loss = base_loss + sc_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(hier.parameters(), max_norm=5.0)
            opt.step()

            running["loss"] += loss.item()
            running["base"] += base_loss.item()
            running["sc"] += sc_loss.item()

            if (step + 1) % log_interval == 0:
                msg = " ".join(
                    f"{k}={v / (step + 1):.4f}" for k, v in running.items()
                )
                print(f"[stage2][epoch {epoch} step {step + 1}] {msg}")

        val_loss = _validate(
            hier, vqgan, val_loader, device, gr_index, geo_sand, geo_shale, geo_probe
        )
        elapsed = time.time() - epoch_start
        print(
            f"[stage2][epoch {epoch}] train_loss={running['loss'] / max(1, len(train_loader)):.4f} "
            f"val_loss={val_loss:.4f} time={elapsed:.1f}s"
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "hier": hier.state_dict(),
                    "stage1_ckpt": str(stage1_ckpt),
                    "norm_stats": state.get("norm_stats", loaders["norm_stats"]),
                    "config": cfg,
                    "epoch": epoch,
                },
                ckpt_path,
            )
            print(f"[stage2] saved checkpoint -> {ckpt_path}")
    return ckpt_path


@torch.no_grad()
def _validate(
    hier, vqgan, loader, device, gr_index, geo_sand, geo_shale, geo_probe
) -> float:
    hier.eval()
    total = 0.0
    n = 0
    for batch in loader:
        x_hist = batch["history"].to(device)
        x_pred = batch["predict"].to(device)
        x_down = batched_geological_downsample(
            x_pred,
            gr_index=gr_index,
            window_sand=geo_sand,
            window_shale=geo_shale,
            probe_window=geo_probe,
        )
        trend_idx = vqgan.trend.encode_indices(x_down)
        target_idx = vqgan.target.encode_indices(x_pred)
        base_logits, sc_logits = hier(x_hist, trend_idx, target_idx)
        loss = F.cross_entropy(
            base_logits.reshape(-1, base_logits.size(-1)), trend_idx.reshape(-1)
        ) + F.cross_entropy(
            sc_logits.reshape(-1, sc_logits.size(-1)), target_idx.reshape(-1)
        )
        total += loss.item() * x_hist.size(0)
        n += x_hist.size(0)
    return total / max(1, n)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Stage 2 of GeoPHiT.")
    p.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs/default.yaml"))
    p.add_argument("--stage1-ckpt", type=str, required=True)
    p.add_argument("--override", nargs="*", default=[])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    deep_update(cfg, parse_dotted_overrides(args.override))
    train(cfg, Path(args.stage1_ckpt))


if __name__ == "__main__":
    main()
