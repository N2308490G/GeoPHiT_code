"""Stage 1 trainer: dual-stream physics-aware VQGAN.

Implements ``L_stream = L_rec + L_commit + lambda_GAN * L_GAN + alpha * L_physics``
for both the target and trend streams. Adversarial loss is enabled after a
warm-up of ``gan_warmup_ratio * total_epochs`` (Section 3.4 of the paper).
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
from geophit.losses import PhysicsLoss, vanilla_d_loss, vanilla_g_loss
from geophit.losses.physics import CHANNEL_INDEX, PhysicsConstants
from geophit.models import DualStreamVQGAN, PatchDiscriminator1D
from geophit.utils import (
    batched_geological_downsample,
    deep_update,
    load_yaml,
    parse_dotted_overrides,
    set_seed,
)


def _move_norm_stats(stats: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "mean": torch.as_tensor(stats["mean"], device=device).float(),
        "std": torch.as_tensor(stats["std"], device=device).float(),
    }


def _stream_loss(
    x: torch.Tensor,
    out,
    physics_loss: PhysicsLoss,
    discriminator: PatchDiscriminator1D | None,
    alpha_physics: float,
    lambda_gan: float,
    gan_active: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    rec = F.mse_loss(out.x_hat, x)
    phys = physics_loss(out.x_hat)
    loss = rec + out.vq_loss + alpha_physics * phys
    log = {
        "rec": rec.item(),
        "vq": out.vq_loss.item(),
        "phys": phys.item(),
    }
    if gan_active and discriminator is not None:
        fake_logits = discriminator(out.x_hat)
        g_loss = vanilla_g_loss(fake_logits)
        loss = loss + lambda_gan * g_loss
        log["g_loss"] = g_loss.item()
    return loss, log


def train(cfg: dict) -> Path:
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    set_seed(cfg["seed"])

    loaders = build_dataloaders(cfg)
    train_loader: DataLoader = loaders["train"]
    val_loader: DataLoader = loaders["val"]
    norm_stats = _move_norm_stats(loaders["norm_stats"], device)

    model_cfg = cfg["model"]
    model = DualStreamVQGAN(
        d_in=model_cfg["d_in"],
        d_out=model_cfg["d_in"],
        channels=model_cfg["encoder_channels"],
        codebook_size=model_cfg["codebook_size"],
        beta=cfg["loss"]["beta_commit"],
    ).to(device)

    disc_target = PatchDiscriminator1D(d_in=model_cfg["d_in"]).to(device)
    disc_trend = PatchDiscriminator1D(d_in=model_cfg["d_in"]).to(device)

    physics = PhysicsLoss(
        constants=PhysicsConstants(
            archie_a=cfg["loss"]["archie_a"],
            archie_m=cfg["loss"]["archie_m"],
            gardner_rho0=cfg["loss"]["gardner_rho0"],
        ),
        norm_stats=norm_stats,
    ).to(device)

    g_params = list(model.parameters())
    d_params = list(disc_target.parameters()) + list(disc_trend.parameters())

    g_opt = torch.optim.Adam(
        g_params, lr=cfg["train"]["lr_stage1"], betas=tuple(cfg["train"]["betas"])
    )
    d_opt = torch.optim.Adam(
        d_params, lr=cfg["train"]["lr_stage1"], betas=tuple(cfg["train"]["betas"])
    )

    epochs = cfg["train"]["epochs_stage1"]
    gan_start = int(cfg["loss"]["gan_warmup_ratio"] * epochs)
    geo_sand = model_cfg["geo_window_sand"]
    geo_shale = model_cfg["geo_window_shale"]
    geo_probe = model_cfg.get("geo_probe_window", 50)
    gr_index = CHANNEL_INDEX["GR"]
    log_interval = cfg["train"].get("log_interval", 50)

    ckpt_dir = Path(cfg["train"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "stage1.pt"
    best_val = float("inf")

    for epoch in range(epochs):
        gan_active = epoch >= gan_start
        model.train()
        disc_target.train(gan_active)
        disc_trend.train(gan_active)
        epoch_start = time.time()
        running: dict[str, float] = {}

        for step, batch in enumerate(train_loader):
            x_pred = batch["predict"].to(device)
            x_down = batched_geological_downsample(
                x_pred,
                gr_index=gr_index,
                window_sand=geo_sand,
                window_shale=geo_shale,
                probe_window=geo_probe,
            )

            if gan_active:
                with torch.no_grad():
                    out_t = model.target(x_pred)
                    out_d = model.trend(x_down)
                d_real_t = disc_target(x_pred)
                d_fake_t = disc_target(out_t.x_hat.detach())
                d_real_d = disc_trend(x_down)
                d_fake_d = disc_trend(out_d.x_hat.detach())
                d_loss = vanilla_d_loss(d_real_t, d_fake_t) + vanilla_d_loss(
                    d_real_d, d_fake_d
                )
                d_opt.zero_grad(set_to_none=True)
                d_loss.backward()
                d_opt.step()
                running["d_loss"] = running.get("d_loss", 0.0) + d_loss.item()

            outs = model(x_pred=x_pred, x_down=x_down)
            loss_t, log_t = _stream_loss(
                x_pred,
                outs["target"],
                physics,
                disc_target,
                cfg["loss"]["alpha_physics"],
                cfg["loss"]["lambda_gan"],
                gan_active,
            )
            loss_d, log_d = _stream_loss(
                x_down,
                outs["trend"],
                physics,
                disc_trend,
                cfg["loss"]["alpha_physics"],
                cfg["loss"]["lambda_gan"],
                gan_active,
            )
            loss = loss_t + loss_d
            g_opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(g_params, max_norm=5.0)
            g_opt.step()

            for k, v in log_t.items():
                running[f"t/{k}"] = running.get(f"t/{k}", 0.0) + v
            for k, v in log_d.items():
                running[f"d/{k}"] = running.get(f"d/{k}", 0.0) + v
            running["loss"] = running.get("loss", 0.0) + loss.item()

            if (step + 1) % log_interval == 0:
                msg = " ".join(
                    f"{k}={v / (step + 1):.4f}" for k, v in sorted(running.items())
                )
                print(f"[stage1][epoch {epoch} step {step + 1}] {msg}")

        val_loss = _validate(
            model, val_loader, device, gr_index, geo_sand, geo_shale, geo_probe
        )
        elapsed = time.time() - epoch_start
        print(
            f"[stage1][epoch {epoch}] train_loss={running['loss'] / max(1, len(train_loader)):.4f} "
            f"val_rec={val_loss:.4f} time={elapsed:.1f}s gan={gan_active}"
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "disc_target": disc_target.state_dict(),
                    "disc_trend": disc_trend.state_dict(),
                    "norm_stats": loaders["norm_stats"],
                    "config": cfg,
                    "epoch": epoch,
                },
                ckpt_path,
            )
            print(f"[stage1] saved checkpoint -> {ckpt_path}")

    return ckpt_path


@torch.no_grad()
def _validate(model, loader, device, gr_index, geo_sand, geo_shale, geo_probe) -> float:
    model.eval()
    total_rec = 0.0
    n = 0
    for batch in loader:
        x_pred = batch["predict"].to(device)
        x_down = batched_geological_downsample(
            x_pred,
            gr_index=gr_index,
            window_sand=geo_sand,
            window_shale=geo_shale,
            probe_window=geo_probe,
        )
        outs = model(x_pred=x_pred, x_down=x_down)
        rec = F.mse_loss(outs["target"].x_hat, x_pred) + F.mse_loss(
            outs["trend"].x_hat, x_down
        )
        total_rec += rec.item() * x_pred.size(0)
        n += x_pred.size(0)
    return total_rec / max(1, n)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Stage 1 of GeoPHiT.")
    p.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs/default.yaml"))
    p.add_argument("--override", nargs="*", default=[], help="dotted overrides like train.batch_size=8")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    deep_update(cfg, parse_dotted_overrides(args.override))
    train(cfg)


if __name__ == "__main__":
    main()
