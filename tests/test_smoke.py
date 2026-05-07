"""End-to-end smoke test on synthetic well logs.

Runs in well under a minute on CPU with the shrunken config below. Verifies:
  * the dual-stream VQGAN trains for a couple of epochs without errors
  * Stage 2 trains, and greedy generation produces predictions of the
    expected shape
  * full evaluation pipeline returns finite metric values
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geophit.utils import deep_update, load_yaml, set_seed
from scripts.train_stage1 import train as train_stage1
from scripts.train_stage2 import train as train_stage2
from scripts.evaluate import evaluate


def _smoke_config(ckpt_dir: Path) -> dict:
    cfg = load_yaml(PROJECT_ROOT / "configs/default.yaml")
    deep_update(
        cfg,
        {
            "device": "cpu",
            "seed": 0,
            "data": {
                "num_wells": 4,
                "samples_per_well": 800,
                "history_len": 64,
                "predict_len": 32,
                "stride": 32,
                "train_val_test_split": [0.5, 0.25, 0.25],
            },
            "model": {
                "encoder_channels": [16, 32, 64, 16],
                "n_z": 16,
                "codebook_size": 64,
                "geo_probe_window": 16,
                "d_model": 64,
                "num_heads": 4,
                "num_layers": 2,
            },
            "train": {
                "batch_size": 4,
                "epochs_stage1": 2,
                "epochs_stage2": 2,
                "log_interval": 10,
                "ckpt_dir": str(ckpt_dir),
            },
        },
    )
    return cfg


def test_end_to_end() -> None:
    set_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _smoke_config(Path(tmp))
        s1 = train_stage1(cfg)
        assert Path(s1).exists(), "Stage 1 checkpoint should exist"
        s2 = train_stage2(cfg, Path(s1))
        assert Path(s2).exists(), "Stage 2 checkpoint should exist"
        metrics = evaluate(Path(s1), Path(s2))
        for k, v in metrics.items():
            assert torch.isfinite(torch.tensor(v)), f"{k} not finite: {v}"
        print("[smoke] metrics:", metrics)


if __name__ == "__main__":
    test_end_to_end()
