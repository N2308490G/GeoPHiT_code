"""GeoPHiT command-line entry point.

A thin dispatcher over the four pipeline stages so users only need to remember
``python main.py <command> ...`` instead of locating individual scripts.

Examples
--------
Train Stage 1 with the default config and synthetic data::

    python main.py train-stage1

Train Stage 2 from an existing Stage 1 checkpoint::

    python main.py train-stage2 --stage1-ckpt ./checkpoints/stage1.pt

Evaluate the full pipeline::

    python main.py evaluate \
        --stage1-ckpt ./checkpoints/stage1.pt \
        --stage2-ckpt ./checkpoints/stage2.pt

Run Stage 1 + Stage 2 + evaluation back-to-back::

    python main.py train-all

Predict the next 100 steps from a saved history window::

    python main.py infer \
        --stage1-ckpt ./checkpoints/stage1.pt \
        --stage2-ckpt ./checkpoints/stage2.pt \
        --history history.npy --out prediction.npy

Override any config field via ``--override key=value`` (dotted paths supported)::

    python main.py train-stage1 --override train.batch_size=8 seed=1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geophit.utils import deep_update, load_yaml, parse_dotted_overrides
from scripts.evaluate import evaluate as run_evaluate
from scripts.infer import predict as run_predict
from scripts.train_stage1 import train as run_train_stage1
from scripts.train_stage2 import train as run_train_stage2


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geophit",
        description=(
            "GeoPHiT: Physics-Informed Hierarchical Discrete Transformer for "
            "Multi-Scale Geological Sequence Prediction."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def _add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--config",
            type=str,
            default=str(DEFAULT_CONFIG),
            help="path to a YAML config file (default: configs/default.yaml)",
        )
        p.add_argument(
            "--override",
            nargs="*",
            default=[],
            help="dotted-path overrides, e.g. train.batch_size=8 seed=1",
        )

    s1 = sub.add_parser("train-stage1", help="train the Stage 1 dual-stream VQGAN")
    _add_common(s1)

    s2 = sub.add_parser("train-stage2", help="train the Stage 2 hierarchical Transformer")
    _add_common(s2)
    s2.add_argument("--stage1-ckpt", type=str, required=True)

    full = sub.add_parser(
        "train-all",
        help="run Stage 1 + Stage 2 + evaluation in sequence",
    )
    _add_common(full)

    ev = sub.add_parser("evaluate", help="evaluate a trained GeoPHiT")
    _add_common(ev)
    ev.add_argument("--stage1-ckpt", type=str, required=True)
    ev.add_argument("--stage2-ckpt", type=str, required=True)

    inf = sub.add_parser("infer", help="predict the next horizon from a history window")
    inf.add_argument("--stage1-ckpt", type=str, required=True)
    inf.add_argument("--stage2-ckpt", type=str, required=True)
    inf.add_argument("--history", type=str, required=True, help=".npy file (T, D)")
    inf.add_argument("--out", type=str, required=True)
    inf.add_argument("--device", type=str, default="cuda")

    return parser


def _load_cfg(args: argparse.Namespace) -> dict:
    cfg = load_yaml(args.config)
    deep_update(cfg, parse_dotted_overrides(args.override))
    return cfg


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "train-stage1":
        cfg = _load_cfg(args)
        run_train_stage1(cfg)
    elif args.command == "train-stage2":
        cfg = _load_cfg(args)
        run_train_stage2(cfg, Path(args.stage1_ckpt))
    elif args.command == "train-all":
        cfg = _load_cfg(args)
        s1 = run_train_stage1(cfg)
        s2 = run_train_stage2(cfg, Path(s1))
        run_evaluate(Path(s1), Path(s2))
    elif args.command == "evaluate":
        overrides = parse_dotted_overrides(args.override)
        run_evaluate(
            Path(args.stage1_ckpt),
            Path(args.stage2_ckpt),
            cfg_override=overrides or None,
        )
    elif args.command == "infer":
        import numpy as np

        history = np.load(args.history).astype(np.float32)
        pred = run_predict(
            history,
            Path(args.stage1_ckpt),
            Path(args.stage2_ckpt),
            device=args.device,
        )
        np.save(args.out, pred)
        print(f"[infer] wrote prediction shape={pred.shape} -> {args.out}")
    else:  # pragma: no cover - argparse rejects unknown commands
        parser.error(f"unknown command: {args.command!r}")


if __name__ == "__main__":
    main()
