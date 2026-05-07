# GeoPHiT

**Physics-Informed Hierarchical Discrete Transformer for Multi-Scale
Geological Sequence Prediction.**

![Status](https://img.shields.io/badge/paper-under%20review-orange.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-red.svg)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Status — under double-blind review.** This repository accompanies the
> paper *"GeoPHiT: A Physics-Informed Hierarchical Discrete Transformer for
> Multi-Scale Geological Sequence Prediction"*, currently under peer review at
> *Information Fusion* (Elsevier). All author names, affiliations, funding
> identifiers, and other identifying information have been removed from this
> repository to comply with the journal's double-blind review policy. A
> de-anonymised, frozen release will be tagged after the paper is accepted.
> Reviewers are kindly asked **not** to attempt to deanonymise the authors via
> repository metadata, commit history, or external links.

---

## Table of contents

1. [Overview](#overview)
2. [Method at a glance](#method-at-a-glance)
3. [Repository layout](#repository-layout)
4. [Installation](#installation)
5. [Quick start](#quick-start)
6. [Datasets](#datasets)
7. [Configuration reference](#configuration-reference)
8. [Reproducing the metric panel](#reproducing-the-metric-panel)
9. [Citation](#citation)
10. [License](#license)
11. [Acknowledgements](#acknowledgements)

---

## Overview

Accurate multi-scale processing of geological signals is a core bottleneck for
high-risk industrial drilling. Existing deep models capture long-range
sequence patterns well but routinely produce predictions that violate
petrophysical laws such as Archie's relation (resistivity vs porosity) and
Gardner's relation (density vs porosity). GeoPHiT closes this gap with two
ideas:

1. **Parameter-wise vector quantization.** Each codebook entry encodes a
   physically consistent multi-channel tuple `(RT, NPHI, DPHI, GR)` at a
   single depth, embedding petrophysical constraints *at the representation
   level* rather than only as a soft loss penalty.
2. **Geology-adaptive hierarchical decoding.** A two-stage pipeline first
   forecasts formation-scale trend tokens (`BaseDecoder`) and then refines
   them into fine-grained parameter tokens (`SelfConditionedDecoder`), with a
   moving-average window whose width is selected per sample from local
   gamma-ray variance.

On three datasets (Volve, Tarim Basin, FORCE 2020 Carbonate), GeoPHiT
reduces RMSE by up to 8.9 % over thirteen physics-informed and
Transformer-based baselines, while keeping the Constraint Violation Rate
(CVR) at 3.2–5.8 %. Real-time inference is 0.73 s per 100-meter sequence on
an RTX 4090.

## Method at a glance

![GeoPHiT framework: dual-stream physics-aware VQGAN (Stage 1) feeding a
hierarchical discrete-token Transformer (Stage 2).](assets/figure_1.png)

* **Stage 1** trains a dual-stream VQGAN with **physics-aware** parameter-wise
  quantization (Eq. 4) and a soft Archie + Gardner loss (Eq. 7). The target
  stream encodes high-resolution `X_pred`; the trend stream encodes its
  geology-adaptive moving average `X_down` (Eq. 1). Each stream owns a
  separate codebook (`Z_t`, `Z_d`).
* **Stage 2** freezes the codebooks and trains a hierarchical token
  Transformer: a *Context Encoder* maps the historical window `X_p` to
  contextual embeddings `H_p`; a *Base Decoder* autoregressively generates
  trend tokens `s_down` from `H_p`; a *Self-Conditioned Decoder* generates
  fine-grained target tokens `s_pred` conditioned jointly on `s_down` and
  `H_p` (Eqs. 9–10).
* **Inference** uses greedy autoregressive token generation followed by
  detokenisation through the target-stream decoder (Algorithm 2).

## Repository layout

```text
GeoPHiT_code/
├── main.py                         # unified CLI entry point
├── configs/default.yaml            # all hyperparameters from §4.4 of the paper
├── geophit/
│   ├── data/                       # synthetic generator + sliding-window Dataset
│   ├── models/
│   │   ├── blocks.py               # 4-block residual encoder / decoder
│   │   ├── vq.py                   # parameter-wise vector quantizer (Eq. 4)
│   │   ├── vqgan.py                # dual-stream VQGAN
│   │   ├── discriminator.py        # 1D PatchGAN + vanilla BCE GAN losses
│   │   └── transformer.py          # ContextEncoder, BaseDec, SelfCondDec
│   ├── losses/
│   │   ├── physics.py              # Archie + Gardner physics loss (Eq. 7)
│   │   └── gan.py                  # vanilla GAN losses (Eq. 6)
│   ├── eval/metrics.py             # RMSE / MAE / MAPE / GCI / MRPR / CVR
│   └── utils/
│       ├── geological_window.py    # GR-variance adaptive window (Eq. 1)
│       ├── config.py               # YAML loader + dotted overrides
│       └── seed.py
├── scripts/                        # train_stage1, train_stage2, evaluate, infer
├── tests/test_smoke.py             # end-to-end smoke test (CPU, < 10 s)
├── assets/figure_1.png             # framework diagram referenced from README
├── requirements.txt                # minimum versions
├── requirements-lock.txt           # exact versions used to validate the smoke test
├── LICENSE                         # MIT
└── README.md
```

## Installation

```bash
# Clone the anonymous review mirror.
git clone <anonymous-repository-url>
cd GeoPHiT

# Recommended: an isolated environment.
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies. For a CUDA build of PyTorch, follow the wheel selector
# at https://pytorch.org/get-started/locally/ first.
pip install -r requirements.txt
```

For byte-exact reproduction of the validated environment use
`pip install -r requirements-lock.txt` instead.

### System requirements

* Python 3.10+ (developed against 3.12).
* PyTorch 1.13+ (paper) — tested up to 2.11.
* CUDA-capable GPU recommended for full-scale training; the smoke test and
  small-config experiments run on CPU.

## Quick start

The CLI exposes four subcommands plus a convenience `train-all`. Every
hyperparameter in [`configs/default.yaml`](configs/default.yaml) can be
overridden via `--override key.path=value`.

```bash
# 1. Train Stage 1 (dual-stream VQGAN).
python main.py train-stage1

# 2. Train Stage 2 (hierarchical Transformer) on the frozen codebooks.
python main.py train-stage2 --stage1-ckpt ./checkpoints/stage1.pt

# 3. Evaluate (RMSE / MAE / MAPE / GCI / MRPR / CVR).
python main.py evaluate \
    --stage1-ckpt ./checkpoints/stage1.pt \
    --stage2-ckpt ./checkpoints/stage2.pt

# Run all three back-to-back:
python main.py train-all
```

Single-window inference from a `(history_len, 5)` `.npy` file:

```bash
python main.py infer \
    --stage1-ckpt ./checkpoints/stage1.pt \
    --stage2-ckpt ./checkpoints/stage2.pt \
    --history history.npy --out prediction.npy
```

Inline overrides (multiple values, dotted paths):

```bash
python main.py train-stage1 \
    --override seed=1 train.batch_size=8 train.epochs_stage1=20
```

End-to-end smoke test (no GPU required, finishes in seconds):

```bash
python tests/test_smoke.py
```

## Datasets

The default config generates a synthetic well-log corpus that follows
Archie's and Gardner's relations, sized to mirror the public Volve dataset
(15 wells, 405 000 samples). To run on real well logs, add a new branch in
[`geophit/data/well_log_dataset.py`](geophit/data/well_log_dataset.py) and
parse LAS / CSV files into the `(T, 5)` channel order
`(GR, RT, NPHI, DPHI, DEN)`.

The paper validates GeoPHiT on three datasets:

| Dataset           | Type                       | Wells | Samples  | Public ? |
|-------------------|----------------------------|-------|----------|----------|
| Volve             | Offshore N. Sea, clastic   | 15    | 405 000  | yes      |
| Tarim Basin       | Onshore complex            | 12    | 180 000  | private  |
| FORCE 2020 Carb.  | NCS mixed carbonate        | 20    | 245 000  | yes      |

Train / val / test split is well-based (70 % / 15 % / 15 %); test wells are
entirely unseen during training and validation, simulating realistic
cross-well deployment.

## Configuration reference

Every value in [`configs/default.yaml`](configs/default.yaml) matches the
explicit declarations of §4.4 of the paper. The most-tuned knobs:

| Field                       | Paper value     | Where it appears                     |
|-----------------------------|-----------------|--------------------------------------|
| `model.codebook_size`       | 1024 (K)        | `ParameterWiseVectorQuantizer`       |
| `model.n_z`                 | 64              | encoder/decoder bottleneck width     |
| `model.encoder_channels`    | [64,128,256,n_z]| four residual blocks                 |
| `model.d_model`             | 256             | Stage 2 Transformer hidden dim       |
| `model.num_heads`           | 8               | Stage 2 attention heads              |
| `model.num_layers`          | 4               | Context Encoder layer count          |
| `model.geo_window_sand`     | 25              | sandstone moving-average window      |
| `model.geo_window_shale`    | 7               | shale moving-average window          |
| `loss.alpha_physics`        | 0.1             | physics-loss weight α                |
| `loss.beta_commit`          | 0.25            | VQ commitment weight β               |
| `loss.lambda_gan`           | 0.25            | GAN-loss weight λ                    |
| `loss.gan_warmup_ratio`     | 0.75            | adversarial loss starts at 0.75 K    |
| `train.batch_size`          | 32              | optimizer mini-batch                 |
| `train.lr_stage1/2`         | 1e-4            | Adam learning rate                   |
| `train.epochs_stage1/2`     | 200             | with early stopping                  |

## Reproducing the metric panel

The evaluator reports six metrics, three for accuracy and three for physical
consistency:

* **RMSE / MAE / MAPE** — standard accuracy metrics.
* **GCI** (Geological Consistency Index) —
  `exp(-|r_pred − r_target|)` over log-RT vs NPHI Pearson correlation.
* **MRPR** (Mean Relative Physical Residual, Eq. 10) — mean relative
  deviation from Archie's law.
* **CVR** (Constraint Violation Rate, Eq. 11) — fraction of predictions whose
  Archie residual exceeds threshold ε = 0.2.

Run with at least five seeds (`--override seed=<n>`) and aggregate
mean ± std to reproduce the numbers in Tables 3 and 5 of the paper.

## Citation

This paper is currently under **double-blind review** at *Information Fusion*.
Author and grant information are intentionally withheld until acceptance. A
finalised BibTeX entry will be added here once the publisher record is
available:

```bibtex
@unpublished{geophit2026anonymous,
  title  = {GeoPHiT: A Physics-Informed Hierarchical Discrete Transformer
            for Multi-Scale Geological Sequence Prediction},
  author = {{Anonymous}},
  note   = {Manuscript under double-blind review at Information Fusion},
  year   = {2026}
}
```

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgements

Funding sources, institutional affiliations, and individual acknowledgements
have been omitted to comply with the journal's double-blind review policy.
The full acknowledgements section will be restored after acceptance.
