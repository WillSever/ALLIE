# ALLIE

Autoencoder-based Low-Light Image Enhancer.

ALLIE is a lightweight autoencoder-based model for low-light image enhancement,
with a focus on robotics and computer vision research workflows.

This repository organizes the original TensorFlow/Keras research notebook into
a reusable project with:

- configurable dataset paths;
- local training from an IDE;
- checkpoint saving;
- prediction image export;
- PSNR and SSIM evaluation;
- traceability between paper, original notebook, and implementation.

## Project Layout

```text
ALLIE/
├── configs/
│   └── allie_base.yaml
├── src/
│   └── allie/
├── scripts/
│   ├── train_tf.py
│   ├── infer_tf.py
│   └── evaluate_tf.py
├── data/
├── checkpoints/
├── results/
└── docs/
```

For private GitHub upload instructions, see
`docs/github_private_setup.md`.

## Dataset Layout

By default, `configs/allie_base.yaml` expects:

```text
data/
└── Real_captured/
    ├── Train/
    │   ├── Low/
    │   └── Normal/
    └── Test/
        ├── Low/
        └── Normal/
```

The `Low` folders are the inputs. The `Normal` folders are the reference images.

## Environment

From the repository root:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On native Windows, current TensorFlow releases are generally CPU-oriented. For
NVIDIA GPU training without WSL2, use the legacy TensorFlow 2.10 setup described
in `docs/windows_gpu_setup.md`.

With an 8 GB GPU, keep the default `320x240` resolution and `batch_size: 2`
first. Increase resolution or batch size only after confirming GPU memory usage.

## Training

Edit `configs/allie_base.yaml` if your dataset is somewhere else, then run:

```bash
python scripts/train_tf.py --config configs/allie_base.yaml
```

The trained model is saved to:

```text
checkpoints/allie_base.h5
```

## Inference

```bash
python scripts/infer_tf.py --config configs/allie_base.yaml
```

Predictions are saved to:

```text
results/predictions/
```

Side-by-side comparisons are saved to:

```text
results/comparisons/
```

## Evaluation

```bash
python scripts/evaluate_tf.py --config configs/allie_base.yaml
```

Metrics are saved to:

```text
results/metrics/allie_base_metrics.csv
```

## Current Priority

The current publication target is the base TensorFlow/Keras ALLIE version
aligned with the submitted paper.
