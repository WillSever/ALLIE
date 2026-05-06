# Reproducibility Notes

The current base configuration in `configs/allie_base.yaml` follows the
submitted paper architecture:

```yaml
architecture: paper_base_4_blocks
encoder_filters: [64, 128, 256, 512]
bottleneck_filters: 512
decoder_filters: [512, 256, 128, 64]
kernel_size: 3
```

An additional local experiment configuration is available in
`configs/allie_384x256.yaml`. It keeps the same architecture and training
hyperparameters, but uses `384x256` so the current `600x400` dataset keeps its
`3:2` aspect ratio after resizing.

Training hyperparameters are based on the project notes, with `batch_size`
lowered to `1` for stable training on 8 GB GPUs:

```yaml
dropout: 0.2
learning_rate: 0.0005
batch_size: 1
epochs: 39
```

Reference metrics are reported in the submitted paper and should be regenerated
locally when testing this repository.

Important: the `descricao.txt` result refers to a later/tuned experiment that
used the original dataset resolution, an additional layer, and explicit weight
initialization. It is useful as a historical reference, but it is not the exact
target configuration of the paper-base four-block implementation.

Important details:

- Images are sorted alphabetically before pairing.
- Pairing assumes `Low` and `Normal` folders contain matching files in the same
  order.
- Training data is loaded into memory, matching the notebook style.
- Metrics are computed on resized images using the configured dimensions.
- Early stopping follows the original notebook workflow.
- Best model selection uses validation loss and saves the best checkpoint.
- Publication training writes to `checkpoints/allie_base.h5`.
- Local experiments should use `--run-id` or `--new-run`, which write
  checkpoints, history, configuration snapshot, predictions, comparisons, and
  metrics under `runs/<run_id>/`.
- The current source code uses Keras default initializers, matching the
  base/paper-oriented ALLIE implementation rather than the later notebook
  variant with He initialization.
- If a checkpoint was trained before the initializer cleanup, regenerate the
  checkpoint and result images before publishing a fully aligned release.
- The default `320x240` size and `batch_size: 1` are intentional for machines
  with around 8 GB of GPU memory. The `384x256` config is the first recommended
  test when preserving the dataset aspect ratio matters.
