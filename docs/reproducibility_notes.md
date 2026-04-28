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

Training hyperparameters remain based on the project notes:

```yaml
dropout: 0.2
learning_rate: 0.0005
batch_size: 2
epochs: 39
```

Recorded reference metrics from `descricao.txt`:

```text
Mean SSIM: 0.8186
Mean PSNR: 20.0671
```

Important details:

- Images are sorted alphabetically before pairing.
- Pairing assumes `Low` and `Normal` folders contain matching files in the same
  order.
- Training data is loaded into memory, matching the notebook style.
- Metrics are computed on resized images using the configured dimensions.
- The default `320x240` size is intentional for machines with around 8 GB of
  GPU memory. To test higher resolution later, change `image.width` and
  `image.height` in `configs/allie_base.yaml`.
