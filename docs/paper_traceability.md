# ALLIE Traceability Notes

This document separates what comes from the paper, what comes from the original
Keras/TensorFlow notebook, and what was added as engineering organization for
publication.

## Paper Reference

- Model family: convolutional autoencoder for low-light image enhancement.
- Task: map low-light images to well-exposed/reference images.
- Application focus: robotics and computer vision.
- Core components:
  - encoder;
  - bottleneck with self-attention;
  - decoder;
  - skip connections;
  - final sigmoid output in `[0, 1]`.
- Architecture used in this repository:
  - four encoder blocks;
  - bottleneck at `15x20x512` when the input is `240x320`;
  - bottleneck at `16x24x512` when the input is `256x384`;
  - four decoder blocks with skip connections.
- Loss:
  - `0.73 * MSE + 0.27 * SSIM_loss`.
- Evaluation:
  - PSNR;
  - SSIM;
  - LOL-v1 and LOL-v2 real datasets.

## Original Notebook Reference

Source: `allie_tun.ipynb`.

- Framework: TensorFlow/Keras.
- Input/target mapping:
  - input: `Low`;
  - target: `Normal`.
- Image preprocessing:
  - OpenCV read;
  - BGR to RGB conversion;
  - resize to `320x240`;
  - normalization to `[0, 1]`.
- Training:
  - batch size: `2`;
  - epochs: `39`;
  - learning rate: `0.0005`;
  - dropout: `0.2`;
  - early stopping patience: `10`.
- Tuned filters:
  - `[256, 256, 256, 96, 256, 256, 96, 96, 256]`.
- Tuned kernel sizes:
  - `[3, 5, 3, 5, 3, 5, 3, 5, 5]`.
- Model format:
  - `.h5`.
- Weight initialization:
  - the publication-oriented implementation uses Keras defaults;
  - no explicit He initialization is used.

## Engineering Adaptation

These changes organize the project for publication:

- paths moved to `configs/allie_base.yaml`;
- aspect-ratio preserving test config added as `configs/allie_384x256.yaml`;
- paper-base architecture moved to `src/allie/model_tf.py`;
- loss moved to `src/allie/losses_tf.py`;
- dataset loading moved to `src/allie/data.py`;
- training command moved to `scripts/train_tf.py`;
- inference command moved to `scripts/infer_tf.py`;
- evaluation command moved to `scripts/evaluate_tf.py`;
- checkpoints saved under `checkpoints/`;
- predicted images saved under `results/predictions/`;
- comparison images saved under `results/comparisons/`;
- local experiment runs saved under `runs/`;
- metrics saved locally under `results/metrics/` or under each `runs/<run_id>/`.
- best validation checkpoint saved under `checkpoints/`.

## Decisions And Known Divergences

- Decision: this repository follows the submitted paper and uses four encoder
  blocks.
- Note: the current Keras notebook used five encoder blocks during exploration.
- Decision: this repository keeps the practical `320x240` training resolution
  as the default publication setting so it can run on GPUs with about 8 GB of
  VRAM.
- Decision: `configs/allie_384x256.yaml` is provided as a local experiment
  configuration because the current dataset images are `600x400`, and
  `384x256` preserves the same `3:2` aspect ratio while remaining divisible by
  16 for the four encoder/decoder blocks.
- Note: the paper discusses LOL images at `400x600`; this can be restored later
  by changing `image.width` and `image.height` in `configs/allie_base.yaml`,
  if the available hardware has enough memory.
- The paper text mentions SSIM weight `0.24` in one sentence, while the equation
  and notebook use `0.27`.
- Decision: the code uses `0.27`, matching the equation and the current notebook.
- Decision: inference uses `Low` as input and `Normal` as ground truth.
