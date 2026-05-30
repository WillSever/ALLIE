[![SBC OpenLib](https://img.shields.io/badge/SBC_OpenLib-article-179bd3)](https://sol.sbc.org.br/index.php/sbrlars/article/view/39242)

# ALLIE

Autoencoder-based Low-Light Image Enhancer.

ALLIE is a TensorFlow/Keras project for low-light image enhancement, organized
from the original research notebook into a reusable repository for robotics and
computer vision researchers.

This repository currently focuses on the publication-oriented base ALLIE model:

- four encoder blocks, following the submitted paper architecture;
- self-attention in the bottleneck;
- skip connections between encoder and decoder;
- configurable dataset paths;
- training, inference, and optional local evaluation scripts;
- saved checkpoint for the reproduced result;
- saved prediction and comparison images.

No metric values are reported in this README. Evaluation CSV files are treated
as local files so researchers can regenerate them on their own datasets.

## Repository Layout

```text
ALLIE/
|-- configs/
|   |-- allie_base.yaml
|   `-- allie_384x256.yaml
|-- src/
|   `-- allie/
|-- scripts/
|   |-- train_tf.py
|   |-- infer_tf.py
|   |-- evaluate_tf.py
|   `-- check_tf_gpu.py
|-- data/
|-- checkpoints/
|   `-- allie_base.h5
|-- results/
|   |-- predictions/
|   `-- comparisons/
|-- docs/
`-- requirements-win-gpu-tf210.txt
```

## Published Artifacts

The repository is configured to publish:

- source code;
- configuration files;
- `checkpoints/allie_base.h5`;
- `checkpoints/allie_384x256.h5`;
- generated prediction images in `results/predictions/`;
- generated comparison images in `results/comparisons/`;
- aspect-ratio preserving prediction images in `results/predictions_384x256/`;
- aspect-ratio preserving comparison images in `results/comparisons_384x256/`.

The repository does not publish by default:

- datasets in `data/`;
- local metric CSV files;
- local experiment folders in `runs/`;
- virtual environments.

The `.h5` checkpoints are larger than normal source files, so they should be
stored with Git LFS.

## Install Miniconda

For native Windows GPU training, use Miniconda with Python 3.10.

Official Miniconda documentation:
[anaconda.com/docs/getting-started/miniconda/main](https://www.anaconda.com/docs/getting-started/miniconda/main)

Official Windows installation guide:
[docs.conda.io/projects/conda/en/stable/user-guide/install/windows.html](https://docs.conda.io/projects/conda/en/stable/user-guide/install/windows.html)

After installing Miniconda, open PowerShell or Anaconda Prompt.

If `conda` is not recognized in PowerShell, open Anaconda Prompt or use the
Miniconda path from your own installation, for example:

```powershell
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" env list
```

## Clone The Repository

```powershell
git clone https://github.com/WillSever/ALLIE.git
cd ALLIE
```

If the model checkpoint is tracked with Git LFS, install and pull LFS files:

```powershell
git lfs install
git lfs pull
```

## Create The Environment

```powershell
conda create -n myenv python=3.10 -y
```

Activate the environment:

```powershell
conda activate myenv
```

If that does not work in PowerShell, open Anaconda Prompt and run the same
command, or use the activation script from your own Miniconda installation.

```powershell
conda activate myenv
```

## Install Dependencies

Install CUDA and cuDNN inside the Conda environment:

```powershell
conda install -n myenv -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y
```

Install Python packages:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements-win-gpu-tf210.txt
```

## Configure Windows GPU Terminal

Before running TensorFlow scripts on native Windows, run:

```powershell
.\scripts\activate_windows_gpu.ps1
```

Then check whether TensorFlow can see the GPU:

```powershell
python scripts\check_tf_gpu.py
```

Expected result:

```text
GPU is available.
```

More details are available in `docs/windows_gpu_setup.md`.

## Dataset Layout

By default, `configs/allie_base.yaml` expects:

```text
data/
`-- Real_captured/
    |-- Train/
    |   |-- Low/
    |   `-- Normal/
    `-- Test/
        |-- Low/
        `-- Normal/
```

`Low` images are the inputs. `Normal` images are the reference targets.

Datasets are not tracked by Git. Each researcher should place their dataset
locally or edit the config to point to another folder.

## Run Inference With Published Checkpoints

Base lightweight configuration:

```powershell
python scripts\infer_tf.py --config configs\allie_base.yaml
```

This uses:

```text
checkpoints/allie_base.h5
```

and writes:

```text
results/predictions/
results/comparisons/
```

Aspect-ratio preserving configuration:

```powershell
python scripts\infer_tf.py --config configs\allie_384x256.yaml
```

This uses:

```text
checkpoints/allie_384x256.h5
```

and writes:

```text
results/predictions_384x256/
results/comparisons_384x256/
```

## Train The Base Model

```powershell
python scripts\train_tf.py --config configs\allie_base.yaml --require-gpu
```

The best validation checkpoint is saved to:

```text
checkpoints/allie_base.h5
```

Training uses `EarlyStopping`, which was already part of the original notebook
workflow. The organized script also uses `ModelCheckpoint` to explicitly save
the best validation model to disk.

## Train Without Overwriting Previous Runs

For research work, prefer numbered runs:

```powershell
python scripts\train_tf.py --config configs\allie_base.yaml --require-gpu --new-run
```

This creates a new folder such as:

```text
runs/treino_001/
|-- checkpoints/
|   `-- allie_model.h5
|-- config.yaml
`-- history.csv
```

Run inference for the same training run:

```powershell
python scripts\infer_tf.py --config configs\allie_base.yaml --run-id treino_001
```

The same folder then receives:

```text
runs/treino_001/
|-- predictions/
|-- comparisons/
`-- metrics.csv
```

The next `--new-run` creates `treino_002`, then `treino_003`, and so on.
The `runs/` folder is local by default and does not overwrite the published
checkpoint or published result images.

## Aspect-Ratio Preserving Resolution Test

The original images in the current dataset are `600x400`, a `3:2` aspect ratio.
The default `320x240` setting is lighter for 8 GB GPUs, but changes that ratio.
The repository also publishes a `384x256` checkpoint and result images because
that size preserves the dataset aspect ratio while remaining lightweight.

To train this configuration again as a local experiment, use:

```powershell
python scripts\train_tf.py --config configs\allie_384x256.yaml --require-gpu --new-run
```

After training, use the printed run id:

```powershell
python scripts\infer_tf.py --config configs\allie_384x256.yaml --run-id YOUR_RUN_ID
```

For example, if training created `runs/treino_002/`, use
`--run-id treino_002`.

## Optional Local Evaluation

```powershell
python scripts\evaluate_tf.py --config configs\allie_base.yaml
```

This computes local PSNR/SSIM CSV files. These CSV files are ignored by Git and
are not part of the published README result.

## Test A New Dataset Without Overwriting Results

Create a local config copy:

```powershell
Copy-Item configs\allie_base.yaml configs\my_dataset_local.yaml
```

Edit this local file:

```yaml
paths:
  train_low_dir: path/to/your/Train/Low
  train_normal_dir: path/to/your/Train/Normal
  test_low_dir: path/to/your/Test/Low
  test_normal_dir: path/to/your/Test/Normal
```

Run the published checkpoint on the new dataset and create a new numbered output
folder:

```powershell
python scripts\infer_tf.py --config configs\my_dataset_local.yaml --new-run --checkpoint-path checkpoints/allie_base.h5
```

This creates folders like:

```text
runs/treino_001/predictions/
runs/treino_001/comparisons/
```

These folders are local by default and do not overwrite the published
`results/predictions/` and `results/comparisons/` folders.

## Train A Named Experiment

For a new dataset or a controlled experiment:

```powershell
python scripts\train_tf.py --config configs\my_dataset_local.yaml --run-id meu_dataset_v1 --require-gpu
```

This saves:

```text
runs/meu_dataset_v1/checkpoints/allie_model.h5
```

Then run inference with the same experiment id:

```powershell
python scripts\infer_tf.py --config configs\my_dataset_local.yaml --run-id meu_dataset_v1
```

This writes:

```text
runs/meu_dataset_v1/predictions/
runs/meu_dataset_v1/comparisons/
```

## Traceability

See:

- `docs/paper_traceability.md`
- `docs/reproducibility_notes.md`
- `docs/windows_gpu_setup.md`
