# Windows GPU Setup

This guide configures ALLIE for native Windows GPU training using TensorFlow
2.10, the last TensorFlow release with native Windows GPU support.

## Current Machine

Detected GPU:

```text
NVIDIA GeForce RTX 3070
VRAM: 8 GB
```

Default ALLIE settings are intentionally conservative for 8 GB VRAM:

```yaml
image:
  width: 320
  height: 240
training:
  batch_size: 2
```

## Why Python 3.12 Does Not Work Here

The IDE currently sees a global Python 3.12 installation. TensorFlow 2.10 does
not provide Windows GPU wheels for Python 3.12.

Use a Conda environment with Python 3.10.

## Conda Path

If `conda` is not recognized in PowerShell, use the full path:

```powershell
& 'C:\Users\willi\miniconda3\Scripts\conda.exe' env list
```

## First-Time Conda Terms

If Conda asks you to accept the Terms of Service, run:

```powershell
& 'C:\Users\willi\miniconda3\Scripts\conda.exe' tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
& 'C:\Users\willi\miniconda3\Scripts\conda.exe' tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
& 'C:\Users\willi\miniconda3\Scripts\conda.exe' tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
```

Only run those commands if you agree with Anaconda's channel terms.

## Create Environment

From the repository root:

```powershell
& 'C:\Users\willi\miniconda3\Scripts\conda.exe' create -n myenv python=3.10 -y
```

Activate it:

```powershell
& 'C:\Users\willi\miniconda3\Scripts\activate.bat' myenv
```

If activation does not change the prompt, open "Anaconda Prompt" and run:

```bat
cd C:\Users\willi\Documents\Codex\2026-04-28\files-mentioned-by-the-user-allie\ALLIE
conda activate myenv
```

## Install GPU Dependencies

```powershell
& 'C:\Users\willi\miniconda3\Scripts\conda.exe' install -n myenv -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y
```

Install Python packages:

```powershell
& 'C:\Users\willi\miniconda3\envs\myenv\python.exe' -m pip install --upgrade pip
& 'C:\Users\willi\miniconda3\envs\myenv\python.exe' -m pip install -r requirements-win-gpu-tf210.txt
```

## Test TensorFlow GPU

In the Antigravity terminal, first run:

```powershell
.\scripts\activate_windows_gpu.ps1
```

This also sets:

```powershell
$env:TF_FORCE_GPU_ALLOW_GROWTH = "true"
$env:TF_CUDNN_USE_AUTOTUNE = "0"
```

Then test:

```powershell
python scripts\check_tf_gpu.py
```

Alternative full-path test:

```powershell
& 'C:\Users\willi\miniconda3\envs\myenv\python.exe' -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

Expected:

```text
2.10.x
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

If TensorFlow prints a warning about `ptxas.exe` but continues running, this is
not fatal. The important line is:

```text
Loaded cuDNN version 8100
```

If training fails with `DNN library is not found`, run
`.\scripts\activate_windows_gpu.ps1` again in the same terminal before training.
The scripts also register `myenv\Library\bin` as a DLL directory on Windows
before TensorFlow is imported.

If training fails with memory allocation errors on an 8 GB GPU, keep
`batch_size: 1` in `configs/allie_base.yaml`.

## Run ALLIE

Train:

```powershell
python scripts\train_tf.py --config configs\allie_base.yaml
```

Quick smoke test without saving a checkpoint:

```powershell
python scripts\train_tf.py --config configs\allie_base.yaml --max-train-images 8 --epochs 1 --no-save
```

Infer:

```powershell
python scripts\infer_tf.py --config configs\allie_base.yaml
```

Evaluate:

```powershell
python scripts\evaluate_tf.py --config configs\allie_base.yaml
```
