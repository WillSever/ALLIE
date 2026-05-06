# Private GitHub Repository Setup

This project is configured to publish the ALLIE code, the publication
checkpoints, and the publication prediction/comparison images.

It intentionally keeps these files local:

- datasets in `data/`;
- metric CSV files;
- local experiment folders in `runs/`;
- virtual environments.

## 1. Configure Git Identity

Run once on your machine:

```powershell
git config --global user.name "Your Name"
git config --global user.email "YOUR_GITHUB_EMAIL"
```

Or configure only this repository:

```powershell
git config user.name "Your Name"
git config user.email "YOUR_GITHUB_EMAIL"
```

## 2. Enable Git LFS

The checkpoint files are large, so use Git LFS:

```powershell
git lfs install
```

The repository already contains `.gitattributes` for:

- `checkpoints/allie_base.h5`
- `checkpoints/allie_384x256.h5`

If source code that affects training was changed, regenerate the publication
artifacts before committing:

```powershell
python scripts\train_tf.py --config configs\allie_base.yaml --require-gpu --overwrite
python scripts\infer_tf.py --config configs\allie_base.yaml
python scripts\train_tf.py --config configs\allie_384x256.yaml --require-gpu --new-run
```

Then run inference for the 384x256 experiment with the run id created by the
training command:

```powershell
python scripts\infer_tf.py --config configs\allie_384x256.yaml --run-id YOUR_RUN_ID
```

## 3. Initialize Git

From the project root:

```powershell
git init
git status --short
```

## 4. Add Files

```powershell
git add .gitattributes
git add .
git status --short
```

Before committing, check that:

- `data/Real_captured/` is not listed;
- `results/metrics/*.csv` is not listed;
- `runs/` is not listed;
- `checkpoints/allie_base.h5` is listed as a Git LFS file;
- `checkpoints/allie_384x256.h5` is listed as a Git LFS file.

You can check the LFS-tracked files with:

```powershell
git lfs ls-files
```

## 5. Commit

```powershell
git commit -m "Publish base ALLIE project"
```

## 6. Connect Local Project To GitHub

For the current private repository:

```powershell
git branch -M main
git remote add origin https://github.com/WillSever/ALLIE.git
git push -u origin main
```

If `origin` already exists:

```powershell
git remote set-url origin https://github.com/WillSever/ALLIE.git
git push -u origin main
```

## 7. Later Updates

```powershell
git status --short
git add .
git commit -m "Describe the change"
git push
```

## 8. Downloading On Another Machine

```powershell
git clone https://github.com/WillSever/ALLIE.git
cd ALLIE
git lfs install
git lfs pull
```
