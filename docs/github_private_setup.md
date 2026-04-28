# Private GitHub Repository Setup

This project should be uploaded without datasets, checkpoints, or generated
results. The `.gitignore` already excludes:

- `data/Real_captured/`
- trained models in `checkpoints/`
- generated predictions and metrics in `results/`
- local Python/Conda environments

## 1. Initialize Git

From the project root:

```powershell
git init
git status --short
```

## 2. Add Files

```powershell
git add .
git status --short
```

Before committing, check that large dataset images are not listed.

## 3. Commit

```powershell
git commit -m "Initial ALLIE project structure"
```

## 4. Create Private Repository On GitHub

In GitHub:

1. Click **New repository**.
2. Repository name: `ALLIE`.
3. Visibility: **Private**.
4. Do not initialize with README, `.gitignore`, or license because this project
   already has local files.
5. Create repository.

## 5. Connect Local Project To GitHub

Replace `YOUR_USERNAME` with your GitHub username:

```powershell
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ALLIE.git
git push -u origin main
```

If Git asks for login, use the browser login flow or a GitHub personal access
token through Git Credential Manager.

## 6. Later Updates

```powershell
git status --short
git add .
git commit -m "Describe the change"
git push
```

