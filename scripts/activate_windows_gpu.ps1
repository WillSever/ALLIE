param(
    [string]$CondaEnvPath = ""
)

if (-not $CondaEnvPath) {
    if ($env:CONDA_PREFIX) {
        $CondaEnvPath = $env:CONDA_PREFIX
    }
    else {
        $CondaEnvPath = Join-Path $env:USERPROFILE "miniconda3\envs\myenv"
    }
}

if (-not (Test-Path $CondaEnvPath)) {
    Write-Error "Conda environment not found: $CondaEnvPath"
    Write-Host "Activate your environment first or run:"
    Write-Host "  .\scripts\activate_windows_gpu.ps1 -CondaEnvPath C:\path\to\your\conda\env"
    exit 1
}

$env:CONDA_ENV = $CondaEnvPath
$env:PATH = "$env:CONDA_ENV;$env:CONDA_ENV\Library\bin;$env:CONDA_ENV\Scripts;$env:PATH"
$env:TF_FORCE_GPU_ALLOW_GROWTH = "true"
$env:TF_CUDNN_USE_AUTOTUNE = "0"

Write-Host "ALLIE Windows GPU environment configured for this terminal."
Write-Host "Python: $env:CONDA_ENV\python.exe"
Write-Host "TF_FORCE_GPU_ALLOW_GROWTH=true"
Write-Host "TF_CUDNN_USE_AUTOTUNE=0"
Write-Host ""
Write-Host "Use:"
Write-Host "  python scripts\check_tf_gpu.py"
Write-Host "  python scripts\train_tf.py --config configs\allie_base.yaml --require-gpu"
Write-Host "  python scripts\infer_tf.py --config configs\allie_base.yaml"
