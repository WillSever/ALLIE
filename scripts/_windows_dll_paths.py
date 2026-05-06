import os
from pathlib import Path


def configure_windows_dll_paths():
    """Make Conda CUDA/cuDNN DLLs visible before importing TensorFlow."""
    if os.name != "nt":
        return

    candidates = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix))

    conda_env = os.environ.get("CONDA_ENV")
    if conda_env:
        candidates.append(Path(conda_env))

    user_profile = os.environ.get("USERPROFILE")
    if user_profile:
        candidates.append(Path(user_profile) / "miniconda3" / "envs" / "myenv")
        candidates.append(Path(user_profile) / "anaconda3" / "envs" / "myenv")

    for env_path in candidates:
        for dll_dir in [env_path, env_path / "Library" / "bin", env_path / "Scripts"]:
            if dll_dir.exists():
                os.environ["PATH"] = f"{dll_dir};{os.environ.get('PATH', '')}"
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(str(dll_dir))
                    except OSError:
                        pass
