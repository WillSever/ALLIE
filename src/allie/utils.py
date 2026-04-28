import os
import random
from pathlib import Path

import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def list_image_files(folder):
    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"Image folder not found: {folder}")

    files = [
        item
        for item in folder.iterdir()
        if item.is_file() and item.suffix.lower() in valid_extensions
    ]
    return sorted(files, key=lambda item: item.name)


def configure_tensorflow_memory_growth():
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        return gpus
    except Exception:
        return []


def add_src_to_path():
    """Allow scripts to run without installing the package."""
    import sys

    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
