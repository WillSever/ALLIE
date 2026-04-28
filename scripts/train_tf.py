import argparse
import os
from pathlib import Path
import sys

from _windows_dll_paths import configure_windows_dll_paths

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_CUDNN_USE_AUTOTUNE", "0")
configure_windows_dll_paths()

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from allie.config import get_path, load_config
from allie.data import load_image_pairs
from allie.model_tf import build_allie_model
from allie.utils import ensure_parent, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train ALLIE TensorFlow model.")
    parser.add_argument(
        "--config",
        default="configs/allie_base.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Train again even if the checkpoint already exists.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of epochs from the config.",
    )
    parser.add_argument(
        "--max-train-images",
        type=int,
        default=None,
        help="Use only the first N training pairs. Useful for smoke tests.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run training without saving model/history files.",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Stop if TensorFlow does not detect a GPU.",
    )
    return parser.parse_args()


def get_tensorflow_gpus():
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    return gpus


def main():
    args = parse_args()
    config = load_config(args.config)

    set_seed(config["project"]["seed"])
    gpus = get_tensorflow_gpus()
    print(f"GPUs detected by TensorFlow: {len(gpus)}")
    for gpu in gpus:
        print(f"  - {gpu}")

    if args.require_gpu and not gpus:
        raise RuntimeError(
            "TensorFlow did not detect a GPU. Run scripts\\activate_windows_gpu.ps1 "
            "in this terminal and check python scripts\\check_tf_gpu.py."
        )

    width = config["image"]["width"]
    height = config["image"]["height"]
    channels = config["image"]["channels"]

    checkpoint_path = get_path(config, "paths", "checkpoint_path")
    history_path = checkpoint_path.with_suffix(".history.csv")

    if checkpoint_path.exists() and not args.overwrite and not args.no_save:
        print(f"Checkpoint already exists: {checkpoint_path}")
        print("Use --overwrite if you want to train again.")
        return

    train_low_dir = get_path(config, "paths", "train_low_dir")
    train_normal_dir = get_path(config, "paths", "train_normal_dir")

    print("Loading training images...")
    x_train, y_train, pairs = load_image_pairs(
        input_dir=train_low_dir,
        target_dir=train_normal_dir,
        width=width,
        height=height,
    )
    print(f"Loaded pairs: {len(pairs)}")
    print(f"X_train shape: {x_train.shape}")
    print(f"Y_train shape: {y_train.shape}")

    if args.max_train_images is not None:
        x_train = x_train[: args.max_train_images]
        y_train = y_train[: args.max_train_images]
        pairs = pairs[: args.max_train_images]
        print(f"Smoke/test mode: using first {len(pairs)} training pairs.")

    input_shape = (height, width, channels)
    model = build_allie_model(
        input_shape=input_shape,
        encoder_filters=config["model"]["encoder_filters"],
        bottleneck_filters=config["model"]["bottleneck_filters"],
        decoder_filters=config["model"]["decoder_filters"],
        dropout_rate=config["training"]["dropout"],
        learning_rate=config["training"]["learning_rate"],
        kernel_size=config["model"]["kernel_size"],
        mse_weight=config["loss"]["mse_weight"],
        ssim_weight=config["loss"]["ssim_weight"],
    )

    model.summary()

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=config["training"]["early_stopping_patience"],
        restore_best_weights=True,
    )

    print("Training ALLIE...")
    history = model.fit(
        x_train,
        y_train,
        epochs=args.epochs or config["training"]["epochs"],
        batch_size=config["training"]["batch_size"],
        validation_split=config["training"]["validation_split"],
        callbacks=[early_stop],
        verbose=1,
    )

    if args.no_save:
        print("Training finished. --no-save enabled, skipping checkpoint/history save.")
        return

    ensure_parent(checkpoint_path)
    model.save(str(checkpoint_path))
    print(f"Model saved at: {checkpoint_path}")

    pd.DataFrame(history.history).to_csv(history_path, index=False)
    print(f"Training history saved at: {history_path}")


if __name__ == "__main__":
    main()
