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

from allie.config import get_path, load_config
from allie.inference import load_trained_model, run_inference
from allie.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ALLIE on paired test images.")
    parser.add_argument(
        "--config",
        default="configs/allie_base.yaml",
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["project"]["seed"])

    checkpoint_path = get_path(config, "paths", "checkpoint_path")
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Run scripts/train_tf.py first."
        )

    model = load_trained_model(checkpoint_path)
    run_inference(
        model=model,
        input_dir=get_path(config, "paths", "test_low_dir"),
        target_dir=get_path(config, "paths", "test_normal_dir"),
        predictions_dir=get_path(config, "paths", "predictions_dir"),
        comparisons_dir=get_path(config, "paths", "comparisons_dir"),
        metrics_path=get_path(config, "paths", "metrics_path"),
        width=config["image"]["width"],
        height=config["image"]["height"],
        save_predictions=False,
        save_comparisons=False,
    )


if __name__ == "__main__":
    main()
