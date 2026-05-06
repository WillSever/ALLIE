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

from allie.config import get_path, load_config, resolve_path
from allie.experiments import apply_experiment_paths, resolve_run_id
from allie.inference import load_trained_model, run_inference
from allie.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Run ALLIE inference.")
    parser.add_argument(
        "--config",
        default="configs/allie_base.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--no-targets",
        action="store_true",
        help="Run inference without ground-truth images.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Use runs/RUN_ID/checkpoints/allie_model.h5 and save outputs "
            "under runs/RUN_ID/."
        ),
    )
    parser.add_argument(
        "--new-run",
        action="store_true",
        help=(
            "Create the next numbered output folder, such as treino_001. "
            "This uses the checkpoint configured in the YAML file."
        ),
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Optional model path to use instead of the checkpoint configured in YAML.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    run_id = resolve_run_id(config, run_id=args.run_id, new_run=args.new_run)
    if args.run_id:
        apply_experiment_paths(
            config,
            run_id,
            include_checkpoint=True,
            include_outputs=True,
        )
        print(f"Experiment run id: {run_id}")
    elif args.new_run:
        apply_experiment_paths(config, run_id, include_outputs=True)
        print(f"Output run id: {run_id}")

    set_seed(config["project"]["seed"])

    checkpoint_path = (
        resolve_path(config, args.checkpoint_path)
        if args.checkpoint_path
        else get_path(config, "paths", "checkpoint_path")
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Run scripts/train_tf.py first."
        )

    model = load_trained_model(checkpoint_path)
    input_dir = get_path(config, "paths", "test_low_dir")
    target_dir = None if args.no_targets else get_path(config, "paths", "test_normal_dir")

    run_inference(
        model=model,
        input_dir=input_dir,
        target_dir=target_dir,
        predictions_dir=get_path(config, "paths", "predictions_dir"),
        comparisons_dir=get_path(config, "paths", "comparisons_dir"),
        metrics_path=get_path(config, "paths", "metrics_path"),
        width=config["image"]["width"],
        height=config["image"]["height"],
        save_predictions=config["inference"]["save_predictions"],
        save_comparisons=config["inference"]["save_comparisons"],
    )


if __name__ == "__main__":
    main()
