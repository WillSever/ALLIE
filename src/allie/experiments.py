import re
import unicodedata
from pathlib import Path

import yaml

from allie.config import resolve_path
from allie.utils import ensure_dir


def slugify_run_id(value):
    """Create a filesystem-friendly experiment id."""
    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", ascii_value.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("._-")
    if not slug:
        raise ValueError("Experiment id cannot be empty.")
    return slug


def experiment_settings(config):
    settings = config.get("experiments", {})
    return {
        "runs_root": resolve_path(config, settings.get("runs_root", "runs")),
        "default_prefix": settings.get("default_prefix", "resultado"),
    }


def next_numbered_run_id(config):
    settings = experiment_settings(config)
    prefix = slugify_run_id(settings["default_prefix"])
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")

    highest = 0
    for root in (settings["runs_root"],):
        root = Path(root)
        if not root.exists():
            continue
        for child in root.iterdir():
            if not child.is_dir():
                continue
            match = pattern.match(child.name)
            if match:
                highest = max(highest, int(match.group(1)))

    return f"{prefix}_{highest + 1:03d}"


def resolve_run_id(config, run_id=None, new_run=False):
    if run_id and new_run:
        raise ValueError("Use either --run-id or --new-run, not both.")
    if new_run:
        return next_numbered_run_id(config)
    if run_id:
        return slugify_run_id(run_id)
    return None


def apply_experiment_paths(
    config,
    run_id,
    *,
    include_checkpoint=False,
    include_outputs=False,
):
    """Override config paths for a named experiment without editing YAML files."""
    run_id = slugify_run_id(run_id)
    settings = experiment_settings(config)
    run_root = settings["runs_root"] / run_id
    config.setdefault("paths", {})
    config["paths"]["run_dir"] = str(run_root)

    if include_checkpoint:
        checkpoint_path = run_root / "checkpoints" / "allie_model.h5"
        config["paths"]["checkpoint_path"] = str(checkpoint_path)
        config["paths"]["history_path"] = str(run_root / "history.csv")
        config["paths"]["config_snapshot_path"] = str(run_root / "config.yaml")

    if include_outputs:
        config["paths"]["predictions_dir"] = str(run_root / "predictions")
        config["paths"]["comparisons_dir"] = str(run_root / "comparisons")
        config["paths"]["metrics_path"] = str(run_root / "metrics.csv")

    return run_id


def save_config_snapshot(config, output_path):
    """Save the effective configuration used by an experiment run."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    serializable = {
        key: value
        for key, value in config.items()
        if not key.startswith("_")
    }
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(serializable, file, sort_keys=False, allow_unicode=False)
