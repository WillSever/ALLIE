from pathlib import Path

import yaml


def load_config(config_path):
    """Load a YAML configuration file and remember its absolute path."""
    path = Path(config_path)
    if not path.is_absolute():
        path = project_root() / path

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    config["_config_path"] = str(path.resolve())
    return config


def project_root():
    """Return the repository root when running from the source tree."""
    return Path(__file__).resolve().parents[2]


def config_root(config):
    """Resolve relative paths from the directory above configs/."""
    config_path = Path(config["_config_path"]).resolve()
    if config_path.parent.name == "configs":
        return config_path.parent.parent
    return project_root()


def resolve_path(config, value):
    """Resolve a config path relative to the project root."""
    path = Path(value)
    if path.is_absolute():
        return path
    return config_root(config) / path


def get_path(config, section, key):
    return resolve_path(config, config[section][key])

