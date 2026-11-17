"""
Lab 06: YAML configuration

One YAML file describes a whole run. load_config() reads it and applies
command-line overrides such as train.epochs=10. Every override must name a
key that already exists, so a typo fails instead of being silently ignored.
"""

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAB_DIR = PROJECT_ROOT / "lab06"
DEFAULT_CONFIG = LAB_DIR / "configs" / "base.yaml"


def load_config(path=DEFAULT_CONFIG, overrides=()):
    with open(path) as f:
        cfg = yaml.safe_load(f)

    for item in overrides:
        apply_override(cfg, item)

    return cfg


def apply_override(cfg, item):
    """Apply one 'section.key=value' override. The value is parsed as YAML."""

    key, separator, raw_value = item.partition("=")
    if not separator:
        raise ValueError(f"Override must look like section.key=value, got {item!r}")

    *sections, name = key.split(".")
    node = cfg
    for section in sections:
        if not isinstance(node.get(section), dict):
            raise KeyError(f"Unknown config section in {key!r}")
        node = node[section]

    if name not in node:
        raise KeyError(f"Unknown config key {key!r}")

    node[name] = yaml.safe_load(raw_value)


def flatten(cfg, prefix=""):
    """{'train': {'epochs': 5}} -> {'train.epochs': 5}, for MLflow params."""

    flat = {}
    for key, value in cfg.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, name + "."))
        else:
            flat[name] = value
    return flat


def resolve_path(path):
    """Relative paths in the config are relative to the repository root."""

    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path
