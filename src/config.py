"""Project paths and configuration loader."""
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_yaml_value(path, section, key, value):
    """Update a single value in a YAML file without rewriting the whole structure."""
    text = Path(path).read_text(encoding="utf-8")
    text = re.sub(
        rf"(^\s*{key}:\s*)[^\n]+",
        lambda m: f"{m.group(1)}{value}",
        text,
        count=1,
        flags=re.M,
    )
    Path(path).write_text(text, encoding="utf-8")

