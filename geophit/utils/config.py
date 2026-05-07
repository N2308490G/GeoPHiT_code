"""YAML config loading and minimal CLI overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def parse_dotted_overrides(items: list[str]) -> dict[str, Any]:
    """Turn ``["train.batch_size=8", "seed=1"]`` into a nested dict."""
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"override must be key=value: {item!r}")
        key, raw = item.split("=", 1)
        try:
            value: Any = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw
        cursor = out
        parts = key.split(".")
        for p in parts[:-1]:
            cursor = cursor.setdefault(p, {})
        cursor[parts[-1]] = value
    return out
