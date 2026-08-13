"""Training configuration loading with paths resolved from the project root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_training_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("training configuration root must be a mapping")
    for section in ("base_model", "data", "training", "calibration", "packaging"):
        if not isinstance(payload.get(section), dict):
            raise ValueError(f"training configuration requires mapping section: {section}")
    payload["_project_root"] = str(path.resolve().parents[1])
    return payload


def resolve_config_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(config["_project_root"]) / path
