"""Configuration loading with explicit defaults and no import-time side effects."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class WebSettings:
    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = False


@dataclass(frozen=True, slots=True)
class Paths:
    project_root: Path
    inference_config: Path
    source_config: Path
    model_bundle: Path | None


@dataclass(frozen=True, slots=True)
class Settings:
    web: WebSettings
    paths: Paths
    inference: dict[str, Any]


def load_settings() -> Settings:
    inference_path = _resolve_path(os.getenv("ROADID_INFERENCE_CONFIG", "configs/inference.yaml"))
    source_path = _resolve_path(
        os.getenv("ROADID_SOURCE_CONFIG", "configs/camera_sources.example.yaml")
    )
    bundle_text = os.getenv("ROADID_MODEL_BUNDLE", "").strip()
    bundle_path = _resolve_path(bundle_text) if bundle_text else None
    inference = _read_yaml(inference_path)
    return Settings(
        web=WebSettings(
            host=os.getenv("ROADID_HOST", "127.0.0.1"),
            port=int(os.getenv("ROADID_PORT", "5000")),
            debug=os.getenv("ROADID_DEBUG", "false").lower() == "true",
        ),
        paths=Paths(
            project_root=PROJECT_ROOT,
            inference_config=inference_path,
            source_config=source_path,
            model_bundle=bundle_path,
        ),
        inference=inference,
    )


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"configuration file not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return payload
