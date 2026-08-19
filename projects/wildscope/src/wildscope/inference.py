"""SpeciesNet inference, exact-label scoring, and feed-specific correction."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import uuid
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from wildscope.contracts import ModelPrediction

SUPERVISED_PROTOCOL_VERSION = "test-then-train-v1"


class StaticWildlifeModel(Protocol):
    def predict(self, images: dict[int, Path], *, country: str) -> dict[int, ModelPrediction]: ...


class SpeciesNetRunner:
    """Run Google's SpeciesNet ensemble as an isolated resumable batch process."""

    model_version = "speciesnet-5.0.5"

    def __init__(self, work_root: Path, *, timeout_seconds: int = 3600) -> None:
        self.work_root = work_root
        self.timeout_seconds = timeout_seconds

    def predict(self, images: dict[int, Path], *, country: str) -> dict[int, ModelPrediction]:
        if not images:
            return {}
        job = self.work_root / f"speciesnet-{uuid.uuid4().hex}"
        input_root = job / "input"
        input_root.mkdir(parents=True)
        for photo_id, source in images.items():
            destination = input_root / f"{photo_id}{source.suffix.lower()}"
            try:
                destination.hardlink_to(source)
            except OSError:
                destination.write_bytes(source.read_bytes())
        output = job / "predictions.json"
        command = [
            sys.executable,
            "-m",
            "speciesnet.scripts.run_model",
            "--folders",
            str(input_root),
            "--predictions_json",
            str(output),
            "--country",
            country,
        ]
        completed = subprocess.run(
            command,
            cwd=self.work_root,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        if completed.returncode != 0 or not output.is_file():
            detail = (completed.stderr or completed.stdout or "SpeciesNet failed")[-1000:]
            raise RuntimeError(detail)
        payload = json.loads(output.read_text(encoding="utf-8"))
        predictions = {}
        for item in payload.get("predictions", []):
            path = Path(str(item.get("filepath", "")))
            try:
                photo_id = int(path.stem)
            except ValueError:
                continue
            label = str(item.get("prediction") or "unknown")
            score = float(item.get("prediction_score") or 0.0)
            predictions[photo_id] = ModelPrediction(
                label=label,
                confidence=max(0.0, min(1.0, score)),
                model_version=str(item.get("model_version") or self.model_version),
            )
        return predictions


def train_adaptive_corrector(
    rows: list[dict[str, Any]],
    existing_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for static_label, labels in (existing_payload or {}).get("counts", {}).items():
        counts[static_label].update({str(label): int(value) for label, value in labels.items()})
    for row in rows:
        counts[str(row["static_label"])][str(row["scientific_name"])] += 1
    return {
        "counts": {
            static_label: dict(sorted(labels.items()))
            for static_label, labels in sorted(counts.items())
        },
        "sample_count": sum(sum(labels.values()) for labels in counts.values()),
        "protocol_version": SUPERVISED_PROTOCOL_VERSION,
        "training_photo_ids": sorted(
            int(row["photo_id"]) for row in rows if row.get("photo_id") is not None
        ),
    }


def identification_matches(
    model_label: str | None,
    scientific_name: str | None,
    common_name: str | None = None,
) -> bool:
    if not model_label:
        return False
    leaf = str(model_label).split(";")[-1]
    predicted = _normalized_identity(leaf)
    targets = {
        _normalized_identity(value)
        for value in (scientific_name, common_name)
        if value
    }
    return bool(predicted and predicted in targets)


def evaluate_identification_rows(
    rows: list[dict[str, Any]], *, label_field: str
) -> dict[str, int | float | None]:
    correct = sum(
        identification_matches(
            row.get(label_field),
            row.get("scientific_name"),
            row.get("common_name"),
        )
        for row in rows
    )
    samples = len(rows)
    return {
        "samples": samples,
        "correct": correct,
        "accuracy": correct / samples if samples else None,
    }


def evaluate_adaptive_corrector(
    rows: list[dict[str, Any]],
    payload: dict[str, Any],
    *,
    trained_at: str,
) -> dict[str, int | float | None]:
    baseline_correct = 0
    adaptive_correct = 0
    for row in rows:
        static = ModelPrediction(
            str(row["static_label"]),
            float(row["static_confidence"]),
            str(row.get("static_model_version") or "speciesnet"),
        )
        adaptive = apply_adaptive_corrector(static, payload, trained_at=trained_at)
        target = str(row["scientific_name"])
        common_name = str(row["common_name"]) if row.get("common_name") else None
        baseline_correct += identification_matches(static.label, target, common_name)
        adaptive_correct += identification_matches(adaptive.label, target, common_name)
    test_samples = len(rows)
    baseline_accuracy = baseline_correct / test_samples if test_samples else None
    adaptive_accuracy = adaptive_correct / test_samples if test_samples else None
    return {
        "test_samples": test_samples,
        "baseline_correct": baseline_correct,
        "adaptive_correct": adaptive_correct,
        "baseline_accuracy": baseline_accuracy,
        "adaptive_accuracy": adaptive_accuracy,
        "accuracy_delta": (
            adaptive_accuracy - baseline_accuracy
            if adaptive_accuracy is not None and baseline_accuracy is not None
            else None
        ),
    }


def apply_adaptive_corrector(
    static: ModelPrediction, payload: dict[str, Any], *, trained_at: str
) -> ModelPrediction:
    labels = payload.get("counts", {}).get(static.label, {})
    if not labels:
        return ModelPrediction(
            static.label,
            static.confidence,
            f"adaptive-fallback:{static.model_version}",
            trained_at,
        )
    total = sum(int(value) for value in labels.values())
    label, count = max(labels.items(), key=lambda item: (int(item[1]), item[0]))
    posterior = int(count) / total
    confidence = max(0.0, min(1.0, (posterior + static.confidence) / 2.0))
    return ModelPrediction(label, confidence, "adaptive-species-corrector-v1", trained_at)


def model_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _normalized_identity(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())
