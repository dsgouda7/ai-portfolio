"""SpeciesNet inference, exact-label scoring, and feed-specific correction."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import uuid
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from wildscope.contracts import ModelPrediction

SUPERVISED_PROTOCOL_VERSION = "test-then-train-v3-bioclip-selective"
BIOCLIP_MODEL_NAME = "hf-hub:imageomics/bioclip"
BIOCLIP_MARGIN_THRESHOLD = 0.075


class StaticWildlifeModel(Protocol):
    def predict(self, images: dict[int, Path], *, country: str) -> dict[int, ModelPrediction]: ...


class VisualSpeciesModel(Protocol):
    def predict(
        self, image: Path, candidates: tuple[str, ...], *, trained_at: str
    ) -> ModelPrediction: ...


class BioClipSpeciesClassifier:
    """Classify an image against a feed's known taxa and abstain on weak margins."""

    model_name = BIOCLIP_MODEL_NAME

    def __init__(self, *, margin_threshold: float = BIOCLIP_MARGIN_THRESHOLD) -> None:
        self.margin_threshold = margin_threshold
        self._model: Any = None
        self._preprocess: Any = None
        self._tokenizer: Any = None
        self._text_cache: dict[tuple[str, ...], Any] = {}
        self._lock = threading.RLock()

    def predict(
        self, image: Path, candidates: tuple[str, ...], *, trained_at: str
    ) -> ModelPrediction:
        if len(candidates) < 2:
            raise ValueError("BioCLIP requires at least two candidate species")
        with self._lock:
            self._load()
            import torch
            from PIL import Image

            text_features = self._text_features(candidates)
            with Image.open(image) as opened, torch.inference_mode():
                tensor = self._preprocess(opened.convert("RGB")).unsqueeze(0)
                image_features = self._model.encode_image(tensor)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                scores = (image_features @ text_features.T)[0]
                values, indices = torch.topk(scores, 2)
            margin = float(values[0] - values[1])
            label = candidates[int(indices[0])]
            if margin < self.margin_threshold:
                label = "unidentified"
            return ModelPrediction(
                label,
                max(0.0, min(1.0, margin)),
                f"bioclip-vit-b16-selective-margin-{self.margin_threshold:.3f}",
                trained_at,
            )

    def _load(self) -> None:
        if self._model is not None:
            return
        import open_clip

        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self.model_name
        )
        self._tokenizer = open_clip.get_tokenizer(self.model_name)
        self._model.eval()

    def _text_features(self, candidates: tuple[str, ...]) -> Any:
        cached = self._text_cache.get(candidates)
        if cached is not None:
            return cached
        import torch

        with torch.inference_mode():
            features = self._model.encode_text(
                self._tokenizer([f"a photo of {name}" for name in candidates])
            )
            features /= features.norm(dim=-1, keepdim=True)
        self._text_cache[candidates] = features
        return features


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
    target_catalog = dict((existing_payload or {}).get("target_catalog", {}))
    for static_label, labels in (existing_payload or {}).get("counts", {}).items():
        counts[canonical_model_label(static_label)].update(
            {str(label): int(value) for label, value in labels.items()}
        )
    for row in rows:
        source_label = canonical_model_label(str(row["static_label"]))
        scientific_name = str(row["scientific_name"])
        counts[source_label][scientific_name] += 1
        target_catalog[scientific_name] = {
            "taxon_id": (
                int(row["taxon_id"]) if row.get("taxon_id") is not None else None
            ),
            "scientific_name": scientific_name,
            "common_name": (
                str(row["common_name"]) if row.get("common_name") else None
            ),
        }
    return {
        "counts": {
            static_label: dict(sorted(labels.items()))
            for static_label, labels in sorted(counts.items())
        },
        "sample_count": sum(sum(labels.values()) for labels in counts.values()),
        "target_catalog": dict(sorted(target_catalog.items())),
        "protocol_version": SUPERVISED_PROTOCOL_VERSION,
        "visual_model": {
            "name": BIOCLIP_MODEL_NAME,
            "decision_metric": "top-1 minus top-2 cosine similarity",
            "margin_threshold": BIOCLIP_MARGIN_THRESHOLD,
            "candidate_scope": "feed training target catalog",
        },
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
    source_label = canonical_model_label(static.label)
    labels = _labels_for_source(payload, source_label)
    if not labels:
        return ModelPrediction(
            static.label,
            static.confidence,
            f"adaptive-fallback:{static.model_version}",
            trained_at,
        )
    if len(labels) > 1:
        return ModelPrediction(
            source_label,
            static.confidence,
            "adaptive-abstain-ambiguous-v1",
            trained_at,
        )
    total = sum(int(value) for value in labels.values())
    label, count = max(labels.items(), key=lambda item: (int(item[1]), item[0]))
    posterior = int(count) / total
    confidence = max(0.0, min(1.0, (posterior + static.confidence) / 2.0))
    return ModelPrediction(label, confidence, "adaptive-species-corrector-v1", trained_at)


def describe_adaptive_prediction(
    prediction: ModelPrediction,
    payload: dict[str, Any],
    *,
    source_label: str,
) -> dict[str, Any]:
    canonical_source = canonical_model_label(source_label)
    candidates = _labels_for_source(payload, canonical_source)
    ambiguous = len(candidates) > 1
    target = (
        {}
        if ambiguous
        else payload.get("target_catalog", {}).get(prediction.label, {})
    )
    return {
        "scientific_name": (
            None
            if ambiguous
            else str(target.get("scientific_name") or prediction.label)
        ),
        "common_name": "Unidentified" if ambiguous else target.get("common_name"),
        "taxon_id": target.get("taxon_id"),
        "source_label": canonical_source,
        "candidate_count": len(candidates),
        "candidate_scientific_names": sorted(candidates),
        "ambiguous": ambiguous,
        "abstained": ambiguous,
    }


def describe_visual_prediction(
    prediction: ModelPrediction, payload: dict[str, Any]
) -> dict[str, Any]:
    candidates = tuple(sorted(payload.get("target_catalog", {})))
    abstained = prediction.label == "unidentified"
    target = payload.get("target_catalog", {}).get(prediction.label, {})
    visual_model = payload.get("visual_model", {})
    return {
        "scientific_name": None if abstained else prediction.label,
        "common_name": "Unidentified" if abstained else target.get("common_name"),
        "taxon_id": None if abstained else target.get("taxon_id"),
        "source_label": str(visual_model.get("name") or BIOCLIP_MODEL_NAME),
        "candidate_count": len(candidates),
        "candidate_scientific_names": list(candidates),
        "ambiguous": abstained,
        "abstained": abstained,
        "decision_metric": str(
            visual_model.get("decision_metric")
            or "top-1 minus top-2 cosine similarity"
        ),
        "decision_margin": prediction.confidence,
        "margin_threshold": float(
            visual_model.get("margin_threshold", BIOCLIP_MARGIN_THRESHOLD)
        ),
    }


def canonical_model_label(value: str) -> str:
    parts = [part.strip() for part in str(value).split(";")]
    named = [part for part in parts[1:] if part]
    return ";".join(named) if named else str(value).strip()


def _labels_for_source(
    payload: dict[str, Any], source_label: str
) -> dict[str, int]:
    counts = payload.get("counts", {})
    direct = counts.get(source_label)
    if direct is not None:
        return {str(label): int(value) for label, value in direct.items()}
    merged: Counter[str] = Counter()
    for stored_source, labels in counts.items():
        if canonical_model_label(stored_source) == source_label:
            merged.update({str(label): int(value) for label, value in labels.items()})
    return dict(merged)


def model_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _normalized_identity(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())
