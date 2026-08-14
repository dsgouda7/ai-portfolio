"""Lazy hierarchical ResNet classification and deterministic demo classification."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from roadid.inference.bundle import ModelBundle


class ClassifierUnavailableError(RuntimeError):
    """Raised when the configured classifier cannot be loaded."""


@dataclass(frozen=True, slots=True)
class LabelSpace:
    body_types: tuple[str, ...]
    makes: tuple[str, ...]
    model_families: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.body_types or not self.makes or not self.model_families:
            raise ValueError("all hierarchy levels require at least one label")
        if any(len(set(level)) != len(level) for level in self.levels):
            raise ValueError("labels must be unique within each hierarchy level")

    @property
    def levels(self) -> tuple[tuple[str, ...], ...]:
        return self.body_types, self.makes, self.model_families

    @property
    def widths(self) -> tuple[int, int, int]:
        return tuple(len(level) for level in self.levels)  # type: ignore[return-value]

    @property
    def size(self) -> int:
        return sum(self.widths)

    @classmethod
    def from_labels(cls, labels: Mapping[str, Any]) -> LabelSpace:
        return cls(
            body_types=tuple(str(value) for value in labels["body_types"]),
            makes=tuple(str(value) for value in labels["makes"]),
            model_families=tuple(str(value) for value in labels["model_families"]),
        )


@dataclass(frozen=True, slots=True)
class HierarchicalScores:
    body_type: tuple[float, ...]
    make: tuple[float, ...]
    model_family: tuple[float, ...]

    def __post_init__(self) -> None:
        for values in self.levels:
            if not values or not all(np.isfinite(values)):
                raise ValueError("hierarchical scores must be non-empty and finite")
            if any(value < 0.0 for value in values) or not np.isclose(sum(values), 1.0):
                raise ValueError("each hierarchy level must be a probability distribution")

    @property
    def levels(self) -> tuple[tuple[float, ...], ...]:
        return self.body_type, self.make, self.model_family

    def flatten(self) -> tuple[float, ...]:
        return (*self.body_type, *self.make, *self.model_family)

    @classmethod
    def from_flat(cls, values: Sequence[float], space: LabelSpace) -> HierarchicalScores:
        if len(values) != space.size:
            raise ValueError(f"expected {space.size} scores, got {len(values)}")
        first = space.widths[0]
        second = first + space.widths[1]
        return cls(
            body_type=_normalize(values[:first]),
            make=_normalize(values[first:second]),
            model_family=_normalize(values[second:]),
        )


class VehicleClassifier(Protocol):
    label_space: LabelSpace

    def classify(self, crop_bgr: np.ndarray) -> HierarchicalScores: ...


class HuggingFaceHierarchicalResNetClassifier:
    """Lazy wrapper for a bundle's CarFace hierarchical ResNet classifier."""

    def __init__(
        self,
        bundle: ModelBundle,
        *,
        offline_only: bool = True,
        device: str = "cpu",
        processor: Any | None = None,
        model: Any | None = None,
    ) -> None:
        self.bundle = bundle
        self.label_space = LabelSpace.from_labels(bundle.labels)
        self.offline_only = offline_only
        self.device = device
        self._processor = processor
        self._model = model

    @property
    def loaded(self) -> bool:
        return self._processor is not None and self._model is not None

    def classify(self, crop_bgr: np.ndarray) -> HierarchicalScores:
        if crop_bgr.ndim != 3 or crop_bgr.shape[2] != 3 or crop_bgr.size == 0:
            raise ValueError("classifier input must be a non-empty BGR image")
        self._ensure_loaded()
        inputs = self._processor(images=crop_bgr[:, :, ::-1], return_tensors="pt")
        inputs = _move_mapping(inputs, self.device)
        try:
            import torch
        except ImportError as error:
            raise ClassifierUnavailableError(
                "HuggingFaceHierarchicalResNetClassifier requires roadid[ml]"
            ) from error
        with torch.no_grad():
            outputs = self._model(**inputs)
        logits = _hierarchical_logits(outputs, self.label_space)
        return HierarchicalScores(
            body_type=_softmax(logits[0]),
            make=_softmax(logits[1]),
            model_family=_softmax(logits[2]),
        )

    def _ensure_loaded(self) -> None:
        if self.loaded:
            return
        try:
            from transformers import AutoImageProcessor
        except ImportError as error:
            raise ClassifierUnavailableError(
                "HuggingFaceHierarchicalResNetClassifier requires roadid[ml]"
            ) from error
        options = {"local_files_only": self.offline_only}
        try:
            self._processor = AutoImageProcessor.from_pretrained(
                self.bundle.classifier_path, **options
            )
            self._model = _load_bundle_model(self.bundle, self.offline_only)
            self._model.to(self.device)
            self._model.eval()
        except Exception as error:
            mode = "offline bundle" if self.offline_only else "bundle/model source"
            raise ClassifierUnavailableError(
                f"unable to load hierarchical classifier from {mode}: {error}"
            ) from error


HFHierarchicalResNetClassifier = HuggingFaceHierarchicalResNetClassifier


class DeterministicDemoClassifier:
    """Color/shape classifier for deterministic replay, tests, and demos only."""

    profile_label = "deterministic-visual-demo-classifier-test-demo-only"

    def __init__(self, label_space: LabelSpace | None = None) -> None:
        self.label_space = label_space or LabelSpace(
            body_types=("sedan", "suv"),
            makes=("blue-motors", "red-motors"),
            model_families=("blue-line", "red-line"),
        )

    def classify(self, crop_bgr: np.ndarray) -> HierarchicalScores:
        if crop_bgr.ndim != 3 or crop_bgr.shape[2] != 3 or crop_bgr.size == 0:
            raise ValueError("classifier input must be a non-empty BGR image")
        means = crop_bgr.astype(float).mean(axis=(0, 1)) / 255.0
        aspect = crop_bgr.shape[1] / crop_bgr.shape[0]
        signal = float(means[2] - means[0])
        return HierarchicalScores(
            body_type=_softmax(_demo_logits(len(self.label_space.body_types), aspect - 1.5)),
            make=_softmax(_demo_logits(len(self.label_space.makes), signal)),
            model_family=_softmax(_demo_logits(len(self.label_space.model_families), signal)),
        )


class DetectionOnlyClassifier:
    """Keep detected tracks while withholding hierarchy levels that were not trained."""

    profile_label = "pretrained-detr-detection-only"

    def __init__(self) -> None:
        self.label_space = LabelSpace(
            body_types=("vehicle", "body-type-not-trained"),
            makes=("make-not-trained", "make-unavailable"),
            model_families=("model-not-trained", "model-unavailable"),
        )

    def classify(self, crop_bgr: np.ndarray) -> HierarchicalScores:
        if crop_bgr.ndim != 3 or crop_bgr.shape[2] != 3 or crop_bgr.size == 0:
            raise ValueError("classifier input must be a non-empty BGR image")
        abstaining = (0.5, 0.5)
        return HierarchicalScores(
            body_type=abstaining,
            make=abstaining,
            model_family=abstaining,
        )


def _hierarchical_logits(
    outputs: Any, space: LabelSpace
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = ("body_type_logits", "make_logits", "model_family_logits")
    if all(_output_value(outputs, name) is not None for name in names):
        return tuple(_vector(_output_value(outputs, name)) for name in names)  # type: ignore[return-value]
    level_names = ("body_type", "make", "model_family")
    if all(_output_value(outputs, name) is not None for name in level_names):
        return tuple(_vector(_output_value(outputs, name)) for name in level_names)  # type: ignore[return-value]
    flat = _output_value(outputs, "logits")
    if flat is None:
        raise ValueError("classifier output does not contain hierarchical or flat logits")
    vector = _vector(flat)
    if vector.size != space.size:
        raise ValueError(f"classifier emitted {vector.size} logits; expected {space.size}")
    first = space.widths[0]
    second = first + space.widths[1]
    return vector[:first], vector[first:second], vector[second:]


def _output_value(outputs: Any, name: str) -> Any | None:
    if isinstance(outputs, Mapping):
        return outputs.get(name)
    return getattr(outputs, name, None)


def _vector(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    array = np.asarray(value, dtype=float).squeeze()
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError("classifier logits must be one finite vector per hierarchy level")
    return array


def _softmax(values: Sequence[float] | np.ndarray) -> tuple[float, ...]:
    vector = np.asarray(values, dtype=float)
    shifted = vector - np.max(vector)
    exponential = np.exp(shifted)
    return tuple((exponential / exponential.sum()).tolist())


def _normalize(values: Sequence[float]) -> tuple[float, ...]:
    vector = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(vector)) or np.any(vector < 0) or vector.sum() <= 0:
        raise ValueError("scores must be finite, non-negative, and have positive mass")
    return tuple((vector / vector.sum()).tolist())


def _demo_logits(size: int, signal: float) -> np.ndarray:
    if size == 1:
        return np.asarray([1.0])
    return np.linspace(-signal, signal, size)


def _move_mapping(inputs: Any, device: str) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    if isinstance(inputs, dict):
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
    return inputs


def _load_bundle_model(bundle: ModelBundle, offline_only: bool) -> Any:
    try:
        import torch
        import torch.nn as neural
        from transformers import AutoModel
    except ImportError as error:
        raise ClassifierUnavailableError(
            "HuggingFaceHierarchicalResNetClassifier requires roadid[ml]"
        ) from error

    config = json.loads((bundle.classifier_path / "config.json").read_text(encoding="utf-8"))
    backbone = AutoModel.from_pretrained(
        str(bundle.manifest["base_model"]),
        revision=str(bundle.manifest["base_model_revision"]),
        local_files_only=offline_only,
    )
    hidden_sizes = getattr(backbone.config, "hidden_sizes", None)
    feature_dim = (
        hidden_sizes[-1] if hidden_sizes else getattr(backbone.config, "hidden_size", None)
    )
    if not isinstance(feature_dim, int):
        raise ValueError("unable to determine Hugging Face backbone feature width")
    class_counts = dict(
        zip(
            ("body_type", "make", "model_family"),
            bundle_label_widths(bundle),
            strict=True,
        )
    )

    class HierarchicalClassifier(neural.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.heads = neural.ModuleDict(
                {level: neural.Linear(feature_dim, width) for level, width in class_counts.items()}
            )

        def forward(self, pixel_values: Any) -> dict[str, Any]:
            output = self.backbone(pixel_values)
            features = getattr(output, "pooler_output", None)
            if features is None:
                features = getattr(output, "last_hidden_state", output)
            if features.ndim == 4:
                features = features.mean(dim=(-2, -1))
            elif features.ndim == 3:
                features = features.mean(dim=1)
            return {level: head(features) for level, head in self.heads.items()}

    model = HierarchicalClassifier()
    serialization = config.get("serialization", config.get("serialization_format", "safetensors"))
    weights_path = bundle.classifier_path / "model.safetensors"
    if serialization in {"torch_state_dict", "torch"}:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    elif serialization == "safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as error:
            raise ClassifierUnavailableError("safetensors is required for this bundle") from error
        state = load_file(str(weights_path), device="cpu")
    else:
        raise ValueError(f"unsupported classifier serialization: {serialization}")
    model.load_state_dict(state, strict=True)
    return model


def bundle_label_widths(bundle: ModelBundle) -> tuple[int, int, int]:
    return LabelSpace.from_labels(bundle.labels).widths
