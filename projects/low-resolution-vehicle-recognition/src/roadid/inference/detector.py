"""Vehicle detectors. Hugging Face dependencies load only on first inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from roadid.contracts import Detection, FramePacket


class DetectorUnavailableError(RuntimeError):
    """Raised when the configured detector cannot be loaded."""


class VehicleDetector(Protocol):
    def detect(
        self, frame: FramePacket | np.ndarray, frame_id: int | None = None
    ) -> tuple[Detection, ...]: ...


@dataclass(frozen=True, slots=True)
class DetrConfig:
    model_id: str = "facebook/detr-resnet-50"
    revision: str | None = "no_timm"
    score_threshold: float = 0.7
    vehicle_classes: tuple[str, ...] = ("car", "truck", "bus")
    offline_only: bool = True
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("score_threshold must be in [0, 1]")
        if not self.vehicle_classes:
            raise ValueError("vehicle_classes cannot be empty")


class HuggingFaceDetrDetector:
    """Lazy DETR wrapper that always emits original-frame pixel coordinates."""

    def __init__(
        self,
        config: DetrConfig | None = None,
        *,
        processor: Any | None = None,
        model: Any | None = None,
    ) -> None:
        self.config = config or DetrConfig()
        self._processor = processor
        self._model = model

    @property
    def loaded(self) -> bool:
        return self._processor is not None and self._model is not None

    def detect(
        self, frame: FramePacket | np.ndarray, frame_id: int | None = None
    ) -> tuple[Detection, ...]:
        image_bgr, resolved_frame_id = _frame_parts(frame, frame_id)
        self._ensure_loaded()
        image_rgb = image_bgr[:, :, ::-1]
        inputs = self._processor(images=image_rgb, return_tensors="pt")
        inputs = _move_mapping(inputs, self.config.device)

        try:
            import torch
        except ImportError as error:
            raise DetectorUnavailableError(
                "HuggingFaceDetrDetector requires the 'ml' optional dependencies"
            ) from error

        with torch.no_grad():
            outputs = self._model(**inputs)
        height, width = image_bgr.shape[:2]
        target_sizes = torch.tensor([[height, width]])
        processed = self._processor.post_process_object_detection(
            outputs,
            threshold=self.config.score_threshold,
            target_sizes=target_sizes,
        )
        if len(processed) != 1:
            raise RuntimeError("DETR processor returned an unexpected batch size")
        result = processed[0]
        id_to_label = getattr(getattr(self._model, "config", None), "id2label", {})
        detections: list[Detection] = []
        for score, label_id, box in zip(
            result["scores"], result["labels"], result["boxes"], strict=True
        ):
            class_name = str(id_to_label.get(int(_scalar(label_id)), label_id)).lower()
            if class_name not in {value.lower() for value in self.config.vehicle_classes}:
                continue
            bbox = _original_bbox(_array(box), width, height)
            detections.append(
                Detection(
                    frame_id=resolved_frame_id,
                    bbox_xyxy=bbox,
                    class_name=class_name,
                    confidence=float(_scalar(score)),
                )
            )
        return tuple(sorted(detections, key=lambda item: item.bbox_xyxy))

    def _ensure_loaded(self) -> None:
        if self.loaded:
            return
        try:
            from transformers import AutoImageProcessor, DetrForObjectDetection
        except ImportError as error:
            raise DetectorUnavailableError(
                "HuggingFaceDetrDetector requires the 'ml' optional dependencies"
            ) from error
        load_options = {
            "revision": self.config.revision,
            "local_files_only": self.config.offline_only,
        }
        try:
            self._processor = AutoImageProcessor.from_pretrained(
                self.config.model_id, **load_options
            )
            self._model = DetrForObjectDetection.from_pretrained(
                self.config.model_id, **load_options
            )
            self._model.to(self.config.device)
            self._model.eval()
        except Exception as error:
            mode = "offline cache" if self.config.offline_only else "configured model source"
            raise DetectorUnavailableError(
                f"unable to load {self.config.model_id} from {mode}: {error}"
            ) from error


HFDetrVehicleDetector = HuggingFaceDetrDetector


class DeterministicVehicleDetector:
    """Detect saturated synthetic rectangles for replay, tests, and demos only."""

    profile_label = "deterministic-synthetic-visuals-test-demo-only"

    def __init__(
        self,
        *,
        minimum_area: int = 64,
        confidence: float = 0.99,
        class_name: str = "car",
    ) -> None:
        if minimum_area <= 0:
            raise ValueError("minimum_area must be positive")
        self.minimum_area = minimum_area
        self.confidence = confidence
        self.class_name = class_name

    def detect(
        self, frame: FramePacket | np.ndarray, frame_id: int | None = None
    ) -> tuple[Detection, ...]:
        image_bgr, resolved_frame_id = _frame_parts(frame, frame_id)
        maximum = image_bgr.max(axis=2)
        minimum = image_bgr.min(axis=2)
        mask = (maximum >= 80) & ((maximum - minimum) >= 40)
        detections = []
        for x1, y1, x2, y2, area in _connected_components(mask):
            if area < self.minimum_area:
                continue
            detections.append(
                Detection(
                    frame_id=resolved_frame_id,
                    bbox_xyxy=(x1, y1, x2, y2),
                    class_name=self.class_name,
                    confidence=self.confidence,
                )
            )
        return tuple(sorted(detections, key=lambda item: item.bbox_xyxy))


def _frame_parts(frame: FramePacket | np.ndarray, frame_id: int | None) -> tuple[np.ndarray, int]:
    if isinstance(frame, FramePacket):
        return frame.image_bgr, frame.frame_id
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("detector input must have shape (height, width, 3)")
    if frame_id is None:
        raise ValueError("frame_id is required when detecting a raw image")
    return frame, frame_id


def _original_bbox(box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    if box.shape != (4,) or not np.all(np.isfinite(box)):
        raise ValueError("detector returned an invalid bounding box")
    x1 = max(0, min(width - 1, int(np.floor(box[0]))))
    y1 = max(0, min(height - 1, int(np.floor(box[1]))))
    x2 = max(x1 + 1, min(width, int(np.ceil(box[2]))))
    y2 = max(y1 + 1, min(height, int(np.ceil(box[3]))))
    return x1, y1, x2, y2


def _move_mapping(inputs: Any, device: str) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    if isinstance(inputs, dict):
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
    return inputs


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value, dtype=float)


def _scalar(value: Any) -> float:
    return float(value.item() if hasattr(value, "item") else value)


def _connected_components(mask: np.ndarray) -> tuple[tuple[int, int, int, int, int], ...]:
    visited = np.zeros(mask.shape, dtype=bool)
    components = []
    height, width = mask.shape
    for start_y, start_x in np.argwhere(mask):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        xs: list[int] = []
        ys: list[int] = []
        while stack:
            y, x = stack.pop()
            xs.append(x)
            ys.append(y)
            for next_y in range(max(0, y - 1), min(height, y + 2)):
                for next_x in range(max(0, x - 1), min(width, x + 2)):
                    if mask[next_y, next_x] and not visited[next_y, next_x]:
                        visited[next_y, next_x] = True
                        stack.append((next_y, next_x))
        components.append((min(xs), min(ys), max(xs) + 1, max(ys) + 1, len(xs)))
    return tuple(components)
