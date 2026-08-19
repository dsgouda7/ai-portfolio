"""Typed WildScope feed, observation, and prediction contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class WildlifeFeed:
    feed_id: str
    name: str
    place_id: int
    country: str
    habitat: str

    def public_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ObservationPhoto:
    observation_id: int
    photo_id: int
    observed_at: str
    created_at: str
    taxon_id: int | None
    scientific_name: str
    common_name: str | None
    photo_url: str
    license_code: str | None
    attribution: str | None
    quality_grade: str
    latitude: float | None = None
    longitude: float | None = None
    positional_accuracy: float | None = None
    coordinates_obscured: bool = False
    original_width: int | None = None
    original_height: int | None = None
    taxon_group: str | None = None


@dataclass(frozen=True, slots=True)
class ModelPrediction:
    label: str
    confidence: float
    model_version: str
    trained_at: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("prediction confidence must be in [0, 1]")
