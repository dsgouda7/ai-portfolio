"""Validation-only calibration identities and selective threshold fitting."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict

import numpy as np

from roadid.contracts import CalibrationContract


def softmax(logits: np.ndarray, *, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    values = np.asarray(logits, dtype=np.float64) / temperature
    values = values - values.max(axis=-1, keepdims=True)
    exponent = np.exp(values)
    return exponent / exponent.sum(axis=-1, keepdims=True)


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    values = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.int64)
    valid = labels >= 0
    if not valid.any():
        return 1.0
    values, labels = values[valid], labels[valid]
    candidates = np.geomspace(0.25, 4.0, 161)
    losses = []
    for candidate in candidates:
        probabilities = softmax(values, temperature=float(candidate))
        losses.append(
            -np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1)).mean()
        )
    return float(candidates[int(np.argmin(losses))])


def select_threshold(
    confidences: Sequence[float], correct: Sequence[bool], *, target_precision: float
) -> float:
    if not 0 < target_precision <= 1:
        raise ValueError("target_precision must be in (0, 1]")
    scores = np.asarray(confidences, dtype=np.float64)
    outcomes = np.asarray(correct, dtype=bool)
    if scores.shape != outcomes.shape or scores.ndim != 1 or not len(scores):
        raise ValueError("confidence and correctness vectors must be non-empty and aligned")
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("confidences must be in [0, 1]")

    best_threshold = 1.0
    best_coverage = -1.0
    for threshold in sorted(set(scores.tolist()) | {1.0}):
        accepted = scores >= threshold
        if not accepted.any():
            continue
        precision = float(outcomes[accepted].mean())
        coverage = float(accepted.mean())
        if precision >= target_precision and coverage > best_coverage:
            best_threshold = float(threshold)
            best_coverage = coverage
    return best_threshold


def split_identity_sha256(identities: Iterable[str]) -> str:
    normalized = sorted(set(identities))
    encoded = json.dumps(normalized, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def create_calibration_contract(
    *,
    method: str,
    dataset_manifest_sha256: str,
    validation_identities: Iterable[str],
    test_identities: Iterable[str],
    thresholds: Mapping[str, float],
) -> CalibrationContract:
    validation = set(validation_identities)
    test = set(test_identities)
    overlap = validation & test
    if overlap:
        raise ValueError(f"validation and test identities overlap: {sorted(overlap)[:3]}")
    if not validation or not test:
        raise ValueError("calibration requires non-empty validation and sealed test identities")
    return CalibrationContract(
        method=method,
        dataset_manifest_sha256=dataset_manifest_sha256,
        validation_split_sha256=split_identity_sha256(validation),
        test_split_sha256=split_identity_sha256(test),
        body_threshold=float(thresholds["body_type"]),
        make_threshold=float(thresholds["make"]),
        model_threshold=float(thresholds["model_family"]),
    )


def contract_dict(contract: CalibrationContract) -> dict[str, object]:
    return asdict(contract)
