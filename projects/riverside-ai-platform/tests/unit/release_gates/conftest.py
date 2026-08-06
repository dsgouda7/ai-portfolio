from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from release_gates import (  # noqa: E402
    DatasetRef,
    Domain,
    EvaluatorRef,
    GatePolicy,
    MetricRecord,
    MetricRequirement,
    Observation,
    Slice,
    Threshold,
    ThresholdOperator,
)


@pytest.fixture
def minimal_policy() -> GatePolicy:
    return GatePolicy(
        policy_id="fixture-policy",
        version="1.0.0",
        requirements=tuple(
            MetricRequirement(
                domain=domain,
                metric_id=f"{domain.value}_score",
                threshold=Threshold(
                    operator=ThresholdOperator.GTE,
                    value=0.5,
                    unit="ratio",
                ),
            )
            for domain in Domain
        ),
    )


@pytest.fixture
def passing_metrics(minimal_policy: GatePolicy) -> tuple[MetricRecord, ...]:
    return tuple(
        MetricRecord(
            domain=requirement.domain,
            metric_id=requirement.metric_id,
            dataset=DatasetRef(id=f"{requirement.domain.value}-fixture", version="1.0"),
            evaluator=EvaluatorRef(id="static-fixture-evaluator", version="1.0.0"),
            slice=Slice(name="population", value="all"),
            threshold=requirement.threshold,
            observed=Observation(value=0.75, unit="ratio"),
        )
        for requirement in minimal_policy.requirements
    )
