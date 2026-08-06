"""Static contract tests for bounded telemetry dimensions."""

import pytest

from telemetry.conventions import METRIC_ATTRIBUTE_KEYS, MetricContext, build_metric_attributes
from telemetry.conventions import retrieval_top_k_bucket, token_bucket


@pytest.fixture
def context() -> MetricContext:
    return MetricContext(
        service_name="riverside-rag-orchestrator",
        environment="staging",
        release_id="riverside-editor-2026-08-05",
        model_alias="riverside-editor",
        deployment_name="riverside-staging-green",
        region="eastus2",
        route="/v1/chat/completions",
        tenant_tier="standard",
    )


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        (0, "0"),
        (1, "1-128"),
        (128, "1-128"),
        (129, "129-512"),
        (512, "129-512"),
        (513, "513-2048"),
        (2048, "513-2048"),
        (2049, "2049-8192"),
        (8192, "2049-8192"),
    ],
)
def test_token_bucket_boundaries(tokens: int, expected: str) -> None:
    assert token_bucket(tokens) == expected


@pytest.mark.parametrize("tokens", [-1, 8193])
def test_token_bucket_rejects_values_outside_contract(tokens: int) -> None:
    with pytest.raises(ValueError):
        token_bucket(tokens)


@pytest.mark.parametrize(
    ("top_k", "expected"),
    [(None, "none"), (1, "1-5"), (5, "1-5"), (6, "6-10"), (10, "6-10"), (11, "11-20"), (20, "11-20")],
)
def test_retrieval_bucket_boundaries(top_k: int | None, expected: str) -> None:
    assert retrieval_top_k_bucket(top_k) == expected


def test_metric_attributes_are_exactly_the_v1_allowlist(context: MetricContext) -> None:
    attributes = build_metric_attributes(
        context,
        outcome="success",
        cache_result="miss",
        prompt_tokens=1024,
        output_tokens=256,
        retrieval_top_k=8,
    )

    assert attributes.keys() == METRIC_ATTRIBUTE_KEYS
    assert "request.id" not in attributes
    assert "tenant.id" not in attributes
    assert "trace_id" not in attributes
    assert attributes["gen_ai.usage.prompt_tokens_bucket"] == "513-2048"
    assert attributes["retrieval.top_k_bucket"] == "6-10"


def test_metric_context_rejects_uncontrolled_dimensions() -> None:
    with pytest.raises(ValueError):
        MetricContext(
            service_name="customer-specific-service",
            environment="staging",
            release_id="release-1",
            model_alias="riverside-editor",
            deployment_name="riverside-staging-green",
            region="eastus2",
            route="/customers/123/chat",
            tenant_tier="tenant-123",
        )


def test_metric_attributes_enforce_api_output_limit_and_error_semantics(context: MetricContext) -> None:
    with pytest.raises(ValueError, match="API limit"):
        build_metric_attributes(
            context,
            outcome="success",
            cache_result="miss",
            prompt_tokens=1,
            output_tokens=2049,
        )

    with pytest.raises(ValueError, match="require an error code"):
        build_metric_attributes(
            context,
            outcome="timeout",
            cache_result="miss",
            prompt_tokens=1,
            output_tokens=0,
        )
