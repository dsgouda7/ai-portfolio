"""Bounded metric attributes defined by the frozen v1 telemetry contract."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Final, Literal

MetricAttributes = dict[str, str]
Outcome = Literal["success", "error", "rejected", "timeout"]
CacheResult = Literal["hit", "miss", "bypass", "error"]
ErrorCode = Literal[
    "none",
    "invalid_request",
    "unauthorized",
    "forbidden",
    "policy_violation",
    "overloaded",
    "timeout",
    "backend_failure",
    "release_unavailable",
    "internal_error",
]

METRIC_ATTRIBUTE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "service.name",
        "deployment.environment",
        "model.release_id",
        "model.alias",
        "deployment.name",
        "cloud.region",
        "http.route",
        "outcome.status",
        "cache.result",
        "gen_ai.usage.prompt_tokens_bucket",
        "gen_ai.usage.output_tokens_bucket",
        "tenant.tier",
        "error.code",
        "retrieval.top_k_bucket",
    }
)

_SERVICES = frozenset(
    {
        "riverside-gateway",
        "riverside-rag-orchestrator",
        "riverside-model-endpoint",
        "riverside-indexer",
    }
)
_ENVIRONMENTS = frozenset({"dev", "staging", "production"})
_ROUTES = frozenset({"/v1/chat/completions", "/health", "/ready"})
_TENANT_TIERS = frozenset({"standard", "premium", "internal"})
_OUTCOMES = frozenset({"success", "error", "rejected", "timeout"})
_CACHE_RESULTS = frozenset({"hit", "miss", "bypass", "error"})
_ERROR_CODES = frozenset(
    {
        "none",
        "invalid_request",
        "unauthorized",
        "forbidden",
        "policy_violation",
        "overloaded",
        "timeout",
        "backend_failure",
        "release_unavailable",
        "internal_error",
    }
)
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_DEPLOYMENT = re.compile(r"^riverside-(dev|staging|production)-(blue|green)$")
_REGION = re.compile(r"^[a-z0-9-]{2,64}$")


def _require_member(name: str, value: str, allowed: frozenset[str]) -> None:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}")


@dataclass(frozen=True, slots=True)
class MetricContext:
    """Controlled deployment dimensions shared by every metric instrument."""

    service_name: str
    environment: str
    release_id: str
    model_alias: str
    deployment_name: str
    region: str
    route: str
    tenant_tier: str

    def __post_init__(self) -> None:
        _require_member("service_name", self.service_name, _SERVICES)
        _require_member("environment", self.environment, _ENVIRONMENTS)
        _require_member("route", self.route, _ROUTES)
        _require_member("tenant_tier", self.tenant_tier, _TENANT_TIERS)
        if self.model_alias != "riverside-editor":
            raise ValueError("model_alias must be 'riverside-editor'")
        if not _IDENTIFIER.fullmatch(self.release_id):
            raise ValueError("release_id must satisfy the v1 identifier contract")
        if not _DEPLOYMENT.fullmatch(self.deployment_name):
            raise ValueError("deployment_name must satisfy the v1 deployment pattern")
        if not _REGION.fullmatch(self.region):
            raise ValueError("region must satisfy the v1 region contract")


def token_bucket(tokens: int) -> str:
    """Map an exact token count to the bounded v1 metric bucket."""

    if isinstance(tokens, bool) or not isinstance(tokens, int):
        raise TypeError("tokens must be an integer")
    if tokens < 0 or tokens > 8192:
        raise ValueError("tokens must be between 0 and 8192")
    if tokens == 0:
        return "0"
    if tokens <= 128:
        return "1-128"
    if tokens <= 512:
        return "129-512"
    if tokens <= 2048:
        return "513-2048"
    return "2049-8192"


def retrieval_top_k_bucket(top_k: int | None) -> str:
    """Map an optional retrieval depth to the bounded v1 metric bucket."""

    if top_k is None:
        return "none"
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be an integer or None")
    if not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20")
    if top_k <= 5:
        return "1-5"
    if top_k <= 10:
        return "6-10"
    return "11-20"


def build_metric_attributes(
    context: MetricContext,
    *,
    outcome: Outcome,
    cache_result: CacheResult,
    prompt_tokens: int,
    output_tokens: int,
    error_code: ErrorCode = "none",
    retrieval_top_k: int | None = None,
) -> MetricAttributes:
    """Build the complete, closed set of v1 metric labels."""

    _require_member("outcome", outcome, _OUTCOMES)
    _require_member("cache_result", cache_result, _CACHE_RESULTS)
    _require_member("error_code", error_code, _ERROR_CODES)
    if output_tokens > 2048:
        raise ValueError("output_tokens must not exceed the v1 API limit of 2048")
    if (outcome == "success") != (error_code == "none"):
        raise ValueError("successful outcomes require error_code 'none'; other outcomes require an error code")
    attributes = {
        "service.name": context.service_name,
        "deployment.environment": context.environment,
        "model.release_id": context.release_id,
        "model.alias": context.model_alias,
        "deployment.name": context.deployment_name,
        "cloud.region": context.region,
        "http.route": context.route,
        "outcome.status": outcome,
        "cache.result": cache_result,
        "gen_ai.usage.prompt_tokens_bucket": token_bucket(prompt_tokens),
        "gen_ai.usage.output_tokens_bucket": token_bucket(output_tokens),
        "tenant.tier": context.tenant_tier,
        "error.code": error_code,
        "retrieval.top_k_bucket": retrieval_top_k_bucket(retrieval_top_k),
    }
    if attributes.keys() != METRIC_ATTRIBUTE_KEYS:
        raise AssertionError("metric attributes drifted from the v1 allowlist")
    return attributes
