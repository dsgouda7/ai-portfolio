"""Shared inputs, ground truth, and helpers used by every experiment pipe."""
from __future__ import annotations

import sys
from pathlib import Path

# Allow importing from the parent package without installing locally.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context_optimizer_benchmark import (  # noqa: E402
    MOCK_INCIDENT_PROMPT,
    build_mock_log_cache,
    CompressedIncident,
)

INCIDENT_PROMPT: str = MOCK_INCIDENT_PROMPT

# Ground truth: what a correct root-cause analysis must identify.
GROUND_TRUTH = {
    "primary_root_cause": (
        "CosmosDB timeout with substatus 21012 causes cascading failures in order-service "
        "and payment-service via CosmosClient.ReadItemAsync / PaymentConnector.SubmitAsync."
    ),
    "affected_services": ["order-service", "payment-service", "api-gateway", "ingress-nginx"],
    "error_codes": ["21012", "504", "499"],
    "key_entities": ["CosmosDB", "AKS", "ingress-nginx", "order-service", "payment-service"],
    "stack_trace_symbols": ["ReadItemAsync", "SubmitAsync"],
    "node_ips": ["10.42.7.19", "10.42.8.44"],
    "metrics": ["p95", "8.7s", "17.6%", "220ms"],
    "immediate_checks": [
        "CosmosDB RU consumption",
        "retry policy configuration",
        "ingress upstream timeout settings",
        "connection pool exhaustion",
    ],
}

# Keywords that MUST appear in a correct answer (required).
REQUIRED_KEYWORDS: list[str] = [
    "cosmos",
    "21012",
    "timeout",
    "order-service",
    "payment",
    "ingress",
    "retry",
    "latency",
]

# High-value but optional keywords (bonus precision markers).
BONUS_KEYWORDS: list[str] = [
    "ReadItemAsync",
    "SubmitAsync",
    "10.42",
    "ru",
    "substatus",
    "504",
    "499",
    "cascade",
    "connection pool",
    "partition",
]


def get_log_corpus(n_lines: int = 1050) -> list[str]:
    """Build the deterministic mock log corpus used across all pipes."""
    return build_mock_log_cache(total_lines=n_lines)


def estimate_tokens(text: str) -> int:
    """Approximate token count (GPT-family: ~4 chars/token)."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Mock canned answers — used when provider="mock" so we can illustrate
# quality *differences* between pipeline designs without real LLM inference.
# These are deliberately calibrated to show what each architecture would
# likely produce in practice.
# ---------------------------------------------------------------------------

MOCK_ANSWER_PIPE_A = (
    "There appears to be some issue with the checkout flow. The logs show multiple errors. "
    "Database connections look bad. There are timeout errors in the system. The latency has "
    "increased significantly. The deployment at 01:55 may be related. You should check your "
    "database and potentially restart some services. Error rate is high. Consider rolling back."
)

MOCK_ANSWER_PIPE_OOTB = (
    "Based on the retrieved logs, the system is experiencing timeouts. The CosmosDB connection "
    "seems to be having issues based on the error patterns in the logs. The ingress controller "
    "is also reporting upstream timeouts. The checkout service is returning 504 errors. "
    "You should investigate the CosmosDB configuration and possibly increase the timeout limits. "
    "The retry policy may also be contributing to the problem."
)

MOCK_ANSWER_PIPE_C = (
    "Root cause (high confidence): CosmosDB RU saturation causing substatus=21012 timeouts. "
    "The cascade path is: CosmosClient.ReadItemAsync timeout → order-service failure → "
    "PaymentConnector.SubmitAsync cancellation. Ingress logs confirm 'upstream timed out while "
    "reading response header' on pods 10.42.7.19 and 10.42.8.44 (nodepool np-user-03).\n\n"
    "Supporting evidence from targeted log retrieval:\n"
    "- 21012 substatus appears on CosmosDB read path (partition=tenant-445, ru_charge=128.44)\n"
    "- p95 climbed from 220ms to 8.7s; error_rate=17.6%\n"
    "- Stack trace: System.TimeoutException at ReadItemAsync:line 214 → SubmitAsync:line 87\n\n"
    "Immediate mitigations:\n"
    "1. Increase CosmosDB provisioned RU or enable autoscale\n"
    "2. Add jitter + exponential backoff to retry policy (current retries amplify RU pressure)\n"
    "3. Set ingress proxy_read_timeout > 8s for /v1/checkout upstream\n"
    "4. Isolate order-service CosmosDB calls behind circuit breaker\n\n"
    "Next observability checks:\n"
    "- RU consumption graph per partition (last 2h)\n"
    "- Retry burst metrics on order-service\n"
    "- CosmosDB connection pool saturation\n"
    "- Ingress upstream queue depth on np-user-03"
)
