"""Shared deterministic support for the OrderFlow agentic AI track."""

from .doubles import (
    DeterministicModel,
    FakeClock,
    IdempotentPurchaseService,
    SupplierServiceDouble,
    approval_route,
)
from .fixtures import (
    APPROVAL_POLICY,
    INVENTORY,
    POLICY_DOCUMENTS,
    PURCHASE_REQUESTS,
    SUPPLIER_QUOTES,
    load_inventory,
    load_policy_documents,
    load_purchase_requests,
    load_supplier_quotes,
    request_by_id,
)
from .metrics import TraceEvent, TraceRecorder, estimate_tokens, exact_rate, percentile, stable_hash

__all__ = [
    "APPROVAL_POLICY",
    "DeterministicModel",
    "FakeClock",
    "INVENTORY",
    "IdempotentPurchaseService",
    "POLICY_DOCUMENTS",
    "PURCHASE_REQUESTS",
    "SUPPLIER_QUOTES",
    "SupplierServiceDouble",
    "TraceEvent",
    "TraceRecorder",
    "approval_route",
    "estimate_tokens",
    "exact_rate",
    "load_inventory",
    "load_policy_documents",
    "load_purchase_requests",
    "load_supplier_quotes",
    "percentile",
    "request_by_id",
    "stable_hash",
]
