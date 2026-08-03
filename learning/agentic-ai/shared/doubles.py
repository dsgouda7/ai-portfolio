"""Deterministic model and service doubles for offline OrderFlow lessons."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from .fixtures import APPROVAL_POLICY, INVENTORY, SUPPLIER_QUOTES

SKU_PATTERN = re.compile(r"SKU-[A-Z]+-\d{2}")
QUANTITY_PATTERN = re.compile(
    r"(?:order|buy|needs?|procure|requests?|asks?\s+for)\s+(\d+)",
    re.IGNORECASE,
)


class DeterministicModel:
    """A repeatable provider double that exposes model-shaped failure modes."""

    def plain_plan(self, email: str) -> str:
        sku_match = SKU_PATTERN.search(email)
        quantity_match = QUANTITY_PATTERN.search(email)
        sku = sku_match.group(0) if sku_match else "UNKNOWN"
        quantity = int(quantity_match.group(1)) if quantity_match else None
        selector = sum(ord(char) for char in email) % 3
        if selector == 0:
            return f"Use check_inventory for {sku}, quantity {quantity}."
        if selector == 1:
            return json.dumps({"tool": "inventory_lookup", "sku": sku, "qty": quantity})
        return json.dumps({"name": "check_inventory", "arguments": {"sku": sku}})

    def propose_inventory_call(self, email: str) -> dict[str, Any]:
        sku_match = SKU_PATTERN.search(email)
        quantity_match = QUANTITY_PATTERN.search(email)
        return {
            "name": "check_inventory",
            "arguments": {
                "sku": sku_match.group(0) if sku_match else None,
                "quantity": int(quantity_match.group(1)) if quantity_match else None,
            },
        }

    def choose_action(self, state: dict[str, Any]) -> dict[str, Any]:
        """Choose one deterministic next action from explicit workflow state."""
        if state.get("terminal"):
            return {"name": "finish", "arguments": {}}
        if not state.get("parsed"):
            return {"name": "parse_purchase_order", "arguments": {"email": state["email"]}}
        if not state.get("inventory_checked"):
            return {
                "name": "check_inventory",
                "arguments": {"sku": state["sku"], "quantity": state["quantity"]},
            }
        if not state.get("quote_checked"):
            return {"name": "quote_price", "arguments": {"sku": state["sku"]}}
        return {"name": "finish", "arguments": {}}


@dataclass
class FakeClock:
    now_ms: int = 0

    def advance(self, milliseconds: int) -> int:
        self.now_ms += milliseconds
        return self.now_ms


@dataclass
class SupplierServiceDouble:
    clock: FakeClock = field(default_factory=FakeClock)
    attempts: dict[str, int] = field(default_factory=dict)

    def quotes(self, sku: str, timeout_ms: int = 1000) -> list[dict[str, Any]]:
        self.attempts[sku] = self.attempts.get(sku, 0) + 1
        quotes = deepcopy(SUPPLIER_QUOTES.get(sku, []))
        if not quotes:
            raise KeyError(f"unknown_sku:{sku}")
        delay = max(quote["delay_ms"] for quote in quotes)
        self.clock.advance(min(delay, timeout_ms))
        if delay > timeout_ms:
            raise TimeoutError(f"supplier_timeout:{sku}")
        return quotes


@dataclass
class IdempotentPurchaseService:
    commitments: dict[str, dict[str, Any]] = field(default_factory=dict)
    reservations: dict[str, int] = field(default_factory=dict)

    def commit(self, idempotency_key: str, sku: str, quantity: int, supplier: str) -> dict[str, Any]:
        if idempotency_key in self.commitments:
            result = deepcopy(self.commitments[idempotency_key])
            result["duplicate_prevented"] = True
            return result
        if sku not in INVENTORY:
            raise KeyError(f"unknown_sku:{sku}")
        result = {
            "commitment_id": f"COMMIT-{len(self.commitments) + 1:04d}",
            "idempotency_key": idempotency_key,
            "sku": sku,
            "quantity": quantity,
            "supplier": supplier,
            "status": "committed",
            "duplicate_prevented": False,
        }
        self.commitments[idempotency_key] = deepcopy(result)
        return result

    def reserve(self, workflow_id: str, sku: str, quantity: int) -> dict[str, Any]:
        self.reservations[workflow_id] = quantity
        return {"workflow_id": workflow_id, "sku": sku, "reserved": quantity}

    def release(self, workflow_id: str) -> dict[str, Any]:
        released = self.reservations.pop(workflow_id, 0)
        return {"workflow_id": workflow_id, "released": released}


def approval_route(total: float) -> str:
    if total <= APPROVAL_POLICY["auto_approve_max"]:
        return "auto"
    if total <= APPROVAL_POLICY["manager_approve_max"]:
        return "manager"
    return "finance"
