"""Deterministic smoke tests for the shared OrderFlow fixtures and doubles."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

TRACK_DIR = Path(__file__).resolve().parents[1]
if str(TRACK_DIR) not in sys.path:
    sys.path.insert(0, str(TRACK_DIR))

from shared import (
    APPROVAL_POLICY,
    DeterministicModel,
    IdempotentPurchaseService,
    load_purchase_requests,
    request_by_id,
)


class SharedFixtureTests(unittest.TestCase):
    def test_twenty_unique_requests_exist(self):
        requests = load_purchase_requests()
        self.assertEqual(len(requests), 20)
        self.assertEqual(len({request["request_id"] for request in requests}), 20)

    def test_fixture_copy_is_isolated(self):
        first = load_purchase_requests()
        first[0]["quantity"] = 999
        self.assertEqual(request_by_id("PO-7293")["quantity"], 12)

    def test_provider_double_is_repeatable(self):
        model = DeterministicModel()
        email = request_by_id("PO-7293")["email"]
        self.assertEqual(model.propose_inventory_call(email), model.propose_inventory_call(email))

    def test_idempotency_prevents_duplicate_commitment(self):
        service = IdempotentPurchaseService()
        first = service.commit("PO-7293#create", "SKU-CPU-01", 12, "Northstar")
        second = service.commit("PO-7293#create", "SKU-CPU-01", 12, "Northstar")
        self.assertEqual(first["commitment_id"], second["commitment_id"])
        self.assertTrue(second["duplicate_prevented"])
        self.assertEqual(len(service.commitments), 1)

    def test_policy_thresholds_are_ordered(self):
        self.assertLess(APPROVAL_POLICY["auto_approve_max"], APPROVAL_POLICY["manager_approve_max"])
        self.assertEqual(
            APPROVAL_POLICY["manager_approve_max"],
            APPROVAL_POLICY["finance_approve_above"],
        )


if __name__ == "__main__":
    unittest.main()
