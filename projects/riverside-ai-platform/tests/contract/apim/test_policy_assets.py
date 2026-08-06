from __future__ import annotations

import re
import unittest

from .policy_static import (
    APIM_ROOT,
    FRAGMENT_ROOT,
    committed_urls,
    declared_named_values,
    expected_fragment_ids,
    forbidden_metric_dimensions,
    fragment_manifest,
    included_fragment_ids,
    load_json,
    parse_xml,
    referenced_named_values,
    scenario_cases,
    token_metric_dimensions,
    xml_assets,
    assert_scenario_markers,
)


class ApimPolicyAssetTests(unittest.TestCase):
    def test_all_json_and_xml_assets_parse(self) -> None:
        for path in APIM_ROOT.rglob("*.json"):
            with self.subTest(path=path):
                load_json(path)
        for path in xml_assets():
            with self.subTest(path=path):
                self.assertIn(parse_xml(path).tag, {"policies", "fragment"})

    def test_composed_fragment_order_matches_manifest(self) -> None:
        self.assertEqual(included_fragment_ids(), expected_fragment_ids())

    def test_manifest_points_to_every_fragment_file(self) -> None:
        manifest_files = {item["file"] for item in fragment_manifest()}
        actual_files = {path.name for path in FRAGMENT_ROOT.glob("*.xml")}
        self.assertEqual(manifest_files, actual_files)

    def test_every_named_value_reference_is_declared(self) -> None:
        self.assertEqual(referenced_named_values() - declared_named_values(), set())

    def test_named_value_manifest_contains_no_secrets(self) -> None:
        document = load_json(APIM_ROOT / "parameters" / "named-values.json")
        self.assertTrue(all(item["secret"] is False for item in document["named_values"]))
        serialized = str(document).casefold()
        for forbidden in ("api_key", "apikey", "connection_string", "password"):
            self.assertNotIn(forbidden, serialized)

    def test_openapi_matches_frozen_route_and_token_bounds(self) -> None:
        openapi = load_json(APIM_ROOT / "api" / "openapi.json")
        operation = openapi["paths"]["/v1/chat/completions"]["post"]
        request_schema = openapi["components"]["schemas"]["ChatCompletionRequest"]
        self.assertEqual(operation["security"], [{"entraBearer": []}])
        self.assertEqual(request_schema["properties"]["model"]["enum"], ["riverside-editor"])
        self.assertEqual(request_schema["properties"]["max_input_tokens"]["maximum"], 8192)
        self.assertEqual(request_schema["properties"]["max_tokens"]["maximum"], 2048)
        self.assertNotIn("tenant_id", request_schema["properties"])
        self.assertNotIn("tenant_tier", request_schema["properties"])

    def test_tenant_tier_is_derived_from_validated_jwt(self) -> None:
        text = (FRAGMENT_ROOT / "tenant-context.xml").read_text(encoding="utf-8")
        self.assertIn('context.Variables[&quot;riversideClientJwt&quot;]', text)
        self.assertIn("riverside_tenant_tier", text)
        self.assertNotIn("context.Request.Headers.GetValueOrDefault", text)

    def test_backend_auth_uses_managed_identity_only(self) -> None:
        text = (FRAGMENT_ROOT / "backend-identity.xml").read_text(encoding="utf-8")
        self.assertIn("authentication-managed-identity", text)
        self.assertNotIn("api-key", text.casefold())
        self.assertNotIn("set-header", text)

    def test_retry_is_bounded_and_streams_are_not_replayed(self) -> None:
        text = (FRAGMENT_ROOT / "bounded-retry.xml").read_text(encoding="utf-8")
        self.assertIn("riverside-max-retries", text)
        self.assertIn("Retry-After", text)
        self.assertIn("DateTimeOffset.TryParse", text)
        self.assertIn("429, 502, 503, 504", text)
        self.assertRegex(text, re.compile(r"riversideStreamRequested.*buffer-response=\"false\"", re.DOTALL))

    def test_backend_pool_has_priority_weight_and_circuit_breakers(self) -> None:
        text = (APIM_ROOT / "backends" / "backends.bicep").read_text(encoding="utf-8")
        for marker in (
            "type: 'Pool'",
            "priority: bluePriority",
            "priority: greenPriority",
            "weight: blueWeight",
            "weight: greenWeight",
            "acceptRetryAfter: true",
            "min: 429",
            "min: 500",
            "max: 599",
        ):
            self.assertIn(marker, text)

    def test_token_metric_dimensions_are_allowlisted_and_bounded(self) -> None:
        expected = {
            "deployment.environment",
            "model.release_id",
            "model.alias",
            "deployment.name",
            "tenant.tier",
        }
        self.assertEqual(token_metric_dimensions(), expected)
        self.assertEqual(forbidden_metric_dimensions(), set())
        self.assertLessEqual(len(token_metric_dimensions()), 5)

    def test_extension_points_default_to_disabled(self) -> None:
        document = load_json(APIM_ROOT / "parameters" / "named-values.json")
        examples = {item["name"]: item.get("example") for item in document["named_values"]}
        self.assertEqual(examples["riverside-content-safety-enabled"], "false")
        self.assertEqual(examples["riverside-semantic-cache-enabled"], "false")

    def test_cache_key_uses_trusted_authorization_context(self) -> None:
        text = (FRAGMENT_ROOT / "cache-lookup-extension.xml").read_text(encoding="utf-8")
        self.assertIn("riversideTenantId", text)
        self.assertIn("riversideActorId", text)
        self.assertIn("riverside-model-release-id", text)
        self.assertNotIn("context.Request.Headers", text)

    def test_committed_urls_are_non_resolving_documentation_placeholders(self) -> None:
        self.assertEqual(committed_urls(), {"https://gateway.example.invalid"})

    def test_behavior_cases_have_static_evidence(self) -> None:
        case_ids: set[str] = set()
        for case in scenario_cases():
            with self.subTest(case=case["id"]):
                self.assertNotIn(case["id"], case_ids)
                case_ids.add(case["id"])
                assert_scenario_markers(case)


if __name__ == "__main__":
    unittest.main()
