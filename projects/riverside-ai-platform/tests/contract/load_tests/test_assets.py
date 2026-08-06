"""Static checks for Azure Load Testing and staged Locust assets."""

import json
from pathlib import Path

LOAD_TESTS = Path(__file__).parents[3] / "load-tests"


def test_load_stages_are_complete_and_ordered() -> None:
    stages = json.loads((LOAD_TESTS / "stages.json").read_text(encoding="utf-8"))

    assert [stage["name"] for stage in stages] == ["warm", "steady", "target", "overload", "recovery"]
    assert stages[3]["users"] > stages[2]["users"]
    assert stages[4]["users"] == stages[1]["users"]


def test_azure_config_contains_supported_locust_and_gate_fields() -> None:
    config = (LOAD_TESTS / "azure-load-test.yaml").read_text(encoding="utf-8")

    assert "version: v0.1" in config
    assert "testType: Locust" in config
    assert "engineInstances: __RIVERSIDE_ENGINE_INSTANCES__" in config
    assert "chat.completions.total.target: percentage(error) > 1" in config
    assert "chat.completions.total.overload: percentage(error) > 25" in config
    assert "p95(response_time_ms)" in config
    assert "errorPercentage: 20" in config
    assert "RIVERSIDE_TOKEN_SCOPE" in config
    assert "RIVERSIDE_MANAGED_IDENTITY_CLIENT_ID" in config
    assert "RIVERSIDE_API_TOKEN" not in config


def test_synthetic_requests_contain_no_identity_or_secret_fields() -> None:
    requests = [
        json.loads(line)
        for line in (LOAD_TESTS / "synthetic-requests.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    forbidden = {"user_id", "request_id", "tenant_id", "api_key", "authorization"}
    assert requests
    assert all(not forbidden.intersection(request) for request in requests)
    assert all(request["model"] == "riverside-editor" for request in requests)
