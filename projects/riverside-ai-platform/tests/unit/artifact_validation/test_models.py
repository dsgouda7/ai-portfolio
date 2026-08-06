from pathlib import Path

import pytest
from pydantic import ValidationError

from src.artifact_validation import ChatCompletionRequest, ModelReleaseManifest


PROJECT_ROOT = Path(__file__).parents[3]


def test_valid_release_fixture_matches_typed_contract() -> None:
    fixture = PROJECT_ROOT / "tests" / "fixtures" / "valid" / "model-release-manifest.json"

    manifest = ModelReleaseManifest.model_validate_json(fixture.read_text(encoding="utf-8"))

    assert manifest.contract_version == "1.0.0"
    assert manifest.adapter.type == "lora"


def test_mutable_release_version_fixture_is_rejected() -> None:
    fixture = (
        PROJECT_ROOT
        / "tests"
        / "fixtures"
        / "invalid"
        / "model-release-manifest.mutable-version.json"
    )

    with pytest.raises(ValidationError):
        ModelReleaseManifest.model_validate_json(fixture.read_text(encoding="utf-8"))


def test_request_rejects_duplicate_stop_sequences() -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(
            {
                "model": "riverside-editor",
                "messages": [{"role": "user", "content": "Continue this scene."}],
                "max_input_tokens": 128,
                "max_tokens": 32,
                "stream": False,
                "stop": ["END", "END"],
            }
        )
