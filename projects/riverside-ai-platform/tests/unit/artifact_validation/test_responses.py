from datetime import datetime, timezone

from src.artifact_validation import (
    DeploymentMetadata,
    GenerationResult,
    completion_response,
    stream_events,
)


def _deployment() -> DeploymentMetadata:
    return DeploymentMetadata(
        contract_version="1.0.0",
        kind="deployment_metadata",
        environment="staging",
        release_id="riverside-editor-2026-08-05",
        model_alias="riverside-editor",
        deployment_name="riverside-staging-green",
        deployment_slot="green",
        region="eastus2",
        runtime="azureml-riverside-runtime",
        runtime_version="1.0.0",
        index_version="1.0.0",
        deployed_at=datetime(2026, 8, 5, 12, tzinfo=timezone.utc),
        source_commit="0123456789abcdef0123456789abcdef01234567",
    )


def test_non_streaming_response_has_normalized_usage() -> None:
    response = completion_response(
        result=GenerationResult("Aria sees the lights.", 8, 5, "stop"),
        model_alias="riverside-editor",
        deployment=_deployment(),
        trace_id="0" * 32,
        completion_id="chatcmpl-testvalue",
        created=1,
    )

    assert response["object"] == "chat.completion"
    assert response["usage"] == {
        "prompt_tokens": 8,
        "completion_tokens": 5,
        "total_tokens": 13,
    }


def test_stream_events_are_deterministic_and_finish_with_usage() -> None:
    events = list(
        stream_events(
            result=GenerationResult("Aria sees the lights.", 8, 5, "stop"),
            model_alias="riverside-editor",
            deployment=_deployment(),
            trace_id="0" * 32,
            completion_id="chatcmpl-testvalue",
            created=1,
        )
    )

    assert "".join(event["choices"][0]["delta"].get("content", "") for event in events) == (
        "Aria sees the lights."
    )
    assert events[-1]["choices"][0]["finish_reason"] == "stop"
    assert events[-1]["usage"]["total_tokens"] == 13
