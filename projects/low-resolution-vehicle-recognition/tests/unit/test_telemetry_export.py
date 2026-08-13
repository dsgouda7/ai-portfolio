import json

from roadid.telemetry import EventRecorder


def test_jsonl_contains_run_and_explicit_gap_metadata(tmp_path) -> None:
    recorder = EventRecorder(
        "run-export",
        capacity=2,
        metadata={"source": "replay", "remote_url": "https://camera.example/private"},
    )
    for frame_id in range(2):
        recorder.start("source_acquisition", frame_id=frame_id)
        recorder.complete("source_acquisition", frame_id=frame_id, duration_ms=1.0)

    destination = tmp_path / "run.jsonl"
    content = recorder.export_jsonl(destination)
    records = [json.loads(line) for line in content.splitlines()]

    assert destination.read_text(encoding="utf-8") == content
    assert records[0]["record_type"] == "run_metadata"
    assert records[0]["retention"]["dropped_count"] == 2
    assert records[1] == {
        "after_sequence": 0,
        "dropped_count": 2,
        "first_available_sequence": 3,
        "record_type": "sequence_gap",
    }
    assert [record["sequence_id"] for record in records[2:]] == [3, 4]
    assert "camera.example" not in content
