from roadid.telemetry import EventRecorder


def test_eviction_tracks_gap_without_reusing_sequence_ids() -> None:
    recorder = EventRecorder("run-retention", capacity=3)

    for frame_id in range(3):
        recorder.start("source_acquisition", frame_id=frame_id)
        recorder.complete("source_acquisition", frame_id=frame_id, duration_ms=1.0)

    batch = recorder.events_after(0)
    assert [event.sequence_id for event in batch.events] == [4, 5, 6]
    assert batch.dropped_count == 3
    assert batch.first_available_sequence == 4
    assert batch.next_sequence == 7
    assert batch.gap is True

    recorder.start("frame_validation", frame_id=3)
    assert recorder.all_events()[-1].sequence_id == 7
    assert recorder.first_available_sequence == 5

    graph = recorder.graph_snapshot()
    assert graph["nodes"][0]["status"] == "completed"
    assert graph["nodes"][1]["status"] == "running"
