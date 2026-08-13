from concurrent.futures import ThreadPoolExecutor

from roadid.telemetry import PIPELINE_STAGES, EventRecorder


def test_retained_events_and_graph_use_sequence_order_under_concurrency() -> None:
    recorder = EventRecorder("run-integration", capacity=200)

    def process_frame(frame_id: int) -> None:
        for stage in PIPELINE_STAGES:
            recorder.start(stage, frame_id=frame_id, input_summary={"frame": frame_id})
            recorder.complete(stage, frame_id=frame_id, duration_ms=float(frame_id + 1))

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(process_frame, range(4)))

    events = recorder.all_events()
    assert [event.sequence_id for event in events] == list(range(1, 81))
    assert all(
        events[index].sequence_id < events[index + 1].sequence_id
        for index in range(len(events) - 1)
    )
    graph = recorder.graph_snapshot()
    assert [node["id"] for node in graph["nodes"]] == list(PIPELINE_STAGES)
    assert all(node["latency"]["count"] == 4 for node in graph["nodes"])
