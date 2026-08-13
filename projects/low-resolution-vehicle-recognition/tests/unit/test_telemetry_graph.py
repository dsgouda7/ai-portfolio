from roadid.telemetry import PIPELINE_STAGES, EventRecorder


def test_graph_has_stable_order_and_latency_aggregation() -> None:
    recorder = EventRecorder("run-graph")
    recorder.start("source_acquisition", frame_id=1)
    recorder.complete("source_acquisition", frame_id=1, duration_ms=4.0)
    recorder.start("source_acquisition", frame_id=2)
    recorder.complete("source_acquisition", frame_id=2, duration_ms=6.0)
    recorder.start("frame_validation", frame_id=2)

    graph = recorder.graph_snapshot()
    nodes = graph["nodes"]

    assert [node["id"] for node in nodes] == list(PIPELINE_STAGES)
    assert [node["order"] for node in nodes] == list(range(len(PIPELINE_STAGES)))
    assert nodes[0]["status"] == "completed"
    assert nodes[0]["latency"] == {
        "count": 2,
        "total_ms": 10.0,
        "mean_ms": 5.0,
        "min_ms": 4.0,
        "max_ms": 6.0,
    }
    assert nodes[1]["status"] == "running"
    assert all(node["status"] == "pending" for node in nodes[2:])
    assert [edge["source"] for edge in graph["edges"]] == list(PIPELINE_STAGES[:-1])
