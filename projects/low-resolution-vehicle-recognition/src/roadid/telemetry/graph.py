"""Stable pipeline graph snapshots derived from ordered events."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from roadid.contracts import JSONValue, PipelineEvent, StageStatus
from roadid.telemetry.events import PIPELINE_STAGES
from roadid.telemetry.metrics import StageLatencyStats


def serialize_graph_snapshot(
    events: Iterable[PipelineEvent],
    *,
    latencies: Mapping[str, StageLatencyStats] | None = None,
) -> dict[str, JSONValue]:
    latest: dict[str, PipelineEvent] = {}
    for event in sorted(events, key=lambda item: item.sequence_id):
        latest[event.stage] = event

    nodes: list[JSONValue] = []
    for order, stage in enumerate(PIPELINE_STAGES):
        event = latest.get(stage)
        node: dict[str, JSONValue] = {
            "id": stage,
            "label": stage.replace("_", " ").title(),
            "order": order,
            "status": event.status.value if event else StageStatus.PENDING.value,
            "sequence_id": event.sequence_id if event else None,
            "frame_id": event.frame_id if event else None,
            "track_id": event.track_id if event else None,
            "started_at": event.to_dict()["started_at"] if event else None,
            "duration_ms": event.duration_ms if event else None,
            "input_summary": dict(event.input_summary) if event else {},
            "output_summary": dict(event.output_summary) if event else {},
            "warning": event.warning if event else None,
            "error_code": event.error_code if event else None,
        }
        if latencies is not None:
            node["latency"] = latencies[stage].to_dict()
        nodes.append(node)

    edges: list[JSONValue] = [
        {
            "id": f"{source}->{target}",
            "source": source,
            "target": target,
        }
        for source, target in zip(PIPELINE_STAGES, PIPELINE_STAGES[1:], strict=False)
    ]
    return {"nodes": nodes, "edges": edges}
