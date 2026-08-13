"""Privacy-safe JSON Lines export for retained run telemetry."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from roadid.telemetry.recorder import EventRecorder


def export_jsonl(
    recorder: EventRecorder,
    destination: str | Path | TextIO | None = None,
) -> str:
    batch = recorder.events_after(0)
    metadata = {
        "record_type": "run_metadata",
        "run_id": recorder.run_id,
        "exported_at": datetime.now(UTC).isoformat(),
        "metadata": recorder.metadata,
        "retention": {
            "capacity": recorder.capacity,
            "retained_event_count": len(batch.events),
            "dropped_count": batch.dropped_count,
            "first_available_sequence": batch.first_available_sequence,
            "next_sequence": batch.next_sequence,
        },
    }
    records = [metadata]
    if batch.gap:
        records.append(
            {
                "record_type": "sequence_gap",
                "after_sequence": 0,
                "first_available_sequence": batch.first_available_sequence,
                "dropped_count": batch.dropped_count,
            }
        )
    records.extend({"record_type": "pipeline_event", **event.to_dict()} for event in batch.events)
    content = "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n"

    if destination is None:
        return content
    if hasattr(destination, "write"):
        destination.write(content)
    else:
        Path(destination).write_text(content, encoding="utf-8")
    return content
