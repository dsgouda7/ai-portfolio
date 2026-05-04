from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.animation_renderer import render_metric_story

STAGES = [
    {
        "label": "Z-score only",
        "value": 45.0,
        "threshold": 0.86,
        "threshold_note": "many frauds missed",
        "caption": "Pure extremeness catches the obvious cases but leaves most structured fraud behind.",
    },
    {
        "label": "+ rules",
        "value": 57.0,
        "threshold": 0.76,
        "threshold_note": "better, still blunt",
        "caption": "Adding hand rules helps, but the overlap is still too large.",
    },
    {
        "label": "IsoForest",
        "value": 72.0,
        "threshold": 0.63,
        "threshold_note": "rare cases pop out",
        "caption": "Path-length scoring separates rare structure from the dense normal cloud.",
    },
    {
        "label": "ROC tuned",
        "value": 80.0,
        "threshold": 0.57,
        "threshold_note": "target recall hit",
        "caption": "Threshold tuning pushes recall toward the production target without flooding false alarms.",
    },
]

if __name__ == "__main__":
    render_metric_story(
        Path(__file__).parent,
        "ch02-isolation-forest-needle",
        "Ch.2 — Isolation Forest lifts fraud recall by isolating rare cases faster",
        "Recall (%)",
        STAGES,
        better="higher",
        style="threshold",
    )
