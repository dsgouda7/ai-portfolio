from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.animation_renderer import render_metric_story

STAGES = [
    {
        "label": "t=0.50",
        "value": 12.0,
        "threshold": 0.50,
        "threshold_note": "rare positives missed",
        "caption": "Accuracy looks high, but Bald recall is only 12%.",
    },
    {
        "label": "t=0.30",
        "value": 45.0,
        "threshold": 0.30,
        "threshold_note": "recall jumps",
        "caption": "Lowering the threshold catches far more minority-class faces.",
    },
    {
        "label": "t=0.25",
        "value": 68.0,
        "threshold": 0.25,
        "threshold_note": "best trade-off",
        "caption": "The right metric reveals a much stronger operating point.",
    },
]

if __name__ == "__main__":
    render_metric_story(
        Path(__file__).parent,
        "ch03-metrics-needle",
        "Ch.3 — Better metrics expose and fix the accuracy paradox",
        "Bald recall (%)",
        STAGES,
        better="higher",
        style="threshold",
    )
