from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.animation_renderer import render_metric_story

STAGES = [
    {
        "label": "Loose threshold",
        "value": 0.62,
        "threshold": 0.35,
        "threshold_note": "too many false alerts",
        "caption": "A permissive threshold catches anomalies but floods operators with noise.",
    },
    {
        "label": "Balanced",
        "value": 0.71,
        "threshold": 0.52,
        "threshold_note": "precision/recall balance",
        "caption": "Balanced thresholding improves detection quality with manageable alert volume.",
    },
    {
        "label": "Calibrated",
        "value": 0.79,
        "threshold": 0.63,
        "threshold_note": "cleaner production signal",
        "caption": "Calibration produces cleaner anomaly signals and better downstream triage.",
    },
]

if __name__ == "__main__":
    render_metric_story(
        Path(__file__).parent,
        "ch06-production-needle",
        "Ch.6 - Production tuning reduces alert noise",
        "Precision@K",
        STAGES,
        better="higher",
        style="threshold",
    )
