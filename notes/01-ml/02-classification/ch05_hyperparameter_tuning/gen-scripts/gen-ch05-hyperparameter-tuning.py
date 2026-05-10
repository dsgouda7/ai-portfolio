from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.animation_renderer import render_metric_story

STAGES = [
    {
        "label": "Default t=0.50",
        "value": 89.0,
        "threshold": 0.50,
        "threshold_note": "safe but conservative",
        "caption": "Hand-picked settings leave easy performance on the table.",
    },
    {
        "label": "Grid search",
        "value": 90.5,
        "threshold": 0.35,
        "threshold_note": "better C, gamma",
        "caption": "Searching C and gamma tightens the separation.",
    },
    {
        "label": "Per-label tune",
        "value": 91.2,
        "threshold": 0.25,
        "threshold_note": "rare labels recovered",
        "caption": "Per-attribute thresholds improve the imbalanced cases.",
    },
    {
        "label": "Production tune",
        "value": 92.0,
        "threshold": 0.22,
        "threshold_note": "~92% reached",
        "caption": "Systematic tuning pushes FaceAI past the 90% target.",
    },
]

if __name__ == "__main__":
    render_metric_story(
        Path(__file__).parent,
        "ch05-hyperparameter-tuning-needle",
        "Ch.5 — Tuning thresholds keeps the classification needle moving",
        "Validation score (%)",
        STAGES,
        better="higher",
        style="threshold",
    )
