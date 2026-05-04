from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.animation_renderer import render_metric_story

STAGES = [
    {
        "label": "Naive split",
        "value": 58.0,
        "m": 0.15,
        "b": -0.30,
        "bend": 0.02,
        "margin": 0.90,
        "caption": "A rough hand-made rule leaves many faces on the wrong side.",
    },
    {
        "label": "Log-odds fit",
        "value": 81.0,
        "m": 0.45,
        "b": 0.05,
        "bend": 0.00,
        "margin": 0.65,
        "caption": "The sigmoid boundary starts separating smiling vs non-smiling cases.",
    },
    {
        "label": "Calibrated boundary",
        "value": 88.0,
        "m": 0.62,
        "b": 0.18,
        "bend": -0.02,
        "margin": 0.45,
        "caption": "Calibration and a better threshold move the classification needle clearly.",
    },
]

if __name__ == "__main__":
    render_metric_story(
        Path(__file__).parent,
        "ch01-logistic-regression-needle",
        "Ch.1 — Logistic regression pushes accuracy above the baseline",
        "Accuracy (%)",
        STAGES,
        better="higher",
        style="classification",
    )
