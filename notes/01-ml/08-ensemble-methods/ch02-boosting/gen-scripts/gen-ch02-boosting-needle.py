from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.animation_renderer import render_metric_story

STAGES = [
    {
        "label": "Mean baseline (92)",
        "value": 92.0,
        "curve": [0.24, 0.02, 1.08, 0.32],
        "caption": "A constant baseline leaves large residuals everywhere and misses most of the housing structure.",
    },
    {
        "label": "1 tree (74)",
        "value": 74.0,
        "curve": [0.31, 0.08, 0.99, 0.20],
        "caption": "The first weak learner knocks out the biggest systematic mistakes.",
    },
    {
        "label": "50 rounds (62)",
        "value": 62.0,
        "curve": [0.36, 0.15, 0.95, 0.11],
        "caption": "Residual fitting keeps shaving down bias after bagging would normally plateau.",
    },
    {
        "label": "200 rounds (55)",
        "value": 55.0,
        "curve": [0.40, 0.20, 0.92, 0.03],
        "caption": "Sequential corrections continue until the remaining residual error is much smaller and more uniform.",
    },
]

if __name__ == "__main__":
    render_metric_story(
        Path(__file__).parent,
        "ch02-boosting-needle",
        "Ch.2 — Boosting drives residual RMSE from 92k to 55k",
        "Residual RMSE ($k)",
        STAGES,
        better="lower",
        style="regression",
    )
