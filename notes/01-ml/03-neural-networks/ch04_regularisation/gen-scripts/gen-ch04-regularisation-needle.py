from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.animation_renderer import render_metric_story

STAGES = [
    {
        "label": "No reg",
        "value": 85.0,
        "curve": [0.48, 0.10, 0.86, 0.30],
        "caption": "The network memorises noisy districts, so test error balloons on unseen areas.",
    },
    {
        "label": "+ L2",
        "value": 66.0,
        "curve": [0.43, 0.15, 0.89, 0.15],
        "caption": "Weight decay shrinks unstable coefficients and removes some of the wobble.",
    },
    {
        "label": "+ Dropout",
        "value": 58.0,
        "curve": [0.40, 0.19, 0.91, 0.07],
        "caption": "Randomly dropping units forces more robust, shared features.",
    },
    {
        "label": "+ Early stop",
        "value": 52.0,
        "curve": [0.39, 0.21, 0.92, 0.03],
        "caption": "Stopping at the best validation epoch closes most of the generalisation gap.",
    },
]

if __name__ == "__main__":
    render_metric_story(
        Path(__file__).parent,
        "ch04-regularisation-needle",
        "Ch.4 — Regularisation visibly closes the generalisation gap",
        "Test MAE ($k)",
        STAGES,
        better="lower",
        style="regression",
    )
