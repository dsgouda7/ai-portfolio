from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.animation_renderer import render_metric_story

STAGES = [
    {
        "label": "Baseline",
        "value": 92.0,
        "curve": [0.0, 0.0, 2.2, 0.0],
        "caption": "A simple baseline leaves substantial residual error.",
    },
    {
        "label": "Core method",
        "value": 72.0,
        "curve": [0.16, 0.46, 1.2, 0.1],
        "caption": "Applying the chapter method captures stronger structure in the data.",
    },
    {
        "label": "Refined setup",
        "value": 58.0,
        "curve": [0.38, 0.20, 0.95, 0.0],
        "caption": "Refinement and tuning reduce error while improving generalization.",
    },
]

if __name__ == "__main__":
    render_metric_story(
        Path(__file__).parent,
        "ch02-collaborative-filtering-needle",
        "Ch.2 - Collaborative filtering captures user similarity",
        "RMSE",
        STAGES,
        better="lower",
        style="regression",
    )
