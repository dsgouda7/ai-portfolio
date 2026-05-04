from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.animation_renderer import render_metric_story

STAGES = [
    {
        "label": "Hand-picked",
        "value": 38.0,
        "curve": [0.39, 0.22, 0.91, 0.03],
        "caption": "A decent model is in place, but its settings are still guesses.",
    },
    {
        "label": "Grid search",
        "value": 36.0,
        "curve": [0.40, 0.20, 0.90, 0.02],
        "caption": "Systematic search improves the bias-variance trade-off.",
    },
    {
        "label": "Random/Bayes",
        "value": 34.0,
        "curve": [0.41, 0.19, 0.89, 0.01],
        "caption": "Exploring smarter regions of the search space drops the error again.",
    },
    {
        "label": "Production fit",
        "value": 32.0,
        "curve": [0.42, 0.17, 0.88, 0.00],
        "caption": "Tuned settings deliver the cleanest, lowest-error curve.",
    },
]

if __name__ == "__main__":
    render_metric_story(
        Path(__file__).parent,
        "ch07-hyperparameter-tuning-needle",
        "Ch.7 — Tuning keeps moving the error needle down",
        "MAE ($k)",
        STAGES,
        better="lower",
        style="regression",
    )
