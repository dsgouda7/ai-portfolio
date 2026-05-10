from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.animation_renderer import render_metric_story

STAGES = [
    {
        "label": "MSE on binary",
        "value": 68.0,
        "m": 0.22,
        "b": -0.15,
        "bend": 0.12,
        "margin": 0.70,
        "caption": "MSE treats every probability gap as squared geometry loss, missing the shape of binary decisions.",
    },
    {
        "label": "Binary CE",
        "value": 74.0,
        "m": 0.40,
        "b": -0.05,
        "bend": 0.06,
        "margin": 0.52,
        "caption": "Cross-entropy amplifies confident wrong predictions, steering weights toward calibrated probabilities.",
    },
    {
        "label": "+ Numeric stable",
        "value": 79.0,
        "m": 0.56,
        "b": 0.10,
        "bend": 0.02,
        "margin": 0.36,
        "caption": "Log-sum-exp tricks prevent under/overflow at extreme scores, closing a hidden accuracy gap.",
    },
    {
        "label": "Matched loss",
        "value": 84.0,
        "m": 0.72,
        "b": 0.20,
        "bend": -0.03,
        "margin": 0.24,
        "caption": "Aligning the loss geometry with the task moves accuracy at zero model complexity cost.",
    },
]

if __name__ == "__main__":
    render_metric_story(
        Path(__file__).parent,
        "ch07-mle-loss-needle",
        "Ch.7 — Matching the loss function to the task visibly lifts accuracy",
        "Accuracy (%)",
        STAGES,
        better="higher",
        style="classification",
    )
