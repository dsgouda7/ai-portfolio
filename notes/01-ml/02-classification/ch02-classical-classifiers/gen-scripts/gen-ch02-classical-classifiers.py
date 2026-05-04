from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.animation_renderer import render_metric_story

STAGES = [
    {
        "label": "LogReg",
        "value": 88.0,
        "m": 0.62,
        "b": 0.18,
        "bend": -0.02,
        "margin": 0.45,
        "caption": "The linear classifier is solid, but edge cases remain.",
    },
    {
        "label": "kNN",
        "value": 86.0,
        "m": 0.55,
        "b": 0.10,
        "bend": 0.05,
        "margin": 0.35,
        "caption": "Instance-based rules can help locally, but they can wobble on noise.",
    },
    {
        "label": "Tree rules",
        "value": 87.0,
        "m": 0.70,
        "b": 0.22,
        "bend": -0.05,
        "margin": 0.32,
        "caption": "Rule-based splits make the boundary more expressive.",
    },
    {
        "label": "Ensemble winner",
        "value": 89.0,
        "m": 0.78,
        "b": 0.24,
        "bend": -0.04,
        "margin": 0.26,
        "caption": "Combining classical methods nudges the score upward again.",
    },
]

if __name__ == "__main__":
    render_metric_story(
        Path(__file__).parent,
        "ch02-classical-classifiers-needle",
        "Ch.2 — Comparing classical classifiers moves the boundary around the hard cases",
        "Accuracy (%)",
        STAGES,
        better="higher",
        style="classification",
    )
