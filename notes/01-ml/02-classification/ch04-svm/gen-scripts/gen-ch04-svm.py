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
        "caption": "Linear separation is decent, but the margin is still tight.",
    },
    {
        "label": "Wide margin",
        "value": 89.0,
        "m": 0.70,
        "b": 0.14,
        "bend": 0.00,
        "margin": 0.60,
        "caption": "SVM maximizes the safe corridor around the decision boundary.",
    },
    {
        "label": "RBF kernel",
        "value": 90.0,
        "m": 0.78,
        "b": 0.18,
        "bend": -0.07,
        "margin": 0.42,
        "caption": "Kernelized separation cleans up the remaining non-linear edge cases.",
    },
]

if __name__ == "__main__":
    render_metric_story(
        Path(__file__).parent,
        "ch04-svm-needle",
        "Ch.4 — SVM widens the margin and reduces boundary error",
        "Accuracy (%)",
        STAGES,
        better="higher",
        style="classification",
    )
