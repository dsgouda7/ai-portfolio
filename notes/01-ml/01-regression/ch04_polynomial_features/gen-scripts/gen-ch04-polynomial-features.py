from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.animation_renderer import render_metric_story

STAGES = [
    {
        "label": "Linear fit",
        "value": 55.0,
        "curve": [0.18, 0.62, 0.82, 0.0],
        "caption": "A straight line still underfits curved housing effects.",
    },
    {
        "label": "Quadratic term",
        "value": 50.0,
        "curve": [0.34, 0.28, 0.88, 0.0],
        "caption": "Adding curvature catches non-linear signal.",
    },
    {
        "label": "Feature crosses",
        "value": 48.0,
        "curve": [0.42, 0.16, 0.90, 0.0],
        "caption": "Polynomial interactions shave off the next chunk of error.",
    },
]

if __name__ == "__main__":
    render_metric_story(
        Path(__file__).parent,
        "ch04-polynomial-features-needle",
        "Ch.4 — Polynomial features bend toward the true pattern",
        "MAE ($k)",
        STAGES,
        better="lower",
        style="regression",
    )
