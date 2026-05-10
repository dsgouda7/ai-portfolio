from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.animation_renderer import render_metric_story

STAGES = [
    {
        "label": "Overfit poly",
        "value": 48.0,
        "curve": [0.44, 0.16, 0.88, 0.28],
        "caption": "Extra flexibility starts chasing noise instead of signal.",
    },
    {
        "label": "Ridge shrinkage",
        "value": 41.0,
        "curve": [0.40, 0.20, 0.90, 0.10],
        "caption": "L2 shrinkage damps unstable coefficients and improves generalization.",
    },
    {
        "label": "Elastic Net",
        "value": 38.0,
        "curve": [0.39, 0.22, 0.91, 0.03],
        "caption": "Shrink + select pushes SmartVal below the $40k target.",
    },
]

if __name__ == "__main__":
    render_metric_story(
        Path(__file__).parent,
        "ch05-regularization-needle",
        "Ch.5 — Regularization removes the noisy wobble",
        "MAE ($k)",
        STAGES,
        better="lower",
        style="regression",
    )
