from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.animation_renderer import render_metric_story

STAGES = [
    {
        "label": "Early model",
        "value": 70.0,
        "curve": [0.0, 0.82, 1.0, 0.0],
        "caption": "Metrics expose how far the first model is from usable.",
    },
    {
        "label": "Poly model",
        "value": 48.0,
        "curve": [0.42, 0.16, 0.90, 0.0],
        "caption": "MAE and RMSE both confirm the residuals are shrinking.",
    },
    {
        "label": "Validated winner",
        "value": 38.0,
        "curve": [0.39, 0.22, 0.91, 0.02],
        "caption": "The evaluation dashboard shows the final gap to error is much smaller.",
    },
]

if __name__ == "__main__":
    render_metric_story(
        Path(__file__).parent,
        "ch06-metrics-needle",
        "Ch.6 — Metrics make the residual drop visible",
        "Validation error ($k)",
        STAGES,
        better="lower",
        style="regression",
    )
