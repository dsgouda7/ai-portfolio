from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.animation_renderer import render_metric_story

STAGES = [
    {
        "label": "Mean baseline",
        "value": 92.0,
        "curve": [0.0, 0.0, 2.4, 0.0],
        "caption": "One flat guess for every house leaves a large average miss.",
    },
    {
        "label": "Income line",
        "value": 70.0,
        "curve": [0.0, 0.82, 1.0, 0.0],
        "caption": "A single linear signal already drops the error sharply.",
    },
]

if __name__ == "__main__":
    render_metric_story(
        Path(__file__).parent,
        "ch01-linear-regression-needle",
        "Ch.1 — Linear regression lowers the first big miss",
        "MAE ($k)",
        STAGES,
        better="lower",
        style="regression",
    )
