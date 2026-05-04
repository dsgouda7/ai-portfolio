from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.animation_renderer import render_metric_story

STAGES = [
    {
        "label": "1 feature",
        "value": 70.0,
        "curve": [0.0, 0.82, 1.0, 0.0],
        "caption": "Income alone helps, but it still misses key neighborhood structure.",
    },
    {
        "label": "More features",
        "value": 62.0,
        "curve": [0.10, 0.78, 0.86, 0.0],
        "caption": "Latitude, rooms, and occupancy pull the fit closer to reality.",
    },
    {
        "label": "8-feature model",
        "value": 55.0,
        "curve": [0.18, 0.62, 0.82, 0.0],
        "caption": "Combining all core inputs moves the needle again.",
    },
]

if __name__ == "__main__":
    render_metric_story(
        Path(__file__).parent,
        "ch02-multiple-regression-needle",
        "Ch.2 — Multiple regression stacks more evidence",
        "MAE ($k)",
        STAGES,
        better="lower",
        style="regression",
    )
