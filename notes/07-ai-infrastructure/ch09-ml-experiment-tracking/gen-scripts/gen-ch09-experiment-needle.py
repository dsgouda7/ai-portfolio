from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.animation_renderer import render_metric_story

STAGES = [
    {
        "label": "No tracking (2 days)",
        "value": 2880.0,
        "curve": [0.24, 0.02, 1.08, 0.32],
        "caption": "Manual notebook experiments with scattered parameters and results buried in print statements; reproducibility requires detective work.",
    },
    {
        "label": "Basic logging (12 hr)",
        "value": 720.0,
        "curve": [0.28, 0.06, 1.00, 0.22],
        "caption": "CSV logs capture metrics but parameters live in separate files; comparisons still require manual spreadsheet work.",
    },
    {
        "label": "Structured runs (3 hr)",
        "value": 180.0,
        "curve": [0.33, 0.12, 0.94, 0.10],
        "caption": "JSON configs link params to metrics, enabling faster comparison but artifact versioning remains manual.",
    },
    {
        "label": "MLflow tracking (30 min)",
        "value": 30.0,
        "curve": [0.38, 0.18, 0.90, 0.02],
        "caption": "Automatic logging of code, params, metrics, and artifacts; searchable UI + model registry enables instant debugging and rollback.",
    },
]

if __name__ == "__main__":
    render_metric_story(
        Path(__file__).parent.parent / "img",
        "ch09-experiment-needle",
        "Ch.9 — ML experiment tracking cuts debugging from 2 days to 30 minutes",
        "Debugging Time (minutes)",
        STAGES,
        better="lower",
        style="regression",
    )
