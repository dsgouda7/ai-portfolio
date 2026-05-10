from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.animation_renderer import render_metric_story

STAGES = [
    {
        "label": "Before validation",
        "value": 0.48,
        "centers": [[-2.5, -1.5], [-1.2, 1.5], [0.3, -0.3], [2.0, 1.7], [2.5, -1.2]],
        "caption": "The segmentation looks good, but it still needs a quantitative check.",
    },
    {
        "label": "Metric sweep",
        "value": 0.50,
        "centers": [[-2.55, -1.6], [-1.3, 1.6], [0.35, -0.35], [2.1, 1.8], [2.65, -1.3]],
        "caption": "Silhouette, DBI, and CHI make the quality trade-offs explicit.",
    },
    {
        "label": "K=5 accepted",
        "value": 0.52,
        "centers": [[-2.6, -1.8], [-1.4, 1.7], [0.4, -0.4], [2.2, 1.9], [2.8, -1.4]],
        "caption": "Crossing the 0.5 threshold validates the business-friendly 5-segment choice.",
    },
]

if __name__ == "__main__":
    render_metric_story(
        Path(__file__).parent,
        "ch03-unsupervised-metrics-needle",
        "Ch.3 — Validation metrics confirm the segment quality threshold",
        "Silhouette",
        STAGES,
        better="higher",
        style="clustering",
    )
