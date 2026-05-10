from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.animation_renderer import render_metric_story

STAGES = [
    {
        "label": "Raw space",
        "value": 0.42,
        "centers": [[-2.3, -1.2], [-1.0, 1.2], [0.3, -0.2], [1.8, 1.5], [2.1, -1.0]],
        "caption": "In the original high-dimensional view, groups still overlap.",
    },
    {
        "label": "PCA 2D",
        "value": 0.46,
        "centers": [[-2.5, -1.5], [-1.2, 1.5], [0.3, -0.3], [2.0, 1.7], [2.5, -1.2]],
        "caption": "Projection strips noise and makes the segment geometry clearer.",
    },
    {
        "label": "Cleaner view",
        "value": 0.48,
        "centers": [[-2.6, -1.8], [-1.4, 1.7], [0.4, -0.4], [2.2, 1.9], [2.8, -1.4]],
        "caption": "The 2D embedding raises quality and makes the clusters explainable.",
    },
]

if __name__ == "__main__":
    render_metric_story(
        Path(__file__).parent,
        "ch02-dimensionality-reduction-needle",
        "Ch.2 — Dimensionality reduction separates the segments further",
        "Silhouette",
        STAGES,
        better="higher",
        style="clustering",
    )
