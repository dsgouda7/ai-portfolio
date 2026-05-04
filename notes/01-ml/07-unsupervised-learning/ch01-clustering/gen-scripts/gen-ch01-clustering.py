from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.animation_renderer import render_metric_story

STAGES = [
    {
        "label": "Random init",
        "value": 0.18,
        "centers": [[-3.0, 2.2], [-2.4, -2.4], [0.1, 0.8], [2.9, 2.6], [3.2, -2.4]],
        "caption": "With poor centroids, the customer groups overlap heavily.",
    },
    {
        "label": "K-Means step",
        "value": 0.32,
        "centers": [[-2.8, -1.3], [-1.8, 1.3], [0.0, -0.1], [2.0, 1.2], [2.6, -1.0]],
        "caption": "Centroids move toward denser regions and structure starts to appear.",
    },
    {
        "label": "5 segments",
        "value": 0.42,
        "centers": [[-2.6, -1.8], [-1.4, 1.7], [0.4, -0.4], [2.2, 1.9], [2.8, -1.4]],
        "caption": "The final clustering gives SegmentAI its first actionable segmentation.",
    },
]

if __name__ == "__main__":
    render_metric_story(
        Path(__file__).parent,
        "ch01-clustering-needle",
        "Ch.1 — Clustering sharpens as centroids settle",
        "Silhouette",
        STAGES,
        better="higher",
        style="clustering",
    )
