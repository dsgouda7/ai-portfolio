from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.animation_renderer import render_metric_story

STAGES = [
    {
        "label": "Item-CF",
        "value": 68.0,
        "centers": [[-2.2, -1.1], [-0.9, 1.0], [0.2, -0.1], [1.7, 1.3], [2.0, -0.9]],
        "caption": "Neighborhood methods help, but sparsity keeps users and items only loosely aligned.",
    },
    {
        "label": "+ Biases",
        "value": 72.0,
        "centers": [[-2.35, -1.3], [-1.1, 1.25], [0.25, -0.2], [1.9, 1.45], [2.25, -1.05]],
        "caption": "User and item biases explain the obvious popularity effects first.",
    },
    {
        "label": "Latent factors",
        "value": 78.0,
        "centers": [[-2.5, -1.5], [-1.2, 1.45], [0.3, -0.25], [2.1, 1.65], [2.45, -1.2]],
        "caption": "The shared embedding space bridges sparsity and brings similar tastes closer together.",
    },
    {
        "label": "Reg. MF",
        "value": 82.0,
        "centers": [[-2.6, -1.7], [-1.35, 1.7], [0.4, -0.35], [2.25, 1.9], [2.7, -1.35]],
        "caption": "Regularised matrix factorisation pushes hit rate higher while keeping the factors stable.",
    },
]

if __name__ == "__main__":
    render_metric_story(
        Path(__file__).parent,
        "ch03-matrix-factorization-needle",
        "Ch.3 — Latent factors align users and items and lift ranking quality",
        "Hit@10 (%)",
        STAGES,
        better="higher",
        style="clustering",
    )
