from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.animation_renderer import render_metric_story

STAGES = [
    {
        "label": "Matrix factorization",
        "value": 78.0,
        "centers": [[-2.3, -1.3], [-1.0, 1.2], [0.25, -0.20], [1.90, 1.50], [2.30, -1.10]],
        "caption": "Linear dot products capture broad taste signals but miss non-linear preference patterns.",
    },
    {
        "label": "+ Embedding layers",
        "value": 82.0,
        "centers": [[-2.40, -1.45], [-1.15, 1.35], [0.30, -0.25], [2.05, 1.65], [2.45, -1.20]],
        "caption": "Dedicated learnable embeddings give each user and item a richer, more expressive latent identity.",
    },
    {
        "label": "+ MLP interaction",
        "value": 86.0,
        "centers": [[-2.55, -1.60], [-1.30, 1.55], [0.35, -0.30], [2.20, 1.80], [2.60, -1.30]],
        "caption": "Non-linear layers capture 'likes A but not B' patterns that a simple dot product can never express.",
    },
    {
        "label": "Neural CF",
        "value": 90.0,
        "centers": [[-2.65, -1.75], [-1.40, 1.70], [0.40, -0.40], [2.30, 1.95], [2.75, -1.40]],
        "caption": "Combining linear and non-linear paths lifts Hit@10 well past the matrix factorization ceiling.",
    },
]

if __name__ == "__main__":
    render_metric_story(
        Path(__file__).parent,
        "ch04-neural-cf-needle",
        "Ch.4 — Non-linear interactions push recommendation quality beyond matrix factorization",
        "Hit@10 (%)",
        STAGES,
        better="higher",
        style="clustering",
    )
