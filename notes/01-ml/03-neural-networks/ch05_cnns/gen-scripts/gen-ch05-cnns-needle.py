from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.animation_renderer import render_metric_story

STAGES = [
    {
        "label": "Dense net",
        "value": 68.0,
        "m": 0.36,
        "b": -0.15,
        "bend": 0.10,
        "margin": 0.70,
        "caption": "A dense vision baseline treats each pixel independently and misses local structure.",
    },
    {
        "label": "Edge filters",
        "value": 78.0,
        "m": 0.50,
        "b": 0.00,
        "bend": 0.05,
        "margin": 0.48,
        "caption": "Shared 3×3 filters start detecting edges and tidy-vs-distressed textures anywhere in the grid.",
    },
    {
        "label": "+ Pooling",
        "value": 84.0,
        "m": 0.66,
        "b": 0.12,
        "bend": -0.01,
        "margin": 0.34,
        "caption": "Pooling removes brittle location dependence and makes the classifier more stable.",
    },
    {
        "label": "CNN stack",
        "value": 90.0,
        "m": 0.78,
        "b": 0.22,
        "bend": -0.04,
        "margin": 0.24,
        "caption": "Local filters plus hierarchy deliver a much cleaner boundary and a strong accuracy lift.",
    },
]

if __name__ == "__main__":
    render_metric_story(
        Path(__file__).parent,
        "ch05-cnns-needle",
        "Ch.5 — CNN building blocks move image classification accuracy upward",
        "Accuracy (%)",
        STAGES,
        better="higher",
        style="classification",
    )
