from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.animation_renderer import render_metric_story

STAGES = [
    {
        "label": "Baseline",
        "value": 72.0,
        "m": 0.56,
        "b": 0.24,
        "bend": -0.02,
        "margin": 0.42,
        "caption": "Baseline setup captures coarse signal but leaves avoidable errors.",
    },
    {
        "label": "Core method",
        "value": 79.0,
        "m": 0.66,
        "b": 0.18,
        "bend": -0.04,
        "margin": 0.54,
        "caption": "The chapter's core method widens separation and improves stability.",
    },
    {
        "label": "Refined setup",
        "value": 84.0,
        "m": 0.74,
        "b": 0.13,
        "bend": -0.07,
        "margin": 0.60,
        "caption": "Refinements move the performance needle further on held-out cases.",
    },
]

if __name__ == "__main__":
    render_metric_story(
        Path(__file__).parent,
        "ch01-ensembles-needle",
        "Ch.1 - Bagging reduces variance and improves robustness",
        "AUC",
        STAGES,
        better="higher",
        style="classification",
    )
