from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.animation_renderer import render_metric_story

STAGES = [
    {
        "label": "Best single (88%)",
        "value": 88.0,
        "m": 0.60,
        "b": 0.20,
        "bend": -0.03,
        "margin": 0.48,
        "caption": "The strongest individual model sets the starting ceiling, but some examples are still systematically missed.",
    },
    {
        "label": "+ bagged view (90%)",
        "value": 90.0,
        "m": 0.68,
        "b": 0.22,
        "bend": -0.04,
        "margin": 0.38,
        "caption": "Adding a decorrelated bagging signal reduces variance and closes part of the gap.",
    },
    {
        "label": "+ boosted view (91.5%)",
        "value": 91.5,
        "m": 0.74,
        "b": 0.24,
        "bend": -0.04,
        "margin": 0.30,
        "caption": "A boosted learner contributes a different bias profile, giving the meta-learner more useful signal.",
    },
    {
        "label": "Stacked (93%)",
        "value": 93.0,
        "m": 0.80,
        "b": 0.25,
        "bend": -0.05,
        "margin": 0.22,
        "caption": "The stack learns when to trust each base model and pushes accuracy beyond any single member.",
    },
]

if __name__ == "__main__":
    render_metric_story(
        Path(__file__).parent,
        "ch05-stacking-needle",
        "Ch.5 — Stacking lifts validation accuracy from 88% to 93%",
        "Validation Accuracy (%)",
        STAGES,
        better="higher",
        style="classification",
    )
