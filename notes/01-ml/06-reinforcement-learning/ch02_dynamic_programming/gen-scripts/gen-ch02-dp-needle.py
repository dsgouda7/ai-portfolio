from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.animation_renderer import render_metric_story

STAGES = [
    {
        "label": "Init (Δ=8.20)",
        "value": 8.20,
        "curve": [0.45, 0.00, 0.00, 0.40],
        "caption": "All values start at zero, so the Bellman backup still has the full error budget to erase.",
    },
    {
        "label": "5 sweeps (Δ=3.60)",
        "value": 3.60,
        "curve": [0.38, 0.12, 0.40, 0.22],
        "caption": "A few synchronous sweeps propagate reward information out from the goal and cut the error by more than half.",
    },
    {
        "label": "20 sweeps (Δ=0.82)",
        "value": 0.82,
        "curve": [0.41, 0.16, 0.72, 0.08],
        "caption": "The value wave reaches distant states, and the remaining Bellman mismatch becomes small everywhere.",
    },
    {
        "label": "Converged (Δ=0.04)",
        "value": 0.04,
        "curve": [0.42, 0.16, 0.90, 0.02],
        "caption": "Once Δ falls below θ, the optimal value function and policy are effectively locked in.",
    },
]

if __name__ == "__main__":
    render_metric_story(
        Path(__file__).parent,
        "ch02-dynamic-programming-needle",
        "Ch.2 — Value iteration drives Bellman error from 8.20 to 0.04",
        "Max Bellman Error Δ",
        STAGES,
        better="lower",
        style="regression",
    )
