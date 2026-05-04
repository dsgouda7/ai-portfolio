from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.animation_renderer import render_metric_story

STAGES = [
    {
        "label": "0 eps (12)",
        "value": 12.0,
        "threshold": 0.82,
        "threshold_note": "Q ≈ uniform noise",
        "caption": "With mostly random actions, the agent collects weak returns and the Q-table is nearly flat.",
    },
    {
        "label": "100 eps (40)",
        "value": 40.0,
        "threshold": 0.67,
        "threshold_note": "good/bad actions separate",
        "caption": "Early TD updates reveal which actions are obviously better, so the return starts climbing fast.",
    },
    {
        "label": "500 eps (66)",
        "value": 66.0,
        "threshold": 0.55,
        "threshold_note": "ε shrinking",
        "caption": "Repeated visits stabilize most state-action values, and the policy becomes much more reliable.",
    },
    {
        "label": "2k eps (84)",
        "value": 84.0,
        "threshold": 0.48,
        "threshold_note": "greedy ≈ optimal",
        "caption": "The learned Q-values are now close to fixed, and the greedy policy captures nearly all available return.",
    },
]

if __name__ == "__main__":
    render_metric_story(
        Path(__file__).parent,
        "ch03-q-learning-needle",
        "Ch.3 — Q-learning lifts average return from 12 to 84",
        "Avg Episode Return",
        STAGES,
        better="higher",
        style="threshold",
    )
