from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.animation_renderer import render_metric_story

STAGES = [
    {
        "label": "No monitoring",
        "value": 78.0,
        "curve": [0.46, 0.12, 0.88, 0.28],
        "caption": "Training blind, the model memorises training data long after the optimal generalisation point.",
    },
    {
        "label": "Loss curves",
        "value": 64.0,
        "curve": [0.43, 0.16, 0.90, 0.16],
        "caption": "Visualising train vs validation loss reveals exactly when the generalisation gap starts to open.",
    },
    {
        "label": "Early stop",
        "value": 56.0,
        "curve": [0.40, 0.19, 0.91, 0.07],
        "caption": "Halting at the best validation epoch recovers the optimal weights before late-epoch memorisation.",
    },
    {
        "label": "LR schedule tuned",
        "value": 51.0,
        "curve": [0.39, 0.21, 0.92, 0.03],
        "caption": "Scheduler visualisations confirm the best decay, squeezing out the final MAE improvement.",
    },
]

if __name__ == "__main__":
    render_metric_story(
        Path(__file__).parent,
        "ch08-tensorboard-needle",
        "Ch.8 — TensorBoard monitoring moves the validation metric in the right direction",
        "Test MAE ($k)",
        STAGES,
        better="lower",
        style="regression",
    )
