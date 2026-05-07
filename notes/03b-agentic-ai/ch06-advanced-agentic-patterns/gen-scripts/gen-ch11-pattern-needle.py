from pathlib import Path
import sys

# Add shared animation renderer to path
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.animation_renderer import render_metric_story

STAGES = [
    {
        "label": "Single-pass",
        "value": 8.0,
        "curve": [0.30, 0.15, 0.85, 0.70],
        "caption": "Standard LLM with prompt engineering — baseline error rate.",
    },
    {
        "label": "+ Reflection",
        "value": 4.2,
        "curve": [0.35, 0.20, 0.88, 0.45],
        "caption": "Self-critique loop cuts errors by ~50% via iterative refinement.",
    },
    {
        "label": "+ Ensemble",
        "value": 1.8,
        "curve": [0.38, 0.25, 0.90, 0.20],
        "caption": "Multi-model voting filters hallucinations and edge cases.",
    },
    {
        "label": "+ Constitutional",
        "value": 0.7,
        "curve": [0.40, 0.28, 0.92, 0.05],
        "caption": "Principle-based self-correction pushes below 1% target.",
    },
]

if __name__ == "__main__":
    render_metric_story(
        Path(__file__).parent,
        "ch11-pattern-needle",
        "Ch.11 — Agentic patterns drive error rate below 1%",
        "Error Rate (%)",
        STAGES,
        better="lower",
        style="ai",  # Use AI-themed styling
        target_line=1.0,  # Show 1% target threshold
    )
