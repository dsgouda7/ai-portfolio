from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.animation_renderer import render_metric_story

STAGES = [
    {
        "label": "IsoForest baseline",
        "value": 72.0,
        "threshold": 0.65,
        "threshold_note": "72% recall",
        "caption": "Path-length scoring misses structured fraud that hides within normal-looking feature ranges.",
    },
    {
        "label": "AE trained",
        "value": 79.0,
        "threshold": 0.58,
        "threshold_note": "reconstruction gap widens",
        "caption": "The autoencoder learns normal transaction structure and fails to reconstruct anomalies, widening the score gap.",
    },
    {
        "label": "+ Bottleneck tuned",
        "value": 84.0,
        "threshold": 0.52,
        "threshold_note": "near-perfect separation",
        "caption": "A smaller bottleneck forces generalisation over memorisation, sharpening the normal-vs-anomaly boundary.",
    },
    {
        "label": "ROC calibrated",
        "value": 89.0,
        "threshold": 0.46,
        "threshold_note": "production target hit",
        "caption": "Threshold tuned on the validation ROC curve pushes recall to the production target at controlled FPR.",
    },
]

if __name__ == "__main__":
    render_metric_story(
        Path(__file__).parent,
        "ch03-autoencoders-needle",
        "Ch.3 — Autoencoder reconstruction error separates anomalies from normals step by step",
        "Recall (%)",
        STAGES,
        better="higher",
        style="threshold",
    )
