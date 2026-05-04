from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.animation_renderer import render_metric_story

# Ensemble arc: best single detector (OCSVM 78%) → ensemble 82% ✅
STAGES = [
    {
        "label": "Best single detector\n(OCSVM)",
        "value": 0.78,
        "threshold": 0.80,
        "threshold_note": "80% FraudShield target",
        "caption": "One-Class SVM reaches 78% recall — the single-detector ceiling. 2pp below the 80% target.",
    },
    {
        "label": "Score averaging\n(IF + AE + OCSVM)",
        "value": 0.80,
        "threshold": 0.80,
        "threshold_note": "80% FraudShield target",
        "caption": "Simple score averaging already reaches 80% recall by combining complementary failure modes.",
    },
    {
        "label": "Weighted ensemble\n(AUC-proportional)",
        "value": 0.81,
        "threshold": 0.80,
        "threshold_note": "80% FraudShield target",
        "caption": "Weighting by individual AUC (IF=0.85, AE=0.90, OCSVM=0.87) squeezes another percentage point.",
    },
    {
        "label": "Stacking\n(logistic regression)",
        "value": 0.82,
        "threshold": 0.80,
        "threshold_note": "80% FraudShield target",
        "caption": "Meta-learner learns optimal combination — 82% recall @ 0.5% FPR. FraudShield Constraint #1 ACHIEVED ✅",
    },
]

if __name__ == "__main__":
    render_metric_story(
        Path(__file__).parent,
        "ch05-ensemble-anomaly-needle",
        "Ch.5 — Ensemble: 78% → 82% recall ✅",
        "Recall @ 0.5% FPR",
        STAGES,
        better="higher",
        style="needle",
    )
