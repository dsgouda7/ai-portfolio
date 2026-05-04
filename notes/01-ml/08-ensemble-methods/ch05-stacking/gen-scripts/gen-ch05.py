"""
gen_ch05.py — Generate visual assets for Ch.5 Stacking & Blending.

Outputs (written to ../img/):
  ch05-stacking-needle.gif        — MAE needle: XGBoost 22k → avg 23k → stacking 20k
  ch05-stacking-needle.png        — Static last frame
  ch05-stacking-progress-check.png — EnsembleAI constraint scorecard
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.animation_renderer import render_metric_story

# ── colour palette (project standard) ──────────────────────────────────────
BG      = "#1a1a2e"   # dark background
PRIMARY = "#1e3a8a"   # navy
SUCCESS = "#15803d"   # green
CAUTION = "#b45309"   # amber
DANGER  = "#b91c1c"   # red
INFO    = "#1d4ed8"   # blue
LIGHT   = "#f8fafc"

rng = np.random.default_rng(42)

IMG_DIR = Path(__file__).resolve().parent.parent / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# ── needle animation stages ─────────────────────────────────────────────────
# MAE in $k — lower is better
# Story: XGBoost alone → simple averaging regresses → meta-learner stacking wins
NEEDLE_STAGES = [
    {
        "label": "XGBoost alone: MAE=22k",
        "value": 22.0,
        "curve": [0.40, 0.14, 0.90, 0.09],
        "caption": (
            "Ch.3 best single model: XGBoost with 2nd-order Taylor expansion "
            "and L1/L2 regularization. MAE = $22k — the previous EnsembleAI champion."
        ),
    },
    {
        "label": "Simple average (LR+RF+XGB): MAE=23k",
        "value": 23.0,
        "curve": [0.38, 0.16, 0.92, 0.11],
        "caption": (
            "Equal-weight averaging of Linear Regression ($42k) + Random Forest ($30k) "
            "+ XGBoost ($22k). The weaker LR model pollutes the signal — MAE regresses "
            "to $23k, worse than XGBoost alone. Equal weighting is not neutral."
        ),
    },
    {
        "label": "Stacking (Ridge meta-learner, 5-fold OOF): MAE=20k",
        "value": 20.0,
        "curve": [0.41, 0.15, 0.89, 0.07],
        "caption": (
            "Ridge meta-learner trained on 5-fold out-of-fold predictions. "
            "Learned weights: w=[0.12, 0.21, 0.65] — heavily trusting XGBoost. "
            "MAE = $20k: 9% better than XGBoost alone, clearing the >5% EnsembleAI constraint."
        ),
    },
]


def _generate_needle() -> None:
    render_metric_story(
        IMG_DIR,
        "ch05-stacking-needle",
        "Ch.5 — Stacking vs Averaging: MAE Needle ($k, lower=better)",
        "MAE ($k)",
        NEEDLE_STAGES,
        better="lower",
        style="regression",
    )


# ── progress-check scorecard ────────────────────────────────────────────────
CONSTRAINTS = [
    ("#1 IMPROVEMENT",      True,  "Stacking $20k vs XGBoost $22k\n(9.1% improvement, clears >5% target)"),
    ("#2 DIVERSITY",        True,  "LR (linear) + RF (bagging)\n+ XGB (boosting): 3 families"),
    ("#3 EFFICIENCY",       False, "3 base models = 3× latency\nneeds pruning in Ch.6"),
    ("#4 INTERPRETABILITY", True,  "SHAP on Ridge meta-learner\nw=[0.12, 0.21, 0.65] readable"),
    ("#5 ROBUSTNESS",       True,  "5-fold OOF reduces split variance\n±0.3k across 5 seeds"),
]


def _generate_progress_check() -> None:
    fig, ax = plt.subplots(figsize=(11, 5.8), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.6, len(CONSTRAINTS) - 0.4)
    ax.axis("off")

    # title
    ax.text(
        5.0, len(CONSTRAINTS) - 0.1,
        "Ch.5 Stacking — EnsembleAI Constraint Scorecard",
        ha="center", va="bottom",
        fontsize=13, fontweight="bold", color=LIGHT,
    )

    # subtitle with MAE result
    ax.text(
        5.0, len(CONSTRAINTS) - 0.42,
        "MAE: XGBoost $22k  →  avg $23k (regressed)  →  stacking $20k  ✓  (−9.1%)",
        ha="center", va="bottom",
        fontsize=9.5, color="#94a3b8",
    )

    y_positions = list(range(len(CONSTRAINTS) - 1, -1, -1))

    for idx, (label, met, detail) in enumerate(CONSTRAINTS):
        y = y_positions[idx]
        color = SUCCESS if met else DANGER
        symbol = "[+]" if met else "[ ]"

        # background bar
        ax.barh(y, 9.4, left=0.3, height=0.72, color=color, alpha=0.15)

        # constraint label
        ax.text(
            0.55, y, f"{symbol}  {label}",
            va="center", ha="left",
            fontsize=11.5, fontweight="bold", color=LIGHT,
        )

        # detail text
        ax.text(
            5.2, y, detail,
            va="center", ha="left",
            fontsize=8.8, color="#cbd5e1", linespacing=1.4,
        )

        # status badge
        status_text = "MET" if met else "OPEN"
        ax.text(
            9.65, y, status_text,
            va="center", ha="right",
            fontsize=10, fontweight="bold", color=color,
        )

    # bottom annotation
    ax.text(
        5.0, -0.52,
        "Next: Ch.6 Production Ensembles — latency benchmarks, model pruning, "
        "A/B testing, ensemble-or-not decision",
        ha="center", va="bottom",
        fontsize=8.5, color="#64748b", style="italic",
    )

    fig.tight_layout(pad=0.6)
    out = IMG_DIR / "ch05-stacking-progress-check.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating Ch.5 Stacking assets...")
    _generate_needle()
    _generate_progress_check()
    print("Done.")
