"""
gen_ch03.py — Generate visual assets for Ch.3 XGBoost & LightGBM.

Outputs (written to ../img/):
  ch03-xgboost-needle.gif   — MAE needle: GB 25k → XGBoost 22k → LightGBM 22k (3x faster)
  ch03-xgboost-needle.png   — Static last frame of the needle animation
  ch03-xgboost-progress-check.png — EnsembleAI constraint scorecard
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
BG       = "#1a1a2e"   # dark background for generated plots
PRIMARY  = "#1e3a8a"   # navy
SUCCESS  = "#15803d"   # green
CAUTION  = "#b45309"   # amber
DANGER   = "#b91c1c"   # red
INFO     = "#1d4ed8"   # blue
LIGHT    = "#f8fafc"

IMG_DIR = Path(__file__).resolve().parent.parent / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)

# ── needle animation stages ─────────────────────────────────────────────────
# MAE in $k — lower is better
# The regression style shows scatter + fit curves; value drives the gauge needle
NEEDLE_STAGES = [
    {
        "label": "sklearn GB (Ch.2): MAE=25k",
        "value": 25.0,
        "curve": [0.38, 0.10, 0.95, 0.18],
        "caption": (
            "Vanilla Gradient Boosting (sklearn): exact split finding, no L1/L2 regularization. "
            "MAE = $25k — strong, but slow and regularization-limited."
        ),
    },
    {
        "label": "XGBoost: MAE=22k",
        "value": 22.0,
        "curve": [0.40, 0.14, 0.90, 0.09],
        "caption": (
            "XGBoost adds 2nd-order Taylor expansion and L1/L2 leaf-weight regularization. "
            "MAE = $22k — 12% improvement over sklearn GB."
        ),
    },
    {
        "label": "LightGBM: MAE=22k (3× faster)",
        "value": 22.0,
        "curve": [0.41, 0.15, 0.89, 0.07],
        "caption": (
            "LightGBM histogram binning achieves identical MAE = $22k "
            "while training 3× faster than sklearn GB via 80× fewer split evaluations."
        ),
    },
]


def _generate_needle():
    render_metric_story(
        IMG_DIR,
        "ch03-xgboost-needle",
        "Ch.3 — XGBoost & LightGBM: MAE Needle ($k, lower=better)",
        "MAE ($k)",
        NEEDLE_STAGES,
        better="lower",
        style="regression",
    )


# ── progress-check scorecard ────────────────────────────────────────────────
CONSTRAINTS = [
    ("#1 IMPROVEMENT",    True,  "XGBoost $22k vs GB $25k\n(12% improvement)"),
    ("#2 DIVERSITY",      True,  "2nd-order + histogram:\ntwo different engines"),
    ("#3 EFFICIENCY",     True,  "LightGBM 3× faster;\ninference <1ms"),
    ("#4 INTERPRETABILITY", False, "Global importance only;\nneed SHAP (Ch.4)"),
    ("#5 ROBUSTNESS",     True,  "L1/L2 + early stopping\nstabilize predictions"),
]


def _generate_progress_check():
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=BG)
    ax.set_facecolor(BG)

    y_pos = list(range(len(CONSTRAINTS) - 1, -1, -1))

    for idx, (label, met, detail) in enumerate(CONSTRAINTS):
        y = y_pos[idx]
        color = SUCCESS if met else DANGER
        symbol = "[+]" if met else "[ ]"

        # background bar
        ax.barh(y, 9.5, left=0.3, height=0.7, color=color, alpha=0.18)

        # constraint label
        ax.text(0.5, y, f"{symbol}  {label}", va="center", ha="left",
                fontsize=12, fontweight="bold", color=LIGHT)

        # detail text
        ax.text(5.5, y, detail, va="center", ha="left",
                fontsize=9, color="#cbd5e1", linespacing=1.4)

        # status badge
        status_text = "MET" if met else "OPEN"
        badge_color = SUCCESS if met else DANGER
        ax.text(9.7, y, status_text, va="center", ha="right",
                fontsize=10, fontweight="bold", color=badge_color)

    ax.set_xlim(0, 10)
    ax.set_ylim(-0.6, len(CONSTRAINTS) - 0.4)
    ax.axis("off")

    ax.set_title(
        "EnsembleAI Scorecard — After Ch.3: XGBoost & LightGBM",
        fontsize=14, fontweight="bold", color=LIGHT, pad=16,
    )

    met_count = sum(1 for _, m, _ in CONSTRAINTS if m)
    ax.text(
        5, -0.55,
        f"{met_count}/{len(CONSTRAINTS)} constraints met  ·  "
        "Remaining: Interpretability (Ch.4 SHAP)",
        ha="center", va="center", fontsize=10, color="#94a3b8",
        style="italic",
    )

    out = IMG_DIR / "ch03-xgboost-progress-check.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    print("Generating Ch.3 XGBoost & LightGBM visual assets …")
    _generate_needle()
    _generate_progress_check()
    print("Done.")
