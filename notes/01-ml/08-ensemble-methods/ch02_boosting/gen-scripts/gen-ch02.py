"""
gen_ch02.py  —  generate visual assets for Ch.2 · Boosting: AdaBoost & Gradient Boosting

Outputs (written to ../img/ relative to this script):
  ../img/ch02-boosting-needle.gif         needle animation MAE: 35k → 28k → 25k
  ../img/ch02-boosting-progress-check.png EnsembleAI progress bar chart

Run:
    python notes/01-ml/08_ensemble_methods/ch02_boosting/gen_scripts/gen_ch02.py
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

# ── palette (authoring-guide canonical colours) ───────────────────────────────
BG      = "#1a1a2e"   # dark navy background
PRIMARY = "#1e3a8a"   # deep blue
SUCCESS = "#15803d"   # green
CAUTION = "#b45309"   # amber
DANGER  = "#b91c1c"   # red
INFO    = "#1d4ed8"   # mid blue
LIGHT   = "#e2e8f0"

SEED = 42
rng = np.random.default_rng(SEED)

OUT_DIR = Path(__file__).parent.parent / "img"


# ─────────────────────────────────────────────────────────────────────────────
# 1.  NEEDLE ANIMATION  MAE: single tree 35k → GBM 50 trees 28k → GBM 100 trees 25k
# ─────────────────────────────────────────────────────────────────────────────

NEEDLE_STAGES = [
    {
        "label": "Single tree (MAE=35k)",
        "value": 35.0,
        "curve": [0.28, 0.04, 1.05, 0.38],
        "caption": (
            "Single decision tree (max_depth=5): MAE ≈ 35k. "
            "High bias — misses coastal premiums and regional patterns."
        ),
    },
    {
        "label": "GBM 10 rounds (MAE=30k)",
        "value": 30.0,
        "curve": [0.33, 0.09, 0.98, 0.25],
        "caption": (
            "After 10 boosting rounds each tree corrects the previous ensemble's residuals. "
            "Biggest systematic errors (coastal under-pricing) are absorbed."
        ),
    },
    {
        "label": "GBM 50 rounds (MAE=28k)",
        "value": 28.0,
        "curve": [0.37, 0.14, 0.94, 0.14],
        "caption": (
            "50 shallow trees — MAE ≈ 28k (20%% improvement). "
            "Residuals are smaller and less systematic; hard districts remain."
        ),
    },
    {
        "label": "GBM 100 rounds (MAE=25k)",
        "value": 25.0,
        "curve": [0.41, 0.20, 0.91, 0.04],
        "caption": (
            "100 rounds — MAE ≈ 25k (28.5%% improvement). "
            "Early stopping fires at ~120 rounds. EnsembleAI Constraint #1 exceeded by 5.7×."
        ),
    },
]


def make_needle_gif(out_dir: Path) -> None:
    """Produce ch02-boosting-needle.gif via the shared animation renderer."""
    render_metric_story(
        out_dir,
        "ch02-boosting-needle",
        "Ch.2 — Gradient Boosting: MAE drops from 35k to 25k",
        "MAE ($k)",
        NEEDLE_STAGES,
        better="lower",
        style="regression",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PROGRESS-CHECK BAR CHART — EnsembleAI track
# ─────────────────────────────────────────────────────────────────────────────

MODELS = [
    ("Single Tree\n(Baseline)",  35.0, DANGER),
    ("Random Forest\nCh.1",      30.0, CAUTION),
    ("GBM 50 rounds\nCh.2",      28.0, INFO),
    ("GBM 100 rounds\nCh.2",     25.0, PRIMARY),
    ("XGBoost\nCh.3 (target)",   22.0, SUCCESS),
]


def make_progress_check(out_dir: Path) -> None:
    """Produce ch02-boosting-progress-check.png (dark background, MAE lower=better)."""
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=BG)
    ax.set_facecolor(BG)

    labels  = [m[0] for m in MODELS]
    values  = [m[1] for m in MODELS]
    colours = [m[2] for m in MODELS]
    x = np.arange(len(labels))

    bars = ax.bar(x, values, color=colours, width=0.55, edgecolor=LIGHT, linewidth=0.8)

    # value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{val:.0f}k",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=LIGHT,
        )

    # EnsembleAI target line (5%+ improvement over 35k = below 33.25k)
    target_mae = 33.25
    ax.axhline(target_mae, color=SUCCESS, linestyle="--", linewidth=1.8, alpha=0.85)
    ax.text(
        len(labels) - 0.44, target_mae + 0.5,
        "5%+ improvement threshold (33.25k)",
        color=SUCCESS, fontsize=8.5, ha="right",
    )

    # highlight the Ch.2 100-round result
    ch2_idx = 3
    rect = mpatches.FancyBboxPatch(
        (x[ch2_idx] - 0.32, 0),
        0.64, values[ch2_idx] + 1.5,
        boxstyle="round,pad=0.05",
        linewidth=2, edgecolor=INFO, facecolor="none",
    )
    ax.add_patch(rect)
    ax.text(
        x[ch2_idx], values[ch2_idx] + 2.5,
        "← Ch.2 result",
        ha="center", va="bottom",
        fontsize=9, color=INFO,
    )

    # improvement annotation
    ax.annotate(
        "−28.5%\n(35k → 25k)",
        xy=(x[ch2_idx], values[ch2_idx]),
        xytext=(x[ch2_idx] + 0.9, values[ch2_idx] + 6),
        fontsize=9, color=SUCCESS,
        arrowprops=dict(arrowstyle="->", color=SUCCESS, lw=1.4),
        ha="center",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, color=LIGHT)
    ax.set_ylabel("MAE ($k)  — lower is better", fontsize=11, color=LIGHT)
    ax.set_ylim(0, 42)
    ax.invert_yaxis()   # lower MAE at top = visually "higher" performance
    ax.set_ylim(42, 0)  # re-apply after invert so bars grow downward from axis
    ax.set_title(
        "EnsembleAI Progress — Ch.2: GBM beats single tree by 28.5% on CA Housing",
        fontsize=13, fontweight="bold", color=LIGHT, pad=14,
    )

    ax.tick_params(colors=LIGHT, which="both")
    for spine in ax.spines.values():
        spine.set_edgecolor(LIGHT)
        spine.set_alpha(0.3)
    ax.grid(axis="y", color=LIGHT, alpha=0.12, linewidth=0.7)

    out_path = out_dir / "ch02-boosting-progress-check.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"wrote {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_needle_gif(OUT_DIR)
    make_progress_check(OUT_DIR)
    print("Done — all assets written to", OUT_DIR)
