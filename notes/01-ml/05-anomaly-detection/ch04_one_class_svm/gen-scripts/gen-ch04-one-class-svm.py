"""
gen_ch04.py  —  generate visual assets for Ch.4 · One-Class SVM

Outputs (written relative to this script's parent directory, i.e. gen_scripts/):
  ../img/ch04-one-class-svm-needle.gif          needle animation  75% → 78% recall
  ../img/ch04-one-class-svm-progress-check.png  progress-check bar chart

Run:
    python gen_scripts/gen_ch04.py
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

# ── palette ──────────────────────────────────────────────────────────────────
BG       = "#1a1a2e"   # dark navy background (required)
PRIMARY  = "#1e3a8a"   # deep blue — OCSVM highlight
SUCCESS  = "#15803d"   # green
CAUTION  = "#b45309"   # amber
DANGER   = "#b91c1c"   # red
INFO     = "#1d4ed8"   # mid blue
LIGHT    = "#e2e8f0"

SEED = 42
rng = np.random.default_rng(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  NEEDLE ANIMATION  75% → 78% recall
# ─────────────────────────────────────────────────────────────────────────────

NEEDLE_STAGES = [
    {
        "label": "AE baseline",
        "value": 75.0,
        "threshold": 0.42,
        "threshold_note": "75% recall — previous chapter",
        "caption": (
            "Autoencoder (Ch.3) reaches ~75% recall. "
            "Fraud that reconstructs passably is missed."
        ),
    },
    {
        "label": "OC-SVM fitted (ν=0.05)",
        "value": 76.5,
        "threshold": 0.48,
        "threshold_note": "sphere too loose",
        "caption": (
            "One-Class SVM with ν=0.05 draws a loose sphere — "
            "some fraud slips inside the boundary undetected."
        ),
    },
    {
        "label": "ν tightened (ν=0.01)",
        "value": 77.5,
        "threshold": 0.55,
        "threshold_note": "tighter sphere",
        "caption": (
            "Reducing ν to 0.01 tightens the sphere: "
            "more fraud transactions land outside it."
        ),
    },
    {
        "label": "ROC-calibrated threshold",
        "value": 78.0,
        "threshold": 0.61,
        "threshold_note": "78% recall @ 0.5% FPR",
        "caption": (
            "Threshold selected on validation ROC curve "
            "locks in 78% recall at the 0.5% FPR budget."
        ),
    },
]


def make_needle_gif(out_dir: Path) -> None:
    """Produce ch04-one-class-svm-needle.gif via the shared animation renderer."""
    render_metric_story(
        out_dir,
        "ch04-one-class-svm-needle",
        "Ch.4 — One-Class SVM: boundary tightening lifts recall",
        "Recall (%)",
        NEEDLE_STAGES,
        better="higher",
        style="threshold",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PROGRESS-CHECK BAR CHART
# ─────────────────────────────────────────────────────────────────────────────

DETECTORS = [
    ("Z-score\nCh.1",      45.0, CAUTION),
    ("iForest\nCh.2",      72.0, CAUTION),
    ("Autoencoder\nCh.3",  75.0, CAUTION),
    ("OC-SVM\nCh.4",       78.0, PRIMARY),
    ("Ensemble\nCh.5 (target)", 80.0, SUCCESS),
]


def make_progress_check(out_dir: Path) -> None:
    """Produce ch04-one-class-svm-progress-check.png (dark background)."""
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=BG)
    ax.set_facecolor(BG)

    labels  = [d[0] for d in DETECTORS]
    values  = [d[1] for d in DETECTORS]
    colours = [d[2] for d in DETECTORS]
    x = np.arange(len(labels))

    bars = ax.bar(x, values, color=colours, width=0.55, edgecolor=LIGHT, linewidth=0.8)

    # value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{val:.0f}%",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=LIGHT,
        )

    # target line
    ax.axhline(80, color=SUCCESS, linestyle="--", linewidth=1.8, alpha=0.85)
    ax.text(len(labels) - 0.42, 80.6, "80% target", color=SUCCESS, fontsize=9, ha="right")

    # current chapter highlight box
    ocsvm_idx = 3
    ax.get_children()  # ensure rendered
    rect = mpatches.FancyBboxPatch(
        (x[ocsvm_idx] - 0.32, 0),
        0.64, values[ocsvm_idx] + 2,
        boxstyle="round,pad=0.05",
        linewidth=2, edgecolor=INFO, facecolor="none",
    )
    ax.add_patch(rect)
    ax.text(
        x[ocsvm_idx], values[ocsvm_idx] + 3.8,
        "← Ch.4 result",
        ha="center", va="bottom",
        fontsize=9, color=INFO,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, color=LIGHT)
    ax.set_ylabel("Recall @ 0.5% FPR (%)", fontsize=11, color=LIGHT)
    ax.set_ylim(0, 92)
    ax.set_title(
        "FraudShield Progress — Ch.4: One-Class SVM reaches ~78% recall",
        fontsize=13, fontweight="bold", color=LIGHT, pad=12,
    )

    ax.tick_params(colors=LIGHT, which="both")
    for spine in ax.spines.values():
        spine.set_edgecolor(LIGHT)
        spine.set_alpha(0.3)
    ax.grid(axis="y", color=LIGHT, alpha=0.12, linewidth=0.7)

    out_path = out_dir / "ch04-one-class-svm-progress-check.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"wrote {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Output directory: ../img/ relative to this script
    img_dir = Path(__file__).resolve().parent.parent / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

    print("Generating ch04 One-Class SVM assets …")
    make_needle_gif(img_dir)
    make_progress_check(img_dir)
    print("Done.")
