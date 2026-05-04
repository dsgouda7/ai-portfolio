"""
gen_ch01.py  —  generate visual assets for Ch.1 · Clustering

Outputs (written relative to this script's parent directory):
  ../img/ch01-clustering-needle.gif          silhouette needle animation
  ../img/ch01-clustering-progress-check.png  silhouette-by-k bar chart

Run:
    python gen_scripts/gen_ch01.py
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
BG      = "#1a1a2e"   # dark navy (required)
PRIMARY = "#1e3a8a"   # deep blue
SUCCESS = "#15803d"   # green
CAUTION = "#b45309"   # amber
DANGER  = "#b91c1c"   # red
INFO    = "#1d4ed8"   # mid blue
LIGHT   = "#e2e8f0"

SEED = 42
rng = np.random.default_rng(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  NEEDLE ANIMATION  random 0.10 → k=2 0.35 → k=4 0.52 ✅
# ─────────────────────────────────────────────────────────────────────────────

NEEDLE_STAGES = [
    {
        "label": "Random init\n(no clustering)",
        "value": 10.0,
        "threshold": 0.10,
        "threshold_note": "silhouette=0.10 — random baseline",
        "caption": (
            "Before clustering: all 440 customers in one group. "
            "Silhouette=0.10 — essentially random assignment. "
            "No actionable segments exist yet."
        ),
    },
    {
        "label": "K-Means k=2",
        "value": 35.0,
        "threshold": 0.35,
        "threshold_note": "k=2: silhouette=0.35 — too coarse",
        "caption": (
            "k=2 splits into 'large buyers' vs 'small buyers'. "
            "Silhouette=0.35 — HoReCa and Retail are lumped together. "
            "Not yet actionable for targeted marketing."
        ),
    },
    {
        "label": "K-Means k=4\n(SegmentAI target)",
        "value": 52.0,
        "threshold": 0.50,
        "threshold_note": "k=4: silhouette=0.52 ✅ > 0.50 target",
        "caption": (
            "k=4 reveals HoReCa, Retail, Mixed, and Outlier-bulk segments. "
            "Silhouette=0.52 exceeds the 0.50 SegmentAI threshold. "
            "Four interpretable segments ready for marketing activation."
        ),
    },
]


def make_needle_gif(out_dir: Path) -> None:
    """Produce ch01-clustering-needle.gif via the shared animation renderer."""
    render_metric_story(
        out_dir,
        "ch01-clustering-needle",
        "Ch.1 — Clustering: k selection lifts silhouette above 0.50",
        "Silhouette Score (×100)",
        NEEDLE_STAGES,
        better="higher",
        style="threshold",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PROGRESS-CHECK BAR CHART — silhouette by k
# ─────────────────────────────────────────────────────────────────────────────

K_RESULTS = [
    ("k=1\n(baseline)", 10.0,  DANGER),
    ("k=2",             35.0,  CAUTION),
    ("k=3",             44.0,  CAUTION),
    ("k=4\n(SegmentAI)", 52.0, PRIMARY),
    ("k=5",             47.0,  INFO),
    ("k=6",             43.0,  INFO),
    ("k=8",             38.0,  INFO),
    ("k=10\n(target)",  50.0,  SUCCESS),
]


def make_progress_check(out_dir: Path) -> None:
    """Produce ch01-clustering-progress-check.png (dark background)."""
    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor=BG)
    ax.set_facecolor(BG)

    labels  = [d[0] for d in K_RESULTS]
    values  = [d[1] for d in K_RESULTS]
    colours = [d[2] for d in K_RESULTS]
    x = np.arange(len(labels))

    bars = ax.bar(x, values, color=colours, width=0.60, edgecolor=LIGHT, linewidth=0.8)

    # value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{val/100:.2f}",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold", color=LIGHT,
        )

    # target line at 0.50
    ax.axhline(50, color=SUCCESS, linestyle="--", linewidth=1.8, alpha=0.85)
    ax.text(len(labels) - 0.45, 51.5, "silhouette=0.50 target",
            color=SUCCESS, fontsize=9, ha="right")

    # highlight k=4 (index 3)
    k4_idx = 3
    rect = mpatches.FancyBboxPatch(
        (x[k4_idx] - 0.34, 0), 0.68, values[k4_idx] + 2,
        boxstyle="round,pad=0.05",
        linewidth=2, edgecolor=INFO, facecolor="none",
    )
    ax.add_patch(rect)
    ax.text(
        x[k4_idx], values[k4_idx] + 4.5,
        "← Ch.1 result",
        ha="center", va="bottom", fontsize=9, color=INFO,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, color=LIGHT)
    ax.set_ylabel("Silhouette Score (×100)", fontsize=11, color=LIGHT)
    ax.set_ylim(0, 72)
    ax.set_title(
        "SegmentAI — Silhouette Score vs Number of Clusters (k)",
        color=LIGHT, fontsize=13, fontweight="bold", pad=12,
    )
    ax.tick_params(colors=LIGHT)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

    # legend
    legend_items = [
        mpatches.Patch(color=DANGER,  label="Below baseline"),
        mpatches.Patch(color=CAUTION, label="Improving (below target)"),
        mpatches.Patch(color=PRIMARY, label="Ch.1 result (k=4, 0.52 ✅)"),
        mpatches.Patch(color=INFO,    label="Higher k — diminishing returns"),
        mpatches.Patch(color=SUCCESS, label="0.50 target line"),
    ]
    ax.legend(
        handles=legend_items, loc="upper right",
        facecolor="#1e293b", edgecolor="#334155",
        labelcolor=LIGHT, fontsize=8,
    )

    fig.tight_layout()
    out_path = out_dir / "ch01-clustering-progress-check.png"
    fig.savefig(out_path, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "img"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Ch.1 Clustering assets …")
    print("  [1/2] needle animation …")
    make_needle_gif(out_dir)

    print("  [2/2] progress-check bar chart …")
    make_progress_check(out_dir)

    print("Done.")
