"""
gen_ch03.py  —  generate visual assets for Ch.3 · Unsupervised Metrics

Outputs (written relative to this script's parent directory):
  ../img/ch03-unsupervised-metrics-needle.gif          silhouette needle animation
  ../img/ch03-unsupervised-metrics-progress-check.png  metrics-by-k comparison chart

Run:
    python gen_scripts/gen_ch03.py
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
LIGHT   = "#e2e8f0"   # near-white

SEED = 42
rng = np.random.default_rng(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  NEEDLE ANIMATION  —  baseline 0.10  →  k-means 0.52  →  UMAP+k-means 0.57
# ─────────────────────────────────────────────────────────────────────────────

NEEDLE_STAGES = [
    {
        "label": "Random baseline\n(no clustering)",
        "value": 10.0,
        "threshold": 0.10,
        "threshold_note": "silhouette=0.10 — random assignment",
        "caption": (
            "Before any clustering: 440 customers in one undifferentiated group. "
            "Silhouette=0.10 — essentially random. "
            "No actionable segments exist yet."
        ),
    },
    {
        "label": "K-Means k=4\n(raw features, Ch.1)",
        "value": 42.0,
        "threshold": 0.42,
        "threshold_note": "k=4 raw: silhouette=0.42 — below 0.50 target",
        "caption": (
            "K-Means k=4 on 6D log-scaled features. "
            "Silhouette=0.42 — clusters visible but overlap is high. "
            "Below the 0.50 SegmentAI threshold. Further improvement needed."
        ),
    },
    {
        "label": "UMAP 3D + K-Means k=4\n(Ch.2 pipeline)",
        "value": 52.0,
        "threshold": 0.52,
        "threshold_note": "UMAP 3D: silhouette≈0.52 — near target",
        "caption": (
            "UMAP 3D compression reveals tighter cluster geometry. "
            "K-Means re-run on UMAP coordinates: silhouette≈0.52. "
            "Approaching target but not yet formally validated."
        ),
    },
    {
        "label": "UMAP + K-Means + Metrics\n(SegmentAI validated ✅)",
        "value": 57.0,
        "threshold": 0.50,
        "threshold_note": "silhouette=0.57 ✅ > 0.50 target",
        "caption": (
            "Metrics sweep (silhouette, DB, CH) confirms k=4 is optimal. "
            "Silhouette=0.57 exceeds the 0.50 SegmentAI threshold. "
            "DB=0.89, CH=210 — all three metrics agree. "
            "All 5 SegmentAI constraints satisfied. Production-ready."
        ),
    },
]


def make_needle_gif(out_dir: Path) -> None:
    """Produce ch03-unsupervised-metrics-needle.gif via the shared animation renderer."""
    render_metric_story(
        out_dir,
        "ch03-unsupervised-metrics-needle",
        "Ch.3 — Metrics validation lifts SegmentAI silhouette above 0.50",
        "Silhouette Score (×100)",
        NEEDLE_STAGES,
        better="higher",
        style="threshold",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PROGRESS-CHECK  —  metrics comparison by k (silhouette + DB + CH)
# ─────────────────────────────────────────────────────────────────────────────

K_VALUES   = [2,    3,    4,    5,    6,    7,    8]
SILHOUETTE = [0.49, 0.53, 0.57, 0.51, 0.47, 0.44, 0.41]
DB_INDEX   = [1.21, 1.05, 0.89, 0.98, 1.14, 1.22, 1.35]   # inverted for visual clarity
CH_NORM    = [165,  188,  210,  197,  180,  168,  151]      # normalised (raw values)

# For the bar chart we show three panels: silhouette, DB (inverted), CH (normalised)


def make_progress_check(out_dir: Path) -> None:
    """Produce ch03-unsupervised-metrics-progress-check.png (dark background)."""
    x = np.arange(len(K_VALUES))
    k_labels = [f"k={k}" for k in K_VALUES]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), facecolor=BG)
    fig.suptitle(
        "SegmentAI — Metrics by k  (k=4 wins all three)",
        fontsize=13, fontweight="bold", color=LIGHT, y=1.01,
    )

    bar_colours = [
        DANGER if k != 4 else SUCCESS for k in K_VALUES
    ]
    bar_colours[2] = SUCCESS   # k=4 is index 2

    # ── Panel 1: Silhouette ──────────────────────────────────────────────────
    ax0 = axes[0]
    ax0.set_facecolor(BG)
    bars0 = ax0.bar(x, SILHOUETTE, color=bar_colours, width=0.60,
                    edgecolor=LIGHT, linewidth=0.7)
    ax0.axhline(0.50, color=SUCCESS, linestyle="--", linewidth=1.8, alpha=0.85)
    ax0.text(len(K_VALUES) - 0.5, 0.515, "target 0.50",
             color=SUCCESS, fontsize=8, ha="right")
    for bar, val in zip(bars0, SILHOUETTE):
        ax0.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.2f}", ha="center", va="bottom",
                 fontsize=8.5, fontweight="bold", color=LIGHT)
    ax0.set_xticks(x)
    ax0.set_xticklabels(k_labels, color=LIGHT, fontsize=9)
    ax0.set_title("Silhouette ↑ (higher better)", color=LIGHT, fontsize=10, pad=8)
    ax0.set_ylabel("Silhouette score", color=LIGHT, fontsize=9)
    ax0.set_ylim(0.3, 0.68)
    ax0.tick_params(colors=LIGHT)
    for spine in ax0.spines.values():
        spine.set_color(LIGHT)
        spine.set_alpha(0.3)

    # Highlight k=4
    rect0 = mpatches.FancyBboxPatch(
        (x[2] - 0.36, 0.3), 0.72, SILHOUETTE[2] - 0.295,
        boxstyle="round,pad=0.03", linewidth=2, edgecolor=INFO, facecolor="none",
    )
    ax0.add_patch(rect0)

    # ── Panel 2: Davies–Bouldin (lower is better, invert bar direction) ──────
    ax1 = axes[1]
    ax1.set_facecolor(BG)
    # Show DB directly — lower bars = better
    bars1 = ax1.bar(x, DB_INDEX, color=list(reversed(bar_colours)), width=0.60,
                    edgecolor=LIGHT, linewidth=0.7)
    # Reverse colors: k=4 should be SUCCESS
    for i, bar in enumerate(bars1):
        bar.set_color(SUCCESS if K_VALUES[i] == 4 else DANGER)
    ax1.axhline(1.0, color=CAUTION, linestyle="--", linewidth=1.8, alpha=0.85)
    ax1.text(len(K_VALUES) - 0.5, 1.02, "threshold 1.0",
             color=CAUTION, fontsize=8, ha="right")
    for bar, val in zip(bars1, DB_INDEX):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.2f}", ha="center", va="bottom",
                 fontsize=8.5, fontweight="bold", color=LIGHT)
    ax1.set_xticks(x)
    ax1.set_xticklabels(k_labels, color=LIGHT, fontsize=9)
    ax1.set_title("Davies–Bouldin ↓ (lower better)", color=LIGHT, fontsize=10, pad=8)
    ax1.set_ylabel("DB index", color=LIGHT, fontsize=9)
    ax1.set_ylim(0.6, 1.55)
    ax1.tick_params(colors=LIGHT)
    for spine in ax1.spines.values():
        spine.set_color(LIGHT)
        spine.set_alpha(0.3)

    rect1 = mpatches.FancyBboxPatch(
        (x[2] - 0.36, 0.6), 0.72, DB_INDEX[2] - 0.595,
        boxstyle="round,pad=0.03", linewidth=2, edgecolor=INFO, facecolor="none",
    )
    ax1.add_patch(rect1)

    # ── Panel 3: Calinski–Harabasz ───────────────────────────────────────────
    ax2 = axes[2]
    ax2.set_facecolor(BG)
    bars2 = ax2.bar(x, CH_NORM, color=[SUCCESS if k == 4 else DANGER for k in K_VALUES],
                    width=0.60, edgecolor=LIGHT, linewidth=0.7)
    for bar, val in zip(bars2, CH_NORM):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{val}", ha="center", va="bottom",
                 fontsize=8.5, fontweight="bold", color=LIGHT)
    ax2.set_xticks(x)
    ax2.set_xticklabels(k_labels, color=LIGHT, fontsize=9)
    ax2.set_title("Calinski–Harabasz ↑ (higher better)", color=LIGHT, fontsize=10, pad=8)
    ax2.set_ylabel("CH score", color=LIGHT, fontsize=9)
    ax2.set_ylim(120, 240)
    ax2.tick_params(colors=LIGHT)
    for spine in ax2.spines.values():
        spine.set_color(LIGHT)
        spine.set_alpha(0.3)

    rect2 = mpatches.FancyBboxPatch(
        (x[2] - 0.36, 120), 0.72, CH_NORM[2] - 118,
        boxstyle="round,pad=0.03", linewidth=2, edgecolor=INFO, facecolor="none",
    )
    ax2.add_patch(rect2)

    # ── shared annotation ────────────────────────────────────────────────────
    fig.text(0.5, -0.03,
             "k=4 highlighted in green — unanimous winner across all three metrics",
             ha="center", color=SUCCESS, fontsize=10, style="italic")

    plt.tight_layout(pad=1.5)
    out_path = out_dir / "ch03-unsupervised-metrics-progress-check.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  saved → {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "img"
    out_dir.mkdir(exist_ok=True)

    print("Generating ch03 Unsupervised Metrics assets …")
    print("  [1/2] needle animation …")
    make_needle_gif(out_dir)

    print("  [2/2] progress-check chart …")
    make_progress_check(out_dir)

    print("Done. Assets written to:", out_dir)
