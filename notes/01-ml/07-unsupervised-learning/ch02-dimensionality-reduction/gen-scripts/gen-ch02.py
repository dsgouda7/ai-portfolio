"""
gen_ch02.py  —  generate visual assets for Ch.2 · Dimensionality Reduction

Outputs (written relative to this script's parent directory):
  ../img/ch02-dimensionality-reduction-needle.gif          silhouette needle animation
  ../img/ch02-dimensionality-reduction-progress-check.png  method comparison bar chart

Run:
    python gen_scripts/gen_ch02.py
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

# ── palette ───────────────────────────────────────────────────────────────────
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
# 1.  NEEDLE ANIMATION  raw 6D 0.52 → PCA 2D 0.52 → PCA 3D 0.54 → UMAP 3D 0.57 ✅
# ─────────────────────────────────────────────────────────────────────────────

NEEDLE_STAGES = [
    {
        "label": "Raw 6D space\n(Ch.1 baseline)",
        "value": 52.0,
        "threshold": 0.52,
        "threshold_note": "silhouette=0.52 — k=4 in raw 6D",
        "caption": (
            "Ch.1 result: K-Means(k=4) in raw 6-feature space. "
            "Silhouette=0.52 already above the 0.50 target. "
            "But correlated features (Grocery+Detergents r=0.93) add distance noise. "
            "And we still cannot visualise 6D clusters."
        ),
    },
    {
        "label": "PCA 2D\n(85% variance)",
        "value": 51.0,
        "threshold": 0.51,
        "threshold_note": "PCA 2D: silhouette=0.51 — slight loss",
        "caption": (
            "PCA 2D captures 85% of variance in 2 components. "
            "PC1='total spend', PC2='HoReCa vs Retail'. "
            "Stakeholders can now see 4 segments in a scatter plot. "
            "Silhouette drops slightly to 0.51 — 15% variance lost."
        ),
    },
    {
        "label": "PCA 3D\n(93% variance)",
        "value": 54.0,
        "threshold": 0.54,
        "threshold_note": "PCA 3D: silhouette=0.54 — noise removed",
        "caption": (
            "PCA 3D retains 93% of variance including PC3 (Milk vs Frozen). "
            "Removing the 4th–6th components strips correlated noise. "
            "K-Means(k=4) on PCA 3D → silhouette=0.54. "
            "Linear structure captured; non-linear boundaries still blurred."
        ),
    },
    {
        "label": "UMAP 3D\n(SegmentAI best)",
        "value": 57.0,
        "threshold": 0.50,
        "threshold_note": "UMAP 3D: silhouette=0.57 ✅ — best result",
        "caption": (
            "UMAP 3D preserves non-linear cluster manifolds. "
            "K-Means(k=4) on UMAP 3D → silhouette=0.57. "
            "Consistent across 10 random seeds. "
            "UMAP transform() saved for new customer embedding in production."
        ),
    },
]


def make_needle_gif(out_dir: Path) -> None:
    """Produce ch02-dimensionality-reduction-needle.gif via the shared animation renderer."""
    render_metric_story(
        out_dir,
        "ch02-dimensionality-reduction-needle",
        "Ch.2 — Dimensionality Reduction: PCA→UMAP lifts silhouette to 0.57",
        "Silhouette Score (×100)",
        NEEDLE_STAGES,
        better="higher",
        style="threshold",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PROGRESS-CHECK BAR CHART — silhouette by method
# ─────────────────────────────────────────────────────────────────────────────

METHOD_RESULTS = [
    ("Raw 6D\n(Ch.1 baseline)",   52.0,  CAUTION),
    ("PCA 2D\n(85% EVR)",         51.0,  INFO),
    ("PCA 3D\n(93% EVR)",         54.0,  INFO),
    ("t-SNE 2D\n(perplexity=30)", 43.0,  DANGER),
    ("UMAP 3D\n(SegmentAI ✅)",   57.0,  SUCCESS),
]


def make_progress_check(out_dir: Path) -> None:
    """Produce ch02-dimensionality-reduction-progress-check.png (dark background)."""
    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor=BG)
    ax.set_facecolor(BG)

    labels  = [d[0] for d in METHOD_RESULTS]
    values  = [d[1] for d in METHOD_RESULTS]
    colours = [d[2] for d in METHOD_RESULTS]
    x = np.arange(len(labels))

    bars = ax.bar(x, values, color=colours, width=0.60, edgecolor=LIGHT, linewidth=0.8)

    # value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val / 100:.2f}",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold", color=LIGHT,
        )

    # target line at 0.50
    ax.axhline(50, color=SUCCESS, linestyle="--", linewidth=1.8, alpha=0.85)
    ax.text(
        len(labels) - 0.45, 51.5,
        "silhouette=0.50 target",
        color=SUCCESS, fontsize=9, ha="right",
    )

    # highlight UMAP 3D (index 4)
    umap_idx = 4
    rect = mpatches.FancyBboxPatch(
        (x[umap_idx] - 0.34, 0), 0.68, values[umap_idx] + 2,
        boxstyle="round,pad=0.05",
        linewidth=2, edgecolor=SUCCESS, facecolor="none",
    )
    ax.add_patch(rect)
    ax.text(
        x[umap_idx], values[umap_idx] + 4.5,
        "← Ch.2 best",
        ha="center", va="bottom", fontsize=9, color=SUCCESS,
    )

    # note on t-SNE
    tsne_idx = 3
    ax.text(
        x[tsne_idx], values[tsne_idx] + 4.5,
        "⚠ distances\ndistorted",
        ha="center", va="bottom", fontsize=8, color=DANGER,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, color=LIGHT)
    ax.set_ylabel("Silhouette Score (×100)", fontsize=11, color=LIGHT)
    ax.set_ylim(0, 72)
    ax.set_title(
        "SegmentAI — Silhouette Score by Dimensionality Reduction Method",
        color=LIGHT, fontsize=13, fontweight="bold", pad=12,
    )
    ax.tick_params(colors=LIGHT)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

    # legend
    legend_items = [
        mpatches.Patch(color=CAUTION, label="Raw 6D baseline (0.52)"),
        mpatches.Patch(color=INFO,    label="PCA (linear, interpretable)"),
        mpatches.Patch(color=DANGER,  label="t-SNE (⚠ distances distorted)"),
        mpatches.Patch(color=SUCCESS, label="UMAP 3D — best result (0.57 ✅)"),
    ]
    ax.legend(
        handles=legend_items, loc="upper left",
        facecolor="#1e293b", edgecolor="#334155",
        labelcolor=LIGHT, fontsize=8,
    )

    fig.tight_layout()
    out_path = out_dir / "ch02-dimensionality-reduction-progress-check.png"
    fig.savefig(out_path, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "img"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Ch.2 Dimensionality Reduction assets …")
    print("  [1/2] needle animation …")
    make_needle_gif(out_dir)

    print("  [2/2] progress-check bar chart …")
    make_progress_check(out_dir)

    print("Done.")
