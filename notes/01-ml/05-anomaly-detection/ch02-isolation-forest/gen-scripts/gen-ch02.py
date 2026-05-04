"""
gen_ch02.py — Generate visual assets for Ch.2 Isolation Forest.

Produces:
  img/ch02-isolation-forest-needle.gif   Recall needle animation: 45% → 65%
  img/ch02-isolation-forest-progress-check.png  FraudShield constraint progress bar

Run from the chapter root:
    python gen_scripts/gen_ch02.py
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

# ---------------------------------------------------------------------------
# 1. Needle animation (recall: 45% → 65%)
# ---------------------------------------------------------------------------

from shared.animation_renderer import render_metric_story

STAGES = [
    {
        "label": "Z-score only",
        "value": 45.0,
        "threshold": 0.86,
        "threshold_note": "many frauds missed",
        "caption": (
            "Statistical thresholds catch only obvious extremes. "
            "Multivariate fraud slips through every per-feature filter."
        ),
    },
    {
        "label": "IF (score > 0.7)",
        "value": 55.0,
        "threshold": 0.74,
        "threshold_note": "high threshold, safe but low recall",
        "caption": (
            "Isolation Forest scores short-path transactions. "
            "Conservative threshold keeps FPR low but leaves structured fraud behind."
        ),
    },
    {
        "label": "IF (score > 0.6)",
        "value": 65.0,
        "threshold": 0.63,
        "threshold_note": "ROC-calibrated threshold",
        "caption": (
            "Threshold tuned from ROC curve at 0.5% FPR. "
            "Path-length scoring lifts recall to 65% — 20 points above statistical baseline."
        ),
    },
]

CHAPTER_DIR = Path(__file__).resolve().parent.parent


def _make_progress_check(chapter_dir: Path) -> None:
    """
    Render a FraudShield constraint progress bar chart and save to
    img/ch02-isolation-forest-progress-check.png.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    SEED = 42
    rng = np.random.default_rng(SEED)
    _ = rng  # seed held for determinism; not used numerically here

    DARK_BG = "#1a1a2e"
    PRIMARY = "#1e3a8a"
    SUCCESS = "#15803d"
    CAUTION = "#b45309"
    DANGER = "#b91c1c"
    INFO = "#1d4ed8"
    TEXT = "#e2e8f0"

    constraints = [
        ("#1 RECALL ≥ 80%",      65, 80, CAUTION, "65% @ 0.5% FPR"),
        ("#2 GENERALIZATION",    50, 100, CAUTION, "Partial (novel fraud undetected)"),
        ("#3 MULTI-SIGNAL",       0, 100, DANGER,  "Blocked (need AE + density)"),
        ("#4 SPEED < 10ms",     100, 100, SUCCESS, "Achieved: O(T·log ψ) ≪ 10ms"),
        ("#5 NO DIST. ASSUMPTION", 100, 100, SUCCESS, "Achieved: IF is distribution-free"),
    ]

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    bar_height = 0.55

    for i, (label, current, target, color, note) in enumerate(constraints):
        # Background bar (target = 100%)
        ax.barh(i, target, height=bar_height, color="#2a2a4a", zorder=1)
        # Progress bar
        ax.barh(i, current, height=bar_height, color=color, zorder=2, alpha=0.9)
        # Target line
        if target < 100:
            ax.axvline(target, color=TEXT, linewidth=1.2, linestyle="--", alpha=0.5, zorder=3)
        # Labels
        ax.text(-2, i, label, va="center", ha="right", color=TEXT,
                fontsize=9, fontweight="bold")
        ax.text(current + 1.5, i, f"{current}% — {note}", va="center",
                ha="left", color=TEXT, fontsize=8, alpha=0.85)

    ax.set_xlim(-35, 130)
    ax.set_ylim(-0.6, len(constraints) - 0.4)
    ax.set_xlabel("% Achieved", color=TEXT, fontsize=10)
    ax.set_title(
        "FraudShield — Ch.2 Isolation Forest: Constraint Progress",
        color=TEXT, fontsize=12, fontweight="bold", pad=12,
    )
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.spines[:].set_color("#3a3a5a")
    ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 65, 80, 100])
    ax.xaxis.label.set_color(TEXT)
    for tick in ax.get_xticklabels():
        tick.set_color(TEXT)

    # Legend
    patches = [
        mpatches.Patch(color=SUCCESS, label="✅ Achieved"),
        mpatches.Patch(color=CAUTION, label="⚡ Partial"),
        mpatches.Patch(color=DANGER, label="❌ Blocked"),
    ]
    ax.legend(
        handles=patches, loc="lower right",
        facecolor=DARK_BG, edgecolor="#3a3a5a",
        labelcolor=TEXT, fontsize=9,
    )

    # Recall callout annotation
    ax.annotate(
        "IF: 45% → 65%\n+20pp vs Ch.1",
        xy=(65, 0),
        xytext=(72, 1.2),
        arrowprops=dict(arrowstyle="->", color=CAUTION, lw=1.5),
        color=CAUTION, fontsize=8.5, fontweight="bold",
    )

    out_path = chapter_dir / "img" / "ch02-isolation-forest-progress-check.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    # --- Needle GIF ---
    render_metric_story(
        CHAPTER_DIR,
        "ch02-isolation-forest-needle",
        "Ch.2 — Isolation Forest: recall from 45% → 65% via path-length scoring",
        "Recall (%)",
        STAGES,
        better="higher",
        style="threshold",
    )

    # --- Progress-check PNG ---
    _make_progress_check(CHAPTER_DIR)
