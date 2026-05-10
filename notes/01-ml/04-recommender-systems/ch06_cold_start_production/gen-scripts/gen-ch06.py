"""
gen_ch06.py — generate visual assets for Ch.6 Cold Start & Production Serving

Outputs:
  img/ch06-cold-start-production-needle.gif  — HR@10 needle: cold start solved, 85%+
  img/ch06-cold-start-production-progress-check.png  — FlixAI mission complete bar chart
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.animation import PillowWriter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent            # gen_scripts/
IMG_DIR = HERE.parent / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(42)
DARK_BG = "#1a1a2e"
BLUE = "#1e3a8a"
BLUE_LIGHT = "#1d4ed8"
GREEN = "#15803d"
AMBER = "#b45309"
RED = "#b91c1c"
WHITE = "#f8fafc"
GREY = "#cbd5e1"

# ---------------------------------------------------------------------------
# 1. Needle animation: HR@10 across FlixAI cold-start stages
# ---------------------------------------------------------------------------
STAGES = [
    {"label": "Cold user\n(popularity fallback)", "hr10": 0.42, "color": RED},
    {"label": "Cold user\n(content embed Day-1)", "hr10": 0.61, "color": AMBER},
    {"label": "Warming up\n(blend 30/70, ~20 interact.)", "hr10": 0.74, "color": AMBER},
    {"label": "Full hybrid DCN\n(50+ interactions)", "hr10": 0.87, "color": GREEN},
    {"label": "Cold start solved\nAll 5 constraints ✅", "hr10": 0.88, "color": GREEN},
]

FRAMES_PER_STAGE = 18
HOLD_FRAMES = 6
TOTAL_FRAMES = (len(STAGES) - 1) * FRAMES_PER_STAGE + HOLD_FRAMES


def _ease(t: float) -> float:
    t = max(0.0, min(1.0, float(t)))
    return 0.5 - 0.5 * np.cos(np.pi * t)


def _draw_needle_frame(ax_needle, ax_scatter, frame: int):
    """Draw one animation frame: gauge needle (left) + scatter cloud (right)."""
    # Determine current stage interpolation
    seg = min(frame // FRAMES_PER_STAGE, len(STAGES) - 2)
    local_t = (frame % FRAMES_PER_STAGE) / max(FRAMES_PER_STAGE - 1, 1)
    local_t = _ease(local_t)

    s0, s1 = STAGES[seg], STAGES[seg + 1]
    hr = s0["hr10"] + (s1["hr10"] - s0["hr10"]) * local_t
    stage_color = s0["color"] if local_t < 0.5 else s1["color"]
    label = s0["label"] if local_t < 0.5 else s1["label"]

    # ── Needle gauge (left panel) ──────────────────────────────────────────
    ax_needle.clear()
    ax_needle.set_facecolor(DARK_BG)
    ax_needle.set_xlim(0, 1)
    ax_needle.set_ylim(0, 1)
    ax_needle.set_xticks([])
    ax_needle.set_yticks([])
    for sp in ax_needle.spines.values():
        sp.set_visible(False)

    # Track bar background
    ax_needle.barh(0.5, 1.0, height=0.18, left=0.0, color="#2d2d4e", zorder=1)
    # Filled portion
    ax_needle.barh(0.5, hr, height=0.18, left=0.0, color=stage_color, zorder=2)
    # Needle tip
    ax_needle.axvline(hr, ymin=0.32, ymax=0.68, color=WHITE, lw=2.5, zorder=3)

    ax_needle.text(hr / 2, 0.5, f"HR@10 = {hr:.0%}", ha="center", va="center",
                   fontsize=11, fontweight="bold", color=WHITE, zorder=4)

    ax_needle.text(0.5, 0.82, "FlixAI — HR@10 Progress",
                   ha="center", va="center", fontsize=12, fontweight="bold",
                   color=WHITE)
    ax_needle.text(0.5, 0.22, label, ha="center", va="center",
                   fontsize=9, color=GREY, style="italic")

    # Target line
    ax_needle.axvline(0.85, ymin=0.28, ymax=0.72, color="#facc15",
                      lw=1.5, ls="--", zorder=3)
    ax_needle.text(0.85, 0.75, "85% target", ha="center", fontsize=7.5,
                   color="#facc15")

    # ── Scatter cloud (right panel) ────────────────────────────────────────
    ax_scatter.clear()
    ax_scatter.set_facecolor(DARK_BG)
    ax_scatter.set_title("User–item\nembedding space", color=WHITE, fontsize=9, pad=4)
    for sp in ax_scatter.spines.values():
        sp.set_color("#3d3d5e")

    n_warm = int(300 * hr)
    n_cold = 300 - n_warm
    if n_warm > 0:
        warm_x = RNG.normal(0, 0.6, n_warm)
        warm_y = RNG.normal(0, 0.6, n_warm)
        ax_scatter.scatter(warm_x, warm_y, s=6, alpha=0.5, color=GREEN, zorder=2)
    if n_cold > 0:
        cold_x = RNG.uniform(-2, 2, n_cold)
        cold_y = RNG.uniform(-2, 2, n_cold)
        ax_scatter.scatter(cold_x, cold_y, s=6, alpha=0.3, color=RED, zorder=2)

    ax_scatter.set_xlim(-2.5, 2.5)
    ax_scatter.set_ylim(-2.5, 2.5)
    ax_scatter.set_xticks([])
    ax_scatter.set_yticks([])
    ax_scatter.tick_params(colors=WHITE)

    warm_patch = mpatches.Patch(color=GREEN, label="warm/clustered")
    cold_patch = mpatches.Patch(color=RED, label="cold/scattered")
    ax_scatter.legend(handles=[warm_patch, cold_patch], loc="lower right",
                      fontsize=6.5, facecolor=DARK_BG, labelcolor=WHITE,
                      framealpha=0.7)


def build_needle_gif():
    fig, (ax_needle, ax_scatter) = plt.subplots(
        1, 2, figsize=(8, 3),
        facecolor=DARK_BG,
        gridspec_kw={"width_ratios": [2, 1]}
    )
    fig.subplots_adjust(left=0.04, right=0.97, top=0.88, bottom=0.08, wspace=0.15)

    writer = PillowWriter(fps=12)
    out_path = IMG_DIR / "ch06-cold-start-production-needle.gif"

    with writer.saving(fig, str(out_path), dpi=110):
        for f in range(TOTAL_FRAMES):
            _draw_needle_frame(ax_needle, ax_scatter, f)
            writer.grab_frame()
        # Hold on final frame
        for _ in range(HOLD_FRAMES):
            writer.grab_frame()

    plt.close(fig)
    print(f"  ✓ {out_path.relative_to(HERE.parent)}")


# ---------------------------------------------------------------------------
# 2. Progress-check PNG: mission complete bar chart
# ---------------------------------------------------------------------------
CHAPTERS = ["Ch.1\nPopularity", "Ch.2\nCollab Filter", "Ch.3\nMatrix Factor",
            "Ch.4\nNeural CF", "Ch.5\nHybrid DCN", "Ch.6\nCold Start\n+Production"]
HR10_VALUES = [0.42, 0.68, 0.78, 0.83, 0.87, 0.88]
CONSTRAINT_STATUS = {
    "#1 Accuracy\n>85% HR@10":   [False, False, False, False, True,  True],
    "#2 Cold Start\nNew users":  [False, False, False, False, False, True],
    "#3 Scalability\n<100ms":    [False, False, False, False, False, True],
    "#4 Diversity\nILD>0.4":     [False, False, False, False, True,  True],
    "#5 Explainability\nStake.": [False, False, False, False, True,  True],
}


def build_progress_check():
    fig = plt.figure(figsize=(13, 8), facecolor=DARK_BG)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 1], hspace=0.55)

    # ── Top: HR@10 bar chart ───────────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[0])
    ax_bar.set_facecolor(DARK_BG)

    bar_colors = [RED, AMBER, AMBER, AMBER, GREEN, GREEN]
    bars = ax_bar.bar(CHAPTERS, HR10_VALUES, color=bar_colors,
                      edgecolor=WHITE, linewidth=0.6, zorder=2)

    # Target line
    ax_bar.axhline(0.85, color="#facc15", lw=1.8, ls="--", zorder=3)
    ax_bar.text(5.5, 0.858, "85% target", color="#facc15", fontsize=9,
                ha="right", va="bottom")

    # Value labels
    for bar, val in zip(bars, HR10_VALUES):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, val + 0.008,
                    f"{val:.0%}", ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold", color=WHITE)

    ax_bar.set_ylim(0, 1.0)
    ax_bar.set_ylabel("HR@10", color=WHITE, fontsize=10)
    ax_bar.tick_params(colors=WHITE, labelsize=8.5)
    ax_bar.set_title("FlixAI Track — HR@10 Journey  ✅  Mission Complete",
                     color=WHITE, fontsize=13, fontweight="bold", pad=8)
    for sp in ax_bar.spines.values():
        sp.set_color("#3d3d5e")
    ax_bar.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    # ── Bottom: constraint heatmap ─────────────────────────────────────────
    ax_heat = fig.add_subplot(gs[1])
    ax_heat.set_facecolor(DARK_BG)
    ax_heat.set_title("Constraint Status per Chapter", color=WHITE,
                      fontsize=10, fontweight="bold", pad=6)

    constraints = list(CONSTRAINT_STATUS.keys())
    n_c = len(constraints)
    n_ch = len(CHAPTERS)

    for ci, (cname, statuses) in enumerate(CONSTRAINT_STATUS.items()):
        for chi, ok in enumerate(statuses):
            color = GREEN if ok else RED
            rect = mpatches.FancyBboxPatch(
                (chi - 0.4, ci - 0.35), 0.8, 0.7,
                boxstyle="round,pad=0.05", facecolor=color, edgecolor=DARK_BG,
                linewidth=1.2, zorder=2
            )
            ax_heat.add_patch(rect)
            symbol = "✅" if ok else "❌"
            ax_heat.text(chi, ci, symbol, ha="center", va="center",
                         fontsize=9, zorder=3)

    ax_heat.set_xlim(-0.6, n_ch - 0.4)
    ax_heat.set_ylim(-0.6, n_c - 0.4)
    ax_heat.set_xticks(range(n_ch))
    ax_heat.set_xticklabels(
        ["Ch.1", "Ch.2", "Ch.3", "Ch.4", "Ch.5", "Ch.6 ✅"],
        color=WHITE, fontsize=8.5
    )
    ax_heat.set_yticks(range(n_c))
    ax_heat.set_yticklabels(constraints, color=WHITE, fontsize=8.5)
    for sp in ax_heat.spines.values():
        sp.set_visible(False)
    ax_heat.tick_params(length=0)

    out_path = IMG_DIR / "ch06-cold-start-production-progress-check.png"
    fig.savefig(str(out_path), dpi=130, bbox_inches="tight",
                facecolor=DARK_BG)
    plt.close(fig)
    print(f"  ✓ {out_path.relative_to(HERE.parent)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating Ch.6 assets …")
    build_needle_gif()
    build_progress_check()
    print("Done.")
