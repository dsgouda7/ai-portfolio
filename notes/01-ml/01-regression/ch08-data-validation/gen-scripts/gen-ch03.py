"""Generate Ch.3 — Data Validation & Drift Detection animations.

Outputs (saved to ../img/ relative to this script):
    ch03-data-validation-needle.gif   — chapter-level needle animation showing
                                        Portland MAE dropping as validation
                                        pipeline catches and corrects drift
    ch03-data-validation-progress-check.png — all 5 constraints unlocked

Run:
    python gen_ch03.py

Requires:
    matplotlib  pillow  numpy
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).parent
OUT_DIR = HERE.parent / "img"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Add shared animation_renderer to path
ROOT = Path(__file__).resolve().parents[6]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

try:
    from shared.animation_renderer import render_metric_story
    HAS_RENDERER = True
except ImportError:
    HAS_RENDERER = False

# ── Palette (matches canonical colours from authoring-guide.md) ───────────
BG       = "#1a1a2e"        # dark background
PRIMARY  = "#1e3a8a"        # deep blue
SUCCESS  = "#15803d"        # green
CAUTION  = "#b45309"        # amber
DANGER   = "#b91c1c"        # red
INFO     = "#1d4ed8"        # medium blue
LIGHT    = "#f8fafc"        # near-white
GREY     = "#cbd5e1"        # light grey
DARK     = "#1f2937"        # text dark

# ── Stages for the needle animation ───────────────────────────────────────
# Each stage shows Portland MAE improving as the validation pipeline matures.
# value  = Portland MAE in $k (lower is better)
# curve  = [a, b, c, wiggle] coefficients for ŷ = a·x² + b·x + c + wiggle·sin(2.4x)
# caption = one-sentence story for this stage
STAGES = [
    {
        "label": "No validation (174k MAE)",
        "value": 174.0,
        "curve": [0.20, 0.60, 0.45, 0.18],
        "caption": "Portland deployed without validation — 37% income shift undetected, model extrapolates wildly.",
    },
    {
        "label": "Ch.1: Outlier removal (148k MAE)",
        "value": 148.0,
        "curve": [0.24, 0.44, 0.60, 0.10],
        "caption": "IQR outlier removal and KNN imputation corrects training data quality.",
    },
    {
        "label": "Ch.2: Class rebalancing (128k MAE)",
        "value": 128.0,
        "curve": [0.30, 0.30, 0.72, 0.05],
        "caption": "SMOTE + class-weighted loss fixes imbalance; MAE drops to 128k.",
    },
    {
        "label": "Schema + GE validation (118k MAE)",
        "value": 118.0,
        "curve": [0.34, 0.20, 0.80, 0.02],
        "caption": "Great Expectations catches the MedInc mean violation immediately on Portland data.",
    },
    {
        "label": "KS + PSI drift detection (105k MAE)",
        "value": 105.0,
        "curve": [0.38, 0.13, 0.86, 0.01],
        "caption": "PSI = 0.34 >= 0.25 blocks deployment and triggers the retrain pipeline.",
    },
    {
        "label": "Drift-corrected retrain (89k MAE)",
        "value": 89.0,
        "curve": [0.42, 0.08, 0.91, 0.0],
        "caption": "Retrain on combined CA + PDX data: Portland MAE = 89k. Target < 95k met.",
    },
]


# ══════════════════════════════════════════════════════════════════════════
# 1.  Needle animation
# ══════════════════════════════════════════════════════════════════════════

def gen_needle() -> None:
    """Generate the chapter needle animation using the shared renderer."""
    if HAS_RENDERER:
        render_metric_story(
            OUT_DIR,
            "ch03-data-validation-needle",
            "Ch.3 — Data Validation & Drift Detection",
            "Portland MAE ($k)",
            STAGES,
            better="lower",
            style="regression",
        )
        print("  ch03-data-validation-needle.gif")
    else:
        # Fallback: simple static animation without the shared renderer
        _gen_needle_fallback()


def _gen_needle_fallback() -> None:
    """Minimal fallback needle animation when shared renderer is unavailable."""
    rng = np.random.default_rng(42)
    x = np.linspace(0, 5, 80)
    y_true = 0.4 * x + 1.2
    y_noisy = y_true + rng.normal(0, 0.3, size=x.shape)

    values = [s["value"] for s in STAGES]
    min_v, max_v = min(values), max(values)
    n_frames = (len(STAGES) - 1) * 10 + 8

    fig, (ax_main, ax_gauge) = plt.subplots(1, 2, figsize=(12, 5),
                                            facecolor="white")

    def animate(frame):
        seg = min(frame // 10, len(STAGES) - 2)
        t = (frame % 10) / 9.0
        s0, s1 = STAGES[seg], STAGES[seg + 1]
        val = s0["value"] + (s1["value"] - s0["value"]) * t

        ax_main.clear()
        ax_main.scatter(x, y_noisy, s=12, alpha=0.4, color="#93c5fd")
        ax_main.plot(x, y_true, color=GREY, linestyle="--", linewidth=2)
        slope = s0["curve"][0] + (s1["curve"][0] - s0["curve"][0]) * t
        intercept = s0["curve"][2] + (s1["curve"][2] - s0["curve"][2]) * t
        ax_main.plot(x, slope * x + intercept, color="#2563eb", linewidth=3)
        ax_main.set_title(f"Ch.3 — Data Validation\n{s1['label']}",
                          fontsize=11, fontweight="bold", color=DARK)
        ax_main.set_xlabel("MedInc")
        ax_main.set_ylabel("MedHouseVal")
        ax_main.grid(alpha=0.2)

        ax_gauge.clear()
        ratio = (max_v - val) / (max_v - min_v)
        ax_gauge.set_xlim(0, 1)
        ax_gauge.set_ylim(0, 1)
        ax_gauge.set_xticks([])
        ax_gauge.set_yticks([])
        for spine in ax_gauge.spines.values():
            spine.set_visible(False)
        ax_gauge.hlines(0.45, 0.05, 0.95, color=GREY, linewidth=8)
        ax_gauge.hlines(0.45, 0.05, 0.05 + 0.9 * ratio, color=SUCCESS, linewidth=8)
        ax_gauge.arrow(0.05 + 0.9 * ratio, 0.85, 0, -0.28,
                       width=0.015, head_width=0.06, head_length=0.08,
                       color=DANGER, length_includes_head=True)
        ax_gauge.text(0.5, 0.72, f"Portland MAE\n${val:,.0f}k",
                      ha="center", va="center", fontsize=13, fontweight="bold",
                      color=INFO,
                      bbox=dict(boxstyle="round,pad=0.35", fc=LIGHT, ec=GREY))
        target_ratio = (max_v - 95.0) / (max_v - min_v)
        ax_gauge.axvline(0.05 + 0.9 * target_ratio, color=CAUTION,
                         linestyle="--", linewidth=1.5)
        ax_gauge.text(0.05 + 0.9 * target_ratio, 0.25, "target\n<95k",
                      ha="center", va="top", fontsize=8, color=CAUTION)

    ani = animation.FuncAnimation(fig, animate, frames=n_frames,
                                  interval=260, blit=False, repeat=True)
    gif_path = OUT_DIR / "ch03-data-validation-needle.gif"
    ani.save(gif_path, writer="pillow", fps=4, dpi=100)
    plt.close(fig)
    print(f"  {gif_path.name}")


# ══════════════════════════════════════════════════════════════════════════
# 2.  Progress check image — all 5 constraints unlocked
# ══════════════════════════════════════════════════════════════════════════

CONSTRAINTS = [
    ("#1", "ACCURACY",        "Portland MAE = 89k",         "< 95k MAE", True),
    ("#2", "GENERALIZATION",  "CA\u2192Portland gated",     "KS + PSI drift gate", True),
    ("#3", "DATA QUALITY",    "GE suite automated",         "No garbage in", True),
    ("#4", "AUDITABILITY",    "Audit trail per batch",      "Traceable decisions", True),
    ("#5", "PRODUCTION-READY","Validation + alerting",      "Automated pipeline", True),
]

MAE_PROGRESSION = [
    ("Start (raw data)",        174.0),
    ("Ch.1 — EDA",              148.0),
    ("Ch.2 — Balance",          128.0),
    ("Ch.3 — Schema gate",      118.0),
    ("Ch.3 — PSI + retrain",     89.0),
]
TARGET_MAE = 95.0


def gen_progress_check() -> None:
    """Generate the progress check PNG: dark-themed, all 5 constraints unlocked."""
    rng = np.random.default_rng(42)
    fig = plt.figure(figsize=(14, 9), facecolor=BG)

    # Layout: left = constraint table, right = MAE bar chart
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.0],
                          left=0.03, right=0.97,
                          bottom=0.06, top=0.92, wspace=0.12)
    ax_table = fig.add_subplot(gs[0])
    ax_bar   = fig.add_subplot(gs[1])

    ax_table.set_facecolor(BG)
    ax_bar.set_facecolor(BG)

    # ── Title ──────────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.96,
        "Ch.3 — Data Validation & Drift Detection: Progress Check",
        ha="center", va="top", fontsize=16, fontweight="bold", color=LIGHT,
    )
    fig.text(
        0.5, 0.905,
        "RealtyML Grand Challenge — All 5 Constraints Tracked",
        ha="center", va="top", fontsize=11, color=GREY,
    )

    # ── Constraint table ───────────────────────────────────────────────────
    ax_table.set_xlim(0, 1)
    ax_table.set_ylim(0, 1)
    ax_table.axis("off")

    # Header row
    headers = ["#", "Constraint", "Status", "Achievement"]
    col_x   = [0.02, 0.10, 0.42, 0.67]
    col_w   = [0.08, 0.32, 0.25, 0.33]
    header_y = 0.92

    for hx, hw, hdr in zip(col_x, col_w, headers):
        rect = FancyBboxPatch(
            (hx, header_y - 0.05), hw - 0.01, 0.08,
            boxstyle="round,pad=0.01",
            linewidth=0, facecolor=PRIMARY,
        )
        ax_table.add_patch(rect)
        ax_table.text(hx + (hw - 0.01) / 2, header_y,
                      hdr, ha="center", va="center",
                      fontsize=9.5, fontweight="bold", color=LIGHT)

    # Data rows
    row_h = 0.13
    for row_i, (num, name, status, achievement, unlocked) in enumerate(CONSTRAINTS):
        y_top = header_y - 0.10 - row_i * row_h
        row_color = SUCCESS if unlocked else CAUTION
        row_bg    = "#0f2417" if unlocked else "#2a1a00"

        # Row background
        bg_rect = FancyBboxPatch(
            (col_x[0], y_top - 0.095), sum(col_w) - 0.01, row_h - 0.015,
            boxstyle="round,pad=0.01",
            linewidth=1, edgecolor=row_color, facecolor=row_bg, alpha=0.85,
        )
        ax_table.add_patch(bg_rect)

        check = "\u2705" if unlocked else "\u274c"
        row_data = [num, f"{check} {name}", status, achievement]
        for col_xi, hw, cell in zip(col_x, col_w, row_data):
            ax_table.text(
                col_xi + 0.005, y_top - 0.028,
                cell,
                ha="left", va="center",
                fontsize=8.8, color=LIGHT,
                wrap=True,
            )

    # Sub-title for table
    ax_table.text(0.5, 0.02, "Data Fundamentals Track — 3 chapters complete",
                  ha="center", va="bottom", fontsize=8.5, color=GREY,
                  style="italic")

    # ── MAE progression bar chart ──────────────────────────────────────────
    ax_bar.set_facecolor(BG)
    labels = [s[0] for s in MAE_PROGRESSION]
    maes   = [s[1] for s in MAE_PROGRESSION]

    bar_colors = []
    for m in maes:
        if m <= TARGET_MAE:
            bar_colors.append(SUCCESS)
        elif m <= 130:
            bar_colors.append(CAUTION)
        else:
            bar_colors.append(DANGER)

    bars = ax_bar.barh(range(len(maes)), maes,
                       color=bar_colors, edgecolor=BG,
                       height=0.6, alpha=0.90)

    # Target line
    ax_bar.axvline(TARGET_MAE, color=CAUTION, linestyle="--",
                   linewidth=2, label=f"Target: <{TARGET_MAE:.0f}k")

    # Value labels
    for bar, mae in zip(bars, maes):
        ax_bar.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                    f"${mae:.0f}k",
                    va="center", ha="left", fontsize=9.5,
                    color=LIGHT, fontweight="bold")

    ax_bar.set_yticks(range(len(labels)))
    ax_bar.set_yticklabels(labels, fontsize=9, color=LIGHT)
    ax_bar.set_xlabel("Portland MAE ($k)", fontsize=10, color=GREY)
    ax_bar.set_title("MAE Progression\nAcross Data Fundamentals",
                     fontsize=11, fontweight="bold", color=LIGHT, pad=8)
    ax_bar.set_xlim(0, 200)
    ax_bar.tick_params(colors=GREY)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    for s in ["bottom", "left"]:
        ax_bar.spines[s].set_color(GREY)
    ax_bar.xaxis.label.set_color(GREY)

    # Legend
    target_patch = mpatches.Patch(color=CAUTION, linestyle="--",
                                  label=f"Target: <{TARGET_MAE:.0f}k", fill=False)
    ax_bar.legend(handles=[target_patch], loc="lower right",
                  facecolor=BG, edgecolor=GREY, labelcolor=LIGHT, fontsize=8)

    # Annotation for final value
    final_y = len(maes) - 1
    final_mae = maes[-1]
    ax_bar.annotate(
        f"TARGET MET \u2713",
        xy=(final_mae, final_y),
        xytext=(final_mae + 15, final_y + 0.3),
        fontsize=9, color=SUCCESS, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=SUCCESS, lw=1.5),
    )

    # ── Save ───────────────────────────────────────────────────────────────
    png_path = OUT_DIR / "ch03-data-validation-progress-check.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  {png_path.name}")


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating Ch.3 — Data Validation & Drift Detection visuals ...")
    gen_needle()
    gen_progress_check()
    print("Done. Outputs written to:", OUT_DIR)
