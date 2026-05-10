"""Generate assets for Ch.3 — Backprop & Optimisers.

Outputs (relative to this script's parent directory):
  ../img/ch03-backprop-optimisers-needle.gif   — MAE needle animation
  ../img/ch03-backprop-optimisers-progress-check.png  — constraint status dashboard

Style: dark background #1a1a2e, seed=42.
MAE progression: $70k → $55k → $40k → $32k (baseline → NN arch → Adam → future target).
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ── Colour palette ────────────────────────────────────────────────────────────
BG      = "#1a1a2e"
CARD    = "#16213e"
BLUE    = "#2563eb"
LBLUE   = "#60a5fa"
GREEN   = "#16a34a"
LGREEN  = "#4ade80"
AMBER   = "#d97706"
RED     = "#dc2626"
PURPLE  = "#7c3aed"
GREY    = "#64748b"
LIGHT   = "#e2e8f0"
WHITE   = "#f8fafc"

# ── Reproducibility ───────────────────────────────────────────────────────────
rng = np.random.default_rng(42)

# ── Output paths ─────────────────────────────────────────────────────────────
HERE    = Path(__file__).resolve().parent
IMG_DIR = HERE.parent / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ease(t: float) -> float:
    t = max(0.0, min(1.0, float(t)))
    return 0.5 - 0.5 * math.cos(math.pi * t)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


# ─────────────────────────────────────────────────────────────────────────────
# Needle GIF
# ─────────────────────────────────────────────────────────────────────────────

# MAE values in thousands of dollars
MAE_STAGES = [
    {"label": "Baseline\n(Linear Reg.)",  "mae": 70.0, "color": RED,    "chapter": "Ch.1 Regression"},
    {"label": "NN Architecture\n(Ch.2)",   "mae": 55.0, "color": AMBER,  "chapter": "Ch.2 NN Arch"},
    {"label": "Adam Optimizer\n(Ch.3)",    "mae": 40.0, "color": GREEN,  "chapter": "Ch.3 Backprop"},
    {"label": "Future Target\n(Ch.4+)",    "mae": 32.0, "color": PURPLE, "chapter": "Ch.4+ Reg."},
]

MAE_MIN  = 25.0
MAE_MAX  = 80.0
TARGET   = 40.0   # Constraint #1 threshold

FRAMES_PER_STAGE = 14
N_FRAMES = (len(MAE_STAGES) - 1) * FRAMES_PER_STAGE + 10


def _needle_angle(mae: float) -> float:
    """Map MAE value to arc angle (180° = worst, 0° = best)."""
    ratio = (mae - MAE_MIN) / (MAE_MAX - MAE_MIN)   # 0 = best, 1 = worst
    ratio = max(0.0, min(1.0, ratio))
    return math.pi * ratio  # π = left/high MAE, 0 = right/low MAE


def _draw_needle_frame(ax, mae: float, stage_label: str, stage_color: str, frame_stage: int):
    ax.clear()
    ax.set_facecolor(BG)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.35, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Arc track ────────────────────────────────────────────────────────────
    theta = np.linspace(math.pi, 0, 300)
    ax.plot(np.cos(theta), np.sin(theta), color=GREY, lw=12, alpha=0.3, solid_capstyle="round")

    # ── Coloured arc (progress) ───────────────────────────────────────────────
    target_angle = _needle_angle(TARGET)
    current_angle = _needle_angle(mae)
    # from π (worst) down to current_angle
    if current_angle < math.pi:
        theta_progress = np.linspace(math.pi, current_angle, 200)
        # colour: red until target, green after
        split = _needle_angle(TARGET)
        if current_angle >= split:
            ax.plot(np.cos(theta_progress), np.sin(theta_progress),
                    color=AMBER, lw=12, alpha=0.85, solid_capstyle="round")
        else:
            theta_red   = np.linspace(math.pi, split, 150)
            theta_green = np.linspace(split, current_angle, 150)
            ax.plot(np.cos(theta_red),   np.sin(theta_red),
                    color=AMBER, lw=12, alpha=0.85, solid_capstyle="round")
            ax.plot(np.cos(theta_green), np.sin(theta_green),
                    color=GREEN, lw=12, alpha=0.95, solid_capstyle="round")

    # ── Target marker ────────────────────────────────────────────────────────
    tx = math.cos(target_angle)
    ty = math.sin(target_angle)
    ax.plot(tx, ty, "o", ms=10, color=GREEN, zorder=5)
    ax.text(tx * 1.17, ty * 1.17 + 0.04, "<$40k\nTarget",
            ha="center", va="bottom", fontsize=8, color=GREEN, fontweight="bold")

    # ── Tick marks ───────────────────────────────────────────────────────────
    tick_values = [80, 70, 60, 50, 40, 30, 25]
    for tv in tick_values:
        ta = _needle_angle(float(tv))
        inner = 0.82
        outer = 0.98
        ax.plot([inner * math.cos(ta), outer * math.cos(ta)],
                [inner * math.sin(ta), outer * math.sin(ta)],
                color=GREY, lw=1.2)
        lx, ly = 1.11 * math.cos(ta), 1.11 * math.sin(ta)
        ax.text(lx, ly, f"${tv}k", ha="center", va="center",
                fontsize=6.5, color=GREY)

    # ── Needle ───────────────────────────────────────────────────────────────
    ang = _needle_angle(mae)
    nx = 0.78 * math.cos(ang)
    ny = 0.78 * math.sin(ang)
    ax.annotate("", xy=(nx, ny), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=WHITE,
                                lw=2.5, mutation_scale=18))
    ax.plot(0, 0, "o", ms=10, color=WHITE, zorder=6)

    # ── Central MAE readout ───────────────────────────────────────────────────
    txt_color = GREEN if mae <= TARGET else AMBER
    ax.text(0, -0.10, f"MAE\n${mae:,.0f}k",
            ha="center", va="top", fontsize=15, fontweight="bold",
            color=txt_color)

    # ── Stage label ───────────────────────────────────────────────────────────
    ax.text(0, -0.30, stage_label, ha="center", va="top",
            fontsize=9, color=stage_color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", fc=CARD, ec=stage_color, alpha=0.9))

    # ── Title ────────────────────────────────────────────────────────────────
    ax.set_title("Ch.3 — Backprop + Optimisers\nMAE Reduction (UnifiedAI)", 
                 fontsize=11, fontweight="bold", color=LIGHT, pad=10)


def _draw_bar_panel(ax, current_mae: float, seg_idx: int):
    ax.clear()
    ax.set_facecolor(CARD)
    for spine in ax.spines.values():
        spine.set_visible(False)

    labels = [s["chapter"] for s in MAE_STAGES]
    values = [s["mae"] for s in MAE_STAGES]
    colors = [GREY] * len(MAE_STAGES)
    for i in range(min(seg_idx + 2, len(MAE_STAGES))):
        colors[i] = MAE_STAGES[i]["color"]

    bars = ax.barh(range(len(labels)), values, color=colors, alpha=0.85,
                   height=0.6, edgecolor=BG, linewidth=0.8)
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + 0.8, bar.get_y() + bar.get_height() / 2,
                f"${val:.0f}k", va="center", fontsize=8, color=LIGHT)

    # Target line
    ax.axvline(TARGET, color=GREEN, lw=2, linestyle="--", alpha=0.8)
    ax.text(TARGET + 0.5, len(labels) - 0.2, "Target\n<$40k",
            fontsize=7, color=GREEN, va="top")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8, color=LIGHT)
    ax.set_xlabel("MAE ($k)", fontsize=8, color=GREY)
    ax.tick_params(colors=GREY)
    ax.set_facecolor(CARD)
    ax.set_title("↓ lower is better", fontsize=8, color=GREY, loc="right")
    ax.set_xlim(0, 85)
    ax.grid(axis="x", color=GREY, alpha=0.15)


def build_needle_gif():
    fig = plt.figure(figsize=(10, 5.5), facecolor=BG)
    ax_needle = fig.add_axes([0.02, 0.08, 0.52, 0.88], facecolor=BG)
    ax_bar    = fig.add_axes([0.56, 0.12, 0.42, 0.76], facecolor=CARD)

    def animate(frame):
        total_segs = len(MAE_STAGES) - 1
        seg    = min(frame // FRAMES_PER_STAGE, total_segs - 1)
        local  = (frame % FRAMES_PER_STAGE) / max(FRAMES_PER_STAGE - 1, 1)
        local  = _ease(local)
        mae    = _lerp(MAE_STAGES[seg]["mae"], MAE_STAGES[seg + 1]["mae"], local)
        color  = MAE_STAGES[seg]["color"] if local < 0.5 else MAE_STAGES[seg + 1]["color"]
        # For the last few hold frames, show final stage
        if frame >= total_segs * FRAMES_PER_STAGE:
            mae   = MAE_STAGES[-1]["mae"]
            color = MAE_STAGES[-1]["color"]
            seg   = total_segs - 1
        label  = MAE_STAGES[seg + (1 if local >= 0.5 else 0)]["label"]
        _draw_needle_frame(ax_needle, mae, label, color, seg)
        _draw_bar_panel(ax_bar, mae, seg)
        return []

    ani = animation.FuncAnimation(fig, animate, frames=N_FRAMES,
                                   interval=280, blit=False, repeat=True)
    gif_path = IMG_DIR / "ch03-backprop-optimisers-needle.gif"
    ani.save(gif_path, writer="pillow", fps=4, dpi=100)
    plt.close(fig)
    print(f"wrote {gif_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Progress-check PNG
# ─────────────────────────────────────────────────────────────────────────────

CONSTRAINTS = [
    {"id": "#1", "name": "ACCURACY",        "target": "<$40k MAE",     "status": "[ACHIEVED]",  "color": GREEN,  "detail": "$40k MAE — Adam unlocked it!"},
    {"id": "#2", "name": "GENERALIZATION",  "target": "Unseen districts","status": "[BLOCKED]",  "color": RED,    "detail": "Train R2=0.88, Test R2=0.64 — overfitting"},
    {"id": "#3", "name": "MULTI-TASK",      "target": "Value + Segment","status": "[PARTIAL]",   "color": AMBER,  "detail": "Regression + binary; no multi-class yet"},
    {"id": "#4", "name": "INTERPRETABILITY","target": "Explainable",    "status": "[PARTIAL]",   "color": AMBER,  "detail": "Black-box neural network"},
    {"id": "#5", "name": "PRODUCTION",      "target": "Scale + Monitor","status": "[BLOCKED]",   "color": RED,    "detail": "Research notebook; no deploy infra"},
]

def build_progress_check():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=BG,
                              gridspec_kw={"width_ratios": [1.6, 1.0]})
    fig.suptitle("Ch.3 — Backprop & Optimisers: Constraint Progress Dashboard",
                 fontsize=13, fontweight="bold", color=LIGHT, y=0.97)

    # ── Left: constraint cards ────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(CONSTRAINTS) + 0.5)

    for i, c in enumerate(CONSTRAINTS):
        y = len(CONSTRAINTS) - i - 0.5
        # Card background
        rect = mpatches.FancyBboxPatch((0.01, y - 0.38), 0.98, 0.72,
                                        boxstyle="round,pad=0.03",
                                        fc=CARD, ec=c["color"], lw=1.5)
        ax.add_patch(rect)
        # ID chip
        chip = mpatches.FancyBboxPatch((0.02, y - 0.26), 0.07, 0.48,
                                        boxstyle="round,pad=0.02",
                                        fc=c["color"], ec="none", alpha=0.85)
        ax.add_patch(chip)
        ax.text(0.055, y + 0.05, c["id"], ha="center", va="center",
                fontsize=9, fontweight="bold", color=WHITE)
        # Name + target
        ax.text(0.12, y + 0.12, c["name"], ha="left", va="center",
                fontsize=9.5, fontweight="bold", color=LIGHT)
        ax.text(0.12, y - 0.14, f"Target: {c['target']}", ha="left", va="center",
                fontsize=7.5, color=GREY)
        # Status badge
        ax.text(0.75, y + 0.12, c["status"], ha="left", va="center",
                fontsize=9, fontweight="bold", color=c["color"])
        ax.text(0.12, y - 0.28, c["detail"], ha="left", va="center",
                fontsize=7, color=GREY, style="italic")

    ax.set_title("UnifiedAI — Constraint Tracker (after Ch.3)", fontsize=10,
                 color=GREY, loc="left", pad=6)

    # ── Right: MAE funnel ─────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(CARD)
    for spine in ax2.spines.values():
        spine.set_color(GREY)
        spine.set_linewidth(0.5)
    ax2.tick_params(colors=GREY)

    chapters = ["Baseline\n(Linear)", "NN Arch\n(Ch.2)", "Adam\n(Ch.3)", "Future\n(Ch.4+)"]
    maes     = [70, 55, 40, 32]
    colors   = [RED, AMBER, GREEN, PURPLE]

    bars = ax2.bar(chapters, maes, color=colors, alpha=0.85,
                   edgecolor=BG, linewidth=1.2, width=0.6)
    for bar, val in zip(bars, maes):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.8,
                 f"${val}k", ha="center", va="bottom",
                 fontsize=10, fontweight="bold", color=LIGHT)

    ax2.axhline(40, color=GREEN, lw=2, linestyle="--", alpha=0.9, zorder=3)
    ax2.text(3.35, 40.8, "Constraint #1\n<$40k target", fontsize=7.5,
             color=GREEN, ha="right")

    ax2.set_ylabel("MAE ($k)  ↓ lower is better", fontsize=8.5, color=GREY)
    ax2.set_ylim(0, 82)
    ax2.set_facecolor(CARD)
    ax2.grid(axis="y", color=GREY, alpha=0.12)
    ax2.set_title("MAE progression", fontsize=10, color=LIGHT, pad=6)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    png_path = IMG_DIR / "ch03-backprop-optimisers-progress-check.png"
    fig.savefig(png_path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"wrote {png_path}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    build_needle_gif()
    build_progress_check()
