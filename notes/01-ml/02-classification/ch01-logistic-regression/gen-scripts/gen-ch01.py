"""Generate ch01 Logistic Regression visual assets.

Outputs
-------
img/ch01-logistic-regression-needle.gif
    Accuracy needle sweeping 0% → 88% (FaceAI constraint #1 progress).

img/ch01-logistic-regression-progress-check.png
    Five-constraint dashboard showing which FaceAI goals Ch.1 addresses.

Usage
-----
    python gen_scripts/gen_ch01.py

Run from the chapter root:
    cd notes/01-ml/02_classification/ch01_logistic_regression
    python gen_scripts/gen_ch01.py
"""
import matplotlib
matplotlib.use("Agg")

import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

# ── reproducibility ───────────────────────────────────────────────
rng = np.random.default_rng(42)

# ── output directory ──────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
IMG_DIR = SCRIPT_DIR.parent / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# ── colour palette (matches authoring-guide canonical colours) ────
BG_DARK    = "#1a1a2e"      # dark background (required by authoring guide)
PRIMARY    = "#1e3a8a"      # dark navy
INFO       = "#1d4ed8"      # medium blue
SUCCESS    = "#15803d"      # green
CAUTION    = "#b45309"      # amber
DANGER     = "#b91c1c"      # red
WHITE      = "#e2e8f0"      # near-white text
GREY_DIM   = "#64748b"      # muted grey for empty segments
NEEDLE_CLR = "#f59e0b"      # amber/gold needle

# ═══════════════════════════════════════════════════════════════════
# 1. ACCURACY NEEDLE GIF  (0% → 88%)
# ═══════════════════════════════════════════════════════════════════

def _needle_angle(pct: float, min_pct: float = 0.0, max_pct: float = 100.0,
                  start_deg: float = 210.0, end_deg: float = -30.0) -> float:
    """Map a percentage to a needle angle (degrees, measured from +x axis)."""
    frac = (pct - min_pct) / (max_pct - min_pct)
    return start_deg + frac * (end_deg - start_deg)


def _build_needle_frame(ax, pct: float, target: float = 88.0) -> None:
    """Draw one frame of the accuracy needle on *ax*."""
    ax.clear()
    ax.set_facecolor(BG_DARK)
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.55, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── colour band arcs ────────────────────────────────────────────
    theta_full = np.linspace(math.radians(210), math.radians(-30), 300)
    # background (empty) arc
    ax.plot(np.cos(theta_full), np.sin(theta_full), color=GREY_DIM,
            lw=18, solid_capstyle="round", alpha=0.35)

    # filled progress arc up to current pct
    n_fill = max(1, int(len(theta_full) * pct / 100.0))
    ax.plot(np.cos(theta_full[:n_fill]), np.sin(theta_full[:n_fill]),
            color=(SUCCESS if pct >= target else INFO),
            lw=18, solid_capstyle="round", alpha=0.85)

    # target marker (dashed line from centre outward)
    t_angle = math.radians(_needle_angle(target))
    ax.plot([0.65 * math.cos(t_angle), 0.95 * math.cos(t_angle)],
            [0.65 * math.sin(t_angle), 0.95 * math.sin(t_angle)],
            color=SUCCESS, lw=2.5, ls="--", alpha=0.7)
    ax.text(1.08 * math.cos(t_angle), 1.08 * math.sin(t_angle),
            f"{target:.0f}%", color=SUCCESS, fontsize=9,
            ha="center", va="center", fontweight="bold")

    # ── needle ──────────────────────────────────────────────────────
    angle_rad = math.radians(_needle_angle(pct))
    nx = 0.82 * math.cos(angle_rad)
    ny = 0.82 * math.sin(angle_rad)
    ax.annotate("", xy=(nx, ny), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>",
                                color=NEEDLE_CLR,
                                lw=3.0,
                                mutation_scale=18))

    # needle pivot
    pivot = plt.Circle((0, 0), 0.06, color=NEEDLE_CLR, zorder=5)
    ax.add_patch(pivot)

    # ── tick labels ─────────────────────────────────────────────────
    for tick_pct in [0, 20, 40, 60, 80, 100]:
        t_rad = math.radians(_needle_angle(tick_pct))
        lx = 1.13 * math.cos(t_rad)
        ly = 1.13 * math.sin(t_rad)
        ax.text(lx, ly, f"{tick_pct}%", color=WHITE, fontsize=8,
                ha="center", va="center", alpha=0.7)

    # ── centre text ─────────────────────────────────────────────────
    ax.text(0, -0.22, f"{pct:.1f}%",
            color=NEEDLE_CLR, fontsize=26, fontweight="bold",
            ha="center", va="center",
            path_effects=[pe.withStroke(linewidth=4, foreground=BG_DARK)])

    ax.text(0, -0.40, "Constraint #1: ACCURACY",
            color=WHITE, fontsize=9, ha="center", va="center", alpha=0.8)

    # ── title ────────────────────────────────────────────────────────
    ax.set_title("FaceAI — Smiling Attribute\nCh.1 Logistic Regression",
                 color=WHITE, fontsize=11, fontweight="bold", pad=4)


# Build frame percentages: ease-in/ease-out ramp to 88%
TARGET_ACCURACY = 88.0
N_FRAMES = 60
t = np.linspace(0, 1, N_FRAMES)
ease = t * t * (3 - 2 * t)          # smoothstep
frame_pcts = ease * TARGET_ACCURACY

fig_needle, ax_needle = plt.subplots(figsize=(4.5, 3.8), facecolor=BG_DARK)


def _update_needle(frame_idx: int):
    _build_needle_frame(ax_needle, frame_pcts[frame_idx], TARGET_ACCURACY)


anim = FuncAnimation(fig_needle, _update_needle, frames=N_FRAMES, interval=40)
needle_path = IMG_DIR / "ch01-logistic-regression-needle.gif"
anim.save(str(needle_path), writer=PillowWriter(fps=25))
plt.close(fig_needle)
print(f"Saved: {needle_path}")


# ═══════════════════════════════════════════════════════════════════
# 2. PROGRESS CHECK DASHBOARD  (5-constraint status panel)
# ═══════════════════════════════════════════════════════════════════

constraints = [
    {
        "id": "#1",
        "name": "ACCURACY",
        "target": ">90% avg\n40 attributes",
        "value": 87.3,
        "max_value": 100,
        "status": "partial",
        "note": "87.3% Smiling\n39 attrs pending",
    },
    {
        "id": "#2",
        "name": "GENERALIZATION",
        "target": "Unseen celebrity\nfaces",
        "value": 0,
        "max_value": 100,
        "status": "locked",
        "note": "Not yet\nvalidated",
    },
    {
        "id": "#3",
        "name": "MULTI-LABEL",
        "target": "40 simultaneous\nattributes",
        "value": 0,
        "max_value": 100,
        "status": "locked",
        "note": "1 of 40\nattributes",
    },
    {
        "id": "#4",
        "name": "INTERPRETABILITY",
        "target": "Feature\nimportance",
        "value": 0,
        "max_value": 100,
        "status": "locked",
        "note": "HOG-level\nonly",
    },
    {
        "id": "#5",
        "name": "PRODUCTION",
        "target": "<200ms\ninference",
        "value": 100,
        "max_value": 100,
        "status": "done",
        "note": "<10ms\nsklearn",
    },
]

STATUS_COLOUR = {
    "done":    SUCCESS,
    "partial": CAUTION,
    "locked":  GREY_DIM,
}
STATUS_LABEL = {
    "done":    "✓ DONE",
    "partial": "⚠ PARTIAL",
    "locked":  "✗ LOCKED",
}

fig_pc, axes_pc = plt.subplots(1, 5, figsize=(14, 5), facecolor=BG_DARK)
fig_pc.suptitle(
    "FaceAI  ·  5-Constraint Dashboard  ·  After Ch.1 Logistic Regression",
    color=WHITE, fontsize=13, fontweight="bold", y=1.01
)

for ax, c in zip(axes_pc, constraints):
    ax.set_facecolor(BG_DARK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    colour = STATUS_COLOUR[c["status"]]

    # outer border rect
    border = mpatches.FancyBboxPatch(
        (0.04, 0.03), 0.92, 0.93,
        boxstyle="round,pad=0.02",
        linewidth=2, edgecolor=colour, facecolor=BG_DARK, alpha=0.9
    )
    ax.add_patch(border)

    # constraint ID badge
    badge = mpatches.FancyBboxPatch(
        (0.30, 0.82), 0.40, 0.12,
        boxstyle="round,pad=0.015",
        linewidth=0, facecolor=colour, alpha=0.9
    )
    ax.add_patch(badge)
    ax.text(0.50, 0.88, c["id"],
            color=WHITE, fontsize=12, fontweight="bold",
            ha="center", va="center")

    # constraint name
    ax.text(0.50, 0.73, c["name"],
            color=WHITE, fontsize=10, fontweight="bold",
            ha="center", va="center")

    # target text
    ax.text(0.50, 0.60, c["target"],
            color=WHITE, fontsize=8, ha="center", va="center",
            alpha=0.75, multialignment="center")

    # progress bar
    bar_bg = mpatches.FancyBboxPatch(
        (0.10, 0.42), 0.80, 0.08,
        boxstyle="round,pad=0.01",
        linewidth=0, facecolor=GREY_DIM, alpha=0.4
    )
    ax.add_patch(bar_bg)
    fill_w = 0.80 * (c["value"] / c["max_value"])
    if fill_w > 0.001:
        bar_fill = mpatches.FancyBboxPatch(
            (0.10, 0.42), fill_w, 0.08,
            boxstyle="round,pad=0.01",
            linewidth=0, facecolor=colour, alpha=0.85
        )
        ax.add_patch(bar_fill)

    # percentage label
    pct_str = f"{c['value']:.0f}%" if c["value"] > 0 else "—"
    ax.text(0.50, 0.35, pct_str,
            color=NEEDLE_CLR if c["value"] > 0 else GREY_DIM,
            fontsize=16, fontweight="bold",
            ha="center", va="center")

    # status badge
    ax.text(0.50, 0.23, STATUS_LABEL[c["status"]],
            color=colour, fontsize=9, fontweight="bold",
            ha="center", va="center")

    # note
    ax.text(0.50, 0.10, c["note"],
            color=WHITE, fontsize=8, ha="center", va="center",
            alpha=0.65, multialignment="center")

plt.tight_layout(pad=0.5)
progress_path = IMG_DIR / "ch01-logistic-regression-progress-check.png"
fig_pc.savefig(str(progress_path), dpi=130, bbox_inches="tight",
               facecolor=BG_DARK)
plt.close(fig_pc)
print(f"Saved: {progress_path}")
