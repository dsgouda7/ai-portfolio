"""Generate ch02 Classical Classifiers visual assets.

Outputs
-------
img/ch02-classical-classifiers-needle.gif
    Accuracy needle sweeping 88% (Ch.1 result) → 91% (RF on Young).
    Shows incremental improvement: logistic gave 88%, RF breaks 90%.

img/ch02-classical-classifiers-progress-check.png
    Five-constraint FaceAI dashboard showing status after Ch.2.

Usage
-----
    cd notes/01-ml/02_classification/ch02_classical_classifiers
    python gen_scripts/gen_ch02.py
"""
import math
import matplotlib
matplotlib.use("Agg")

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

# ── colour palette (authoring-guide canonical colours) ───────────
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
# 1. ACCURACY NEEDLE GIF  (88% → 91%)
#    This needle starts at the Ch.1 result (88%) and sweeps to the
#    Ch.2 Random Forest result (91%) — showing incremental progress.
# ═══════════════════════════════════════════════════════════════════

def _needle_angle(pct: float,
                  min_pct: float = 0.0,
                  max_pct: float = 100.0,
                  start_deg: float = 210.0,
                  end_deg: float = -30.0) -> float:
    """Map a percentage to a needle angle (degrees from +x axis)."""
    frac = (pct - min_pct) / (max_pct - min_pct)
    return start_deg + frac * (end_deg - start_deg)


def _build_needle_frame(ax,
                        pct: float,
                        start_pct: float = 88.0,
                        target_pct: float = 91.0,
                        constraint_pct: float = 90.0) -> None:
    """Draw one frame of the accuracy needle on *ax*.

    Parameters
    ----------
    pct : float
        Current needle position (current chapter's accuracy).
    start_pct : float
        Previous chapter result (shown as a fixed reference marker).
    target_pct : float
        This chapter's achieved accuracy.
    constraint_pct : float
        The FaceAI accuracy constraint threshold (90%).
    """
    ax.clear()
    ax.set_facecolor(BG_DARK)
    ax.set_xlim(-1.30, 1.30)
    ax.set_ylim(-0.60, 1.30)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── background arc ───────────────────────────────────────────
    theta_full = np.linspace(math.radians(210), math.radians(-30), 300)
    ax.plot(np.cos(theta_full), np.sin(theta_full),
            color=GREY_DIM, lw=18, solid_capstyle="round", alpha=0.30)

    # ── filled progress arc (grey band for 0→88 from ch.1) ───────
    n_prev = max(1, int(len(theta_full) * start_pct / 100.0))
    ax.plot(np.cos(theta_full[:n_prev]), np.sin(theta_full[:n_prev]),
            color=GREY_DIM, lw=18, solid_capstyle="round", alpha=0.55)

    # ── filled progress arc (blue/green band for 88→current) ─────
    n_curr = max(1, int(len(theta_full) * pct / 100.0))
    if n_curr > n_prev:
        colour = SUCCESS if pct >= constraint_pct else INFO
        ax.plot(np.cos(theta_full[n_prev:n_curr]),
                np.sin(theta_full[n_prev:n_curr]),
                color=colour, lw=18, solid_capstyle="round", alpha=0.92)

    # ── constraint marker at 90% (dashed) ────────────────────────
    c_angle = math.radians(_needle_angle(constraint_pct))
    ax.plot([0.62 * math.cos(c_angle), 0.92 * math.cos(c_angle)],
            [0.62 * math.sin(c_angle), 0.92 * math.sin(c_angle)],
            color=CAUTION, lw=2.5, ls="--", alpha=0.85)
    ax.text(1.09 * math.cos(c_angle), 1.09 * math.sin(c_angle),
            f"{constraint_pct:.0f}%\ntarget",
            color=CAUTION, fontsize=8, ha="center", va="center",
            fontweight="bold")

    # ── ch.1 baseline marker at 88% (solid thin line) ────────────
    s_angle = math.radians(_needle_angle(start_pct))
    ax.plot([0.68 * math.cos(s_angle), 0.88 * math.cos(s_angle)],
            [0.68 * math.sin(s_angle), 0.88 * math.sin(s_angle)],
            color=WHITE, lw=1.8, alpha=0.55)
    ax.text(1.07 * math.cos(s_angle), 1.07 * math.sin(s_angle),
            f"Ch.1\n{start_pct:.0f}%",
            color=WHITE, fontsize=7.5, ha="center", va="center", alpha=0.65)

    # ── needle ────────────────────────────────────────────────────
    angle_rad = math.radians(_needle_angle(pct))
    nx = 0.82 * math.cos(angle_rad)
    ny = 0.82 * math.sin(angle_rad)
    ax.annotate("", xy=(nx, ny), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>",
                                color=NEEDLE_CLR,
                                lw=3.0,
                                mutation_scale=18))
    pivot = plt.Circle((0, 0), 0.06, color=NEEDLE_CLR, zorder=5)
    ax.add_patch(pivot)

    # ── tick labels ───────────────────────────────────────────────
    for tick_pct in [0, 20, 40, 60, 80, 100]:
        t_rad = math.radians(_needle_angle(tick_pct))
        ax.text(1.15 * math.cos(t_rad), 1.15 * math.sin(t_rad),
                f"{tick_pct}%", color=WHITE, fontsize=8,
                ha="center", va="center", alpha=0.65)

    # ── centre readout ────────────────────────────────────────────
    ax.text(0, -0.22, f"{pct:.1f}%",
            color=NEEDLE_CLR, fontsize=26, fontweight="bold",
            ha="center", va="center",
            path_effects=[pe.withStroke(linewidth=4, foreground=BG_DARK)])

    status_txt = "✓ TARGET MET!" if pct >= constraint_pct else "Constraint #1: ACCURACY"
    status_col = SUCCESS if pct >= constraint_pct else WHITE
    ax.text(0, -0.40, status_txt,
            color=status_col, fontsize=9, ha="center", va="center",
            fontweight="bold" if pct >= constraint_pct else "normal",
            alpha=0.9)

    ax.set_title(
        "FaceAI — Young Attribute\nCh.2 Random Forest",
        color=WHITE, fontsize=11, fontweight="bold", pad=4
    )


# Build frame percentages: ease-in / ease-out from 88% → 91%
START_ACCURACY  = 88.0
TARGET_ACCURACY = 91.0
N_FRAMES = 70

# Phase 1 (frames 0–9): hold at 88% to show starting point
# Phase 2 (frames 10–60): sweep 88% → 91% with smoothstep
# Phase 3 (frames 61–69): hold at 91%
hold_frames  = 10
sweep_frames = 50
tail_frames  = N_FRAMES - hold_frames - sweep_frames  # 10

t_sweep = np.linspace(0, 1, sweep_frames)
ease    = t_sweep * t_sweep * (3 - 2 * t_sweep)      # smoothstep
sweep_pcts = START_ACCURACY + ease * (TARGET_ACCURACY - START_ACCURACY)

frame_pcts = np.concatenate([
    np.full(hold_frames, START_ACCURACY),
    sweep_pcts,
    np.full(tail_frames, TARGET_ACCURACY),
])

fig_needle, ax_needle = plt.subplots(figsize=(4.8, 4.0), facecolor=BG_DARK)


def _update_needle(frame_idx: int) -> None:
    _build_needle_frame(ax_needle, frame_pcts[frame_idx],
                        start_pct=START_ACCURACY,
                        target_pct=TARGET_ACCURACY,
                        constraint_pct=90.0)


anim = FuncAnimation(fig_needle, _update_needle, frames=N_FRAMES, interval=40)
needle_path = IMG_DIR / "ch02-classical-classifiers-needle.gif"
anim.save(str(needle_path), writer=PillowWriter(fps=25))
plt.close(fig_needle)
print(f"Saved: {needle_path}")


# ═══════════════════════════════════════════════════════════════════
# 2. PROGRESS CHECK DASHBOARD  (5-constraint FaceAI status)
# ═══════════════════════════════════════════════════════════════════

constraints = [
    {
        "id": "#1",
        "name": "ACCURACY",
        "target": ">90% avg\n40 attributes",
        "status": "partial",
        "note": "91% Young OK\n(2 of 40 attrs)",
        "bar_value": 91.0,
        "bar_max": 100.0,
    },
    {
        "id": "#2",
        "name": "GENERALI-\nZATION",
        "target": "Unseen celebrity\nfaces",
        "status": "locked",
        "note": "Not yet\nvalidated",
        "bar_value": 0.0,
        "bar_max": 100.0,
    },
    {
        "id": "#3",
        "name": "MULTI-\nLABEL",
        "target": "40 simultaneous\nattributes",
        "status": "locked",
        "note": "2 of 40\nattributes",
        "bar_value": 5.0,
        "bar_max": 100.0,
    },
    {
        "id": "#4",
        "name": "INTERPRET-\nABILITY",
        "target": "Explain each\nprediction",
        "status": "partial",
        "note": "Tree rules\n(partial)",
        "bar_value": 40.0,
        "bar_max": 100.0,
    },
    {
        "id": "#5",
        "name": "PRODUCTION",
        "target": "<200ms\ninference",
        "status": "done",
        "note": "<5ms RF\nsklearn",
        "bar_value": 100.0,
        "bar_max": 100.0,
    },
]

STATUS_COLOUR = {
    "done":    SUCCESS,
    "partial": CAUTION,
    "locked":  GREY_DIM,
}
STATUS_LABEL = {
    "done":    "[DONE]",
    "partial": "[PARTIAL]",
    "locked":  "[LOCKED]",
}

fig_pc, axes_pc = plt.subplots(1, 5, figsize=(14, 5), facecolor=BG_DARK)
fig_pc.suptitle(
    "FaceAI  ·  5-Constraint Dashboard  ·  After Ch.2 Classical Classifiers",
    color=WHITE, fontsize=13, fontweight="bold", y=1.02,
)

for ax, c in zip(axes_pc, constraints):
    ax.set_facecolor(BG_DARK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    col = STATUS_COLOUR[c["status"]]

    # ── outer border rect ────────────────────────────────────────
    border = mpatches.FancyBboxPatch(
        (0.05, 0.04), 0.90, 0.90,
        boxstyle="round,pad=0.02",
        linewidth=2, edgecolor=col, facecolor=BG_DARK,
    )
    ax.add_patch(border)

    # ── constraint id badge ──────────────────────────────────────
    badge = mpatches.FancyBboxPatch(
        (0.30, 0.78), 0.40, 0.16,
        boxstyle="round,pad=0.02",
        linewidth=0, facecolor=col, alpha=0.85,
    )
    ax.add_patch(badge)
    ax.text(0.50, 0.86, c["id"],
            color=WHITE, fontsize=14, fontweight="bold",
            ha="center", va="center")

    # ── constraint name ──────────────────────────────────────────
    ax.text(0.50, 0.70, c["name"],
            color=WHITE, fontsize=10, fontweight="bold",
            ha="center", va="center",
            multialignment="center")

    # ── target text ───────────────────────────────────────────────
    ax.text(0.50, 0.54, c["target"],
            color=WHITE, fontsize=8, ha="center", va="center",
            alpha=0.75, multialignment="center")

    # ── progress bar ─────────────────────────────────────────────
    bar_bg = mpatches.FancyBboxPatch(
        (0.10, 0.36), 0.80, 0.08,
        boxstyle="round,pad=0.01",
        linewidth=0, facecolor=GREY_DIM, alpha=0.40,
    )
    ax.add_patch(bar_bg)
    fill_w = 0.80 * c["bar_value"] / c["bar_max"]
    if fill_w > 0:
        bar_fill = mpatches.FancyBboxPatch(
            (0.10, 0.36), fill_w, 0.08,
            boxstyle="round,pad=0.01",
            linewidth=0, facecolor=col, alpha=0.85,
        )
        ax.add_patch(bar_fill)

    # ── status note ───────────────────────────────────────────────
    ax.text(0.50, 0.24, c["note"],
            color=col, fontsize=8.5, fontweight="bold",
            ha="center", va="center",
            multialignment="center",
            path_effects=[pe.withStroke(linewidth=2, foreground=BG_DARK)])

    # ── status label ─────────────────────────────────────────────
    ax.text(0.50, 0.10, STATUS_LABEL[c["status"]],
            color=col, fontsize=9, fontweight="bold",
            ha="center", va="center", alpha=0.90)

fig_pc.tight_layout(pad=1.2)
progress_path = IMG_DIR / "ch02-classical-classifiers-progress-check.png"
fig_pc.savefig(str(progress_path), dpi=150, bbox_inches="tight",
               facecolor=BG_DARK, edgecolor="none")
plt.close(fig_pc)
print(f"Saved: {progress_path}")

print("Done — all ch02 assets generated.")
