"""Generate ch05 Hyperparameter Tuning visual assets.

Outputs
-------
img/ch05-hyperparameter-tuning-needle.gif
    Accuracy needle sweeping from ~85% (manual defaults) → 91%
    (avg across 40 CelebA attributes after systematic tuning).

img/ch05-hyperparameter-tuning-progress-check.png
    Five-constraint FaceAI mission-complete dashboard.
    All 5 constraints show ✓ DONE — Classification track complete!

Usage
-----
    python gen_scripts/gen_ch05.py

Run from the chapter root:
    cd notes/01-ml/02_classification/ch05_hyperparameter_tuning
    python gen_scripts/gen_ch05.py
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
# 1. ACCURACY NEEDLE GIF  (85% → 91%  avg across 40 attributes)
# ═══════════════════════════════════════════════════════════════════

# Animation story: four distinct phases matching the chapter arc
# Phase 0 (frames  0-14): manual defaults → 85.0%
# Phase 1 (frames 15-29): grid search   → 87.5%
# Phase 2 (frames 30-44): random search → 89.2%
# Phase 3 (frames 45-74): Bayesian opt  → 91.0%

PHASES = [
    {"label": "Manual defaults",  "value": 85.0, "colour": DANGER},
    {"label": "Grid search",      "value": 87.5, "colour": CAUTION},
    {"label": "Random search",    "value": 89.2, "colour": INFO},
    {"label": "Bayesian opt",     "value": 91.0, "colour": SUCCESS},
]

# Build smooth per-frame percentage values
_PHASE_FRAMES = [15, 15, 15, 30]   # frames spent in each phase
assert len(_PHASE_FRAMES) == len(PHASES)
N_FRAMES = sum(_PHASE_FRAMES)      # 75 total frames


def _smoothstep(t: float) -> float:
    """Ease in/out: maps [0,1] → [0,1] with zero derivatives at endpoints."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


# Pre-compute (pct, label, colour) for each frame
_frame_data = []
prev_val = 0.0
for phase_idx, (phase, n_f) in enumerate(zip(PHASES, _PHASE_FRAMES)):
    start = prev_val
    end = phase["value"]
    for fi in range(n_f):
        t = _smoothstep(fi / max(n_f - 1, 1))
        pct = start + (end - start) * t
        _frame_data.append((pct, phase["label"], phase["colour"]))
    prev_val = end


def _needle_angle(pct: float, min_pct: float = 75.0, max_pct: float = 100.0,
                  start_deg: float = 210.0, end_deg: float = -30.0) -> float:
    """Map a percentage in [min_pct, max_pct] to a needle angle (degrees)."""
    frac = (pct - min_pct) / (max_pct - min_pct)
    frac = max(0.0, min(1.0, frac))
    return start_deg + frac * (end_deg - start_deg)


TARGET_ACCURACY = 91.0


def _build_needle_frame(ax, pct: float, label: str, phase_colour: str) -> None:
    """Draw one frame of the accuracy needle on *ax*."""
    ax.clear()
    ax.set_facecolor(BG_DARK)
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.65, 1.30)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── colour band arcs ────────────────────────────────────────────
    # Arc spans 75% → 100% (the relevant range for this chapter)
    theta_full = np.linspace(math.radians(210), math.radians(-30), 300)
    # background (empty) arc
    ax.plot(np.cos(theta_full), np.sin(theta_full), color=GREY_DIM,
            lw=18, solid_capstyle="round", alpha=0.35)

    # filled progress arc up to current pct
    n_fill = max(1, int(len(theta_full) * (pct - 75.0) / 25.0))
    n_fill = min(n_fill, len(theta_full))
    fill_colour = SUCCESS if pct >= TARGET_ACCURACY else phase_colour
    ax.plot(np.cos(theta_full[:n_fill]), np.sin(theta_full[:n_fill]),
            color=fill_colour,
            lw=18, solid_capstyle="round", alpha=0.85)

    # 90% target marker (the FaceAI threshold)
    t_angle_90 = math.radians(_needle_angle(90.0))
    ax.plot([0.65 * math.cos(t_angle_90), 0.95 * math.cos(t_angle_90)],
            [0.65 * math.sin(t_angle_90), 0.95 * math.sin(t_angle_90)],
            color=SUCCESS, lw=2.5, ls="--", alpha=0.75)
    ax.text(1.10 * math.cos(t_angle_90), 1.10 * math.sin(t_angle_90),
            "90%\ntarget", color=SUCCESS, fontsize=8,
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
    # pivot circle
    pivot = plt.Circle((0, 0), 0.055, color=NEEDLE_CLR, zorder=5)
    ax.add_patch(pivot)

    # ── tick labels (75, 80, 85, 90, 95, 100) ───────────────────────
    for tick_pct in [75, 80, 85, 90, 95, 100]:
        t_rad = math.radians(_needle_angle(tick_pct))
        lx = 1.13 * math.cos(t_rad)
        ly = 1.13 * math.sin(t_rad)
        ax.text(lx, ly, f"{tick_pct}%", color=WHITE, fontsize=8,
                ha="center", va="center", alpha=0.7)

    # ── centre readout ───────────────────────────────────────────────
    ax.text(0, -0.22, f"{pct:.1f}%",
            color=NEEDLE_CLR, fontsize=26, fontweight="bold",
            ha="center", va="center",
            path_effects=[pe.withStroke(linewidth=4, foreground=BG_DARK)])

    ax.text(0, -0.38, "Avg accuracy — 40 CelebA attributes",
            color=WHITE, fontsize=8, ha="center", va="center", alpha=0.75)

    # ── phase label ──────────────────────────────────────────────────
    ax.text(0, -0.52, label,
            color=phase_colour, fontsize=9, ha="center", va="center",
            fontweight="bold",
            path_effects=[pe.withStroke(linewidth=3, foreground=BG_DARK)])

    ax.set_title("FaceAI — Hyperparameter Tuning\nCh.5 · Classification Track",
                 color=WHITE, fontsize=11, fontweight="bold", pad=4)


fig_needle, ax_needle = plt.subplots(figsize=(4.5, 4.0), facecolor=BG_DARK)


def _update_needle(frame_idx: int):
    pct, lbl, col = _frame_data[frame_idx]
    _build_needle_frame(ax_needle, pct, lbl, col)


anim = FuncAnimation(fig_needle, _update_needle, frames=N_FRAMES, interval=50)
needle_path = IMG_DIR / "ch05-hyperparameter-tuning-needle.gif"
anim.save(str(needle_path), writer=PillowWriter(fps=20))
plt.close(fig_needle)
print(f"Saved: {needle_path}")


# ═══════════════════════════════════════════════════════════════════
# 2. PROGRESS CHECK DASHBOARD  (5-constraint mission-complete panel)
# ═══════════════════════════════════════════════════════════════════

constraints = [
    {
        "id": "#1",
        "name": "ACCURACY",
        "target": ">90% avg\n40 attributes",
        "value": 91.2,
        "max_value": 100,
        "status": "done",
        "note": "91.2% avg\nAll 40 ≥ 90% ✓",
    },
    {
        "id": "#2",
        "name": "GENERALIZATION",
        "target": "Unseen celebrity\nfaces",
        "value": 100,
        "max_value": 100,
        "status": "done",
        "note": "Nested CV\nconfirmed ✓",
    },
    {
        "id": "#3",
        "name": "MULTI-LABEL",
        "target": "40 simultaneous\nattributes",
        "value": 100,
        "max_value": 100,
        "status": "done",
        "note": "40 tuned models\nper-attr threshold ✓",
    },
    {
        "id": "#4",
        "name": "INTERPRETABILITY",
        "target": "Feature\nimportance",
        "value": 100,
        "max_value": 100,
        "status": "done",
        "note": "RF importances\nretained ✓",
    },
    {
        "id": "#5",
        "name": "PRODUCTION",
        "target": "<200ms\nreproducible",
        "value": 100,
        "max_value": 100,
        "status": "done",
        "note": "<10ms sklearn\nseed=42 everywhere ✓",
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

fig_pc, axes_pc = plt.subplots(1, 5, figsize=(16, 5.5), facecolor=BG_DARK)
fig_pc.suptitle(
    "FaceAI  ·  5-Constraint Dashboard  ·  Ch.5 Tuning — MISSION COMPLETE",
    color=WHITE, fontsize=13, fontweight="bold", y=1.02
)

for ax, c in zip(axes_pc, constraints):
    ax.set_facecolor(BG_DARK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    colour = STATUS_COLOUR[c["status"]]

    # outer border rect
    border = mpatches.FancyBboxPatch(
        (0.04, 0.03), 0.92, 0.94,
        boxstyle="round,pad=0.02",
        linewidth=2, edgecolor=colour, facecolor=BG_DARK, alpha=0.9
    )
    ax.add_patch(border)

    # constraint ID badge
    badge = mpatches.FancyBboxPatch(
        (0.28, 0.82), 0.44, 0.12,
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

    # progress bar background
    bar_bg = mpatches.FancyBboxPatch(
        (0.08, 0.42), 0.84, 0.08,
        boxstyle="round,pad=0.01",
        linewidth=0, facecolor=GREY_DIM, alpha=0.4
    )
    ax.add_patch(bar_bg)
    fill_w = 0.84 * (c["value"] / c["max_value"])
    if fill_w > 0.001:
        bar_fill = mpatches.FancyBboxPatch(
            (0.08, 0.42), fill_w, 0.08,
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
    ax.text(0.50, 0.24, STATUS_LABEL[c["status"]],
            color=colour, fontsize=9, fontweight="bold",
            ha="center", va="center")

    # note text
    ax.text(0.50, 0.10, c["note"],
            color=WHITE, fontsize=7.5, ha="center", va="center",
            alpha=0.70, multialignment="center")

plt.tight_layout(pad=0.5)
progress_path = IMG_DIR / "ch05-hyperparameter-tuning-progress-check.png"
fig_pc.savefig(str(progress_path), dpi=130, bbox_inches="tight",
               facecolor=BG_DARK)
plt.close(fig_pc)
print(f"Saved: {progress_path}")
