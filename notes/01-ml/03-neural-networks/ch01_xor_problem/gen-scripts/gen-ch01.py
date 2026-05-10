"""Generate ch01 XOR Problem visual assets.

Outputs
-------
img/ch01-xor-problem-needle.gif
    Error-rate needle showing the XOR ceiling: linear model STUCK at 50%
    error through two stages; one hidden layer drops error to 0%.
    Lower is better.

img/ch01-xor-problem-progress-check.png
    UnifiedAI 5-constraint dashboard after Ch.1 XOR diagnostic.
    All five constraints remain LOCKED — this is a diagnostic chapter.

Usage
-----
    python gen_scripts/gen_ch01.py

Run from the chapter root:
    cd notes/01-ml/03_neural_networks/ch01_xor_problem
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
# 1. ERROR NEEDLE GIF  (Error Rate %, lower = better)
#    Three stages:
#      Stage 1 — "Linear model"          → 50% error (STUCK)
#      Stage 2 — "Linear (more training)"→ 50% error (STILL STUCK)
#      Stage 3 — "One hidden layer"      → 0% error  (SOLVED)
# ═══════════════════════════════════════════════════════════════════

def _needle_angle(pct: float, min_pct: float = 0.0, max_pct: float = 100.0,
                  start_deg: float = 210.0, end_deg: float = -30.0) -> float:
    """Map an error percentage to a needle angle.

    0% error   → end_deg   (-30°, right side, SUCCESS)
    100% error → start_deg (210°, left side, DANGER)
    """
    frac = (pct - min_pct) / (max_pct - min_pct)
    return start_deg + frac * (end_deg - start_deg)


# Stage metadata: (label, error_pct, bar_colour, note)
STAGES = [
    ("Linear model",           50.0, CAUTION, "STUCK at 50%\n(no line separates XOR)"),
    ("Linear (more training)", 50.0, CAUTION, "STILL STUCK at 50%\n(structural impossibility)"),
    ("One hidden layer\n(2 ReLU neurons)", 0.0, SUCCESS, "SOLVED: 0% error\n(2 neurons + ReLU)"),
]

# Frame layout:  hold stage 1 → hold stage 2 → sweep stage 3 to 0%
HOLD_FRAMES   = 20   # frames per "stuck" stage
SWEEP_FRAMES  = 40   # frames for the sweep from 50% → 0%
TOTAL_FRAMES  = HOLD_FRAMES + HOLD_FRAMES + SWEEP_FRAMES   # 80 frames


def _build_needle_frame(ax, error_pct: float, stage_idx: int) -> None:
    """Draw one error-needle frame on *ax*."""
    ax.clear()
    ax.set_facecolor(BG_DARK)
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.55, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")

    label, _, bar_colour, note = STAGES[stage_idx]

    # ── arc geometry ────────────────────────────────────────────────
    # Arc runs from 210° (left = 100% error = WORST) to -30° (right = 0% error = BEST)
    theta_full = np.linspace(math.radians(210), math.radians(-30), 300)

    # background (full arc, dim)
    ax.plot(np.cos(theta_full), np.sin(theta_full), color=GREY_DIM,
            lw=18, solid_capstyle="round", alpha=0.35)

    # green "good zone" arc (right side = low error)
    good_end_idx = len(theta_full)  # 0% error end
    good_start_idx = int(len(theta_full) * 0.5)  # 50% mark
    ax.plot(np.cos(theta_full[good_start_idx:]),
            np.sin(theta_full[good_start_idx:]),
            color=SUCCESS, lw=18, solid_capstyle="round", alpha=0.18)

    # filled arc up to current error (from left/danger side)
    n_fill = max(1, int(len(theta_full) * (1.0 - error_pct / 100.0)))
    ax.plot(np.cos(theta_full[len(theta_full) - n_fill:]),
            np.sin(theta_full[len(theta_full) - n_fill:]),
            color=bar_colour, lw=18, solid_capstyle="round", alpha=0.85)

    # ── target marker at 0% error ──────────────────────────────────
    t_angle = math.radians(_needle_angle(0.0))
    ax.plot([0.65 * math.cos(t_angle), 0.95 * math.cos(t_angle)],
            [0.65 * math.sin(t_angle), 0.95 * math.sin(t_angle)],
            color=SUCCESS, lw=2.5, ls="--", alpha=0.75)
    ax.text(1.10 * math.cos(t_angle), 1.10 * math.sin(t_angle),
            "0%\ntarget", color=SUCCESS, fontsize=8,
            ha="center", va="center", fontweight="bold", multialignment="center")

    # ── 50% "stuck" marker ─────────────────────────────────────────
    stuck_angle = math.radians(_needle_angle(50.0))
    ax.plot([0.65 * math.cos(stuck_angle), 0.92 * math.cos(stuck_angle)],
            [0.65 * math.sin(stuck_angle), 0.92 * math.sin(stuck_angle)],
            color=CAUTION, lw=2.0, ls=":", alpha=0.60)
    ax.text(1.08 * math.cos(stuck_angle), 1.08 * math.sin(stuck_angle),
            "50%\nceiling", color=CAUTION, fontsize=8,
            ha="center", va="center", multialignment="center", alpha=0.80)

    # ── needle ──────────────────────────────────────────────────────
    angle_rad = math.radians(_needle_angle(error_pct))
    nx = 0.82 * math.cos(angle_rad)
    ny = 0.82 * math.sin(angle_rad)
    ax.annotate("", xy=(nx, ny), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>",
                                color=NEEDLE_CLR,
                                lw=3.0,
                                mutation_scale=18))
    pivot = plt.Circle((0, 0), 0.06, color=NEEDLE_CLR, zorder=5)
    ax.add_patch(pivot)

    # ── tick labels ─────────────────────────────────────────────────
    for tick_pct in [0, 20, 40, 60, 80, 100]:
        t_rad = math.radians(_needle_angle(tick_pct))
        lx = 1.13 * math.cos(t_rad)
        ly = 1.13 * math.sin(t_rad)
        ax.text(lx, ly, f"{tick_pct}%", color=WHITE, fontsize=8,
                ha="center", va="center", alpha=0.70)

    # ── centre readout ──────────────────────────────────────────────
    ax.text(0, -0.20, f"{error_pct:.0f}%",
            color=NEEDLE_CLR, fontsize=26, fontweight="bold",
            ha="center", va="center",
            path_effects=[pe.withStroke(linewidth=4, foreground=BG_DARK)])

    ax.text(0, -0.37, "Error Rate  (lower = better)",
            color=WHITE, fontsize=8.5, ha="center", va="center", alpha=0.80)

    # ── stage label (top) ───────────────────────────────────────────
    ax.set_title(f"Stage {stage_idx + 1}: {label}",
                 color=WHITE, fontsize=10, fontweight="bold", pad=4)

    # ── diagnostic note (bottom) ────────────────────────────────────
    ax.text(0, -0.49, note,
            color=bar_colour, fontsize=8.5, fontweight="bold",
            ha="center", va="center", multialignment="center",
            path_effects=[pe.withStroke(linewidth=3, foreground=BG_DARK)])


# ── build frame sequence ────────────────────────────────────────────
# 20 frames: stage 0 (linear, 50%)
# 20 frames: stage 1 (linear more training, 50%)
# 40 frames: stage 2 (hidden layer, sweeping 50% → 0%)

t_sweep = np.linspace(0, 1, SWEEP_FRAMES)
ease = t_sweep * t_sweep * (3 - 2 * t_sweep)    # smoothstep ease-in/out
sweep_errors = 50.0 * (1.0 - ease)              # 50% → 0%

all_frames: list[tuple[float, int]] = []
for _ in range(HOLD_FRAMES):
    all_frames.append((50.0, 0))
for _ in range(HOLD_FRAMES):
    all_frames.append((50.0, 1))
for err in sweep_errors:
    all_frames.append((err, 2))

fig_needle, ax_needle = plt.subplots(figsize=(4.5, 4.0), facecolor=BG_DARK)


def _update_needle(frame_idx: int) -> None:
    error_pct, stage_idx = all_frames[frame_idx]
    _build_needle_frame(ax_needle, error_pct, stage_idx)


anim = FuncAnimation(fig_needle, _update_needle, frames=len(all_frames), interval=60)
needle_path = IMG_DIR / "ch01-xor-problem-needle.gif"
anim.save(str(needle_path), writer=PillowWriter(fps=16))
plt.close(fig_needle)
print(f"Saved: {needle_path}")


# ═══════════════════════════════════════════════════════════════════
# 2. PROGRESS CHECK DASHBOARD  (5-constraint status panel)
#    All 5 LOCKED — diagnostic chapter, nothing unlocked yet.
# ═══════════════════════════════════════════════════════════════════

constraints = [
    {
        "id": "#1",
        "name": "ACCURACY",
        "target": "≤$28k MAE\n≥95% acc",
        "value": 0,
        "max_value": 100,
        "status": "locked",
        "note": "Ceiling\ndiagnosed",
    },
    {
        "id": "#2",
        "name": "GENERALIZATION",
        "target": "Unseen districts\n+ identities",
        "value": 0,
        "max_value": 100,
        "status": "locked",
        "note": "No training\nloop yet",
    },
    {
        "id": "#3",
        "name": "MULTI-TASK",
        "target": "Housing +\nFace attrs",
        "value": 0,
        "max_value": 100,
        "status": "locked",
        "note": "No arch\nyet",
    },
    {
        "id": "#4",
        "name": "INTERPRETABILITY",
        "target": "Attention\nweights",
        "value": 0,
        "max_value": 100,
        "status": "locked",
        "note": "No attn\nmechanism",
    },
    {
        "id": "#5",
        "name": "PRODUCTION",
        "target": "<100ms\nmonitoring",
        "value": 0,
        "max_value": 100,
        "status": "locked",
        "note": "No deployed\nmodel yet",
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
    "UnifiedAI  ·  5-Constraint Dashboard  ·  After Ch.1 XOR Diagnostic",
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

    # progress bar (background)
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

    # note text
    ax.text(0.50, 0.10, c["note"],
            color=WHITE, fontsize=8, ha="center", va="center",
            alpha=0.65, multialignment="center")

plt.tight_layout(pad=0.5)
progress_path = IMG_DIR / "ch01-xor-problem-progress-check.png"
fig_pc.savefig(str(progress_path), dpi=130, bbox_inches="tight",
               facecolor=BG_DARK)
plt.close(fig_pc)
print(f"Saved: {progress_path}")
