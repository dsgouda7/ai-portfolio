"""
gen_ch04.py — Generate visual assets for Ch.4 SHAP Interpretability.

Outputs
-------
img/ch04-shap-needle.gif          Needle animation: black box → explained → ranked
img/ch04-shap-progress-check.png  Progress-check bar chart for §11
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────
SEED = 42
RNG = np.random.default_rng(SEED)

BG        = "#1a1a2e"   # dark background
BLUE      = "#1e3a8a"   # primary blue
GREEN     = "#15803d"   # success green
AMBER     = "#b45309"   # caution amber
RED       = "#b91c1c"   # danger red
LIGHT_BLU = "#1d4ed8"   # info blue
GREY      = "#64748b"
LIGHT     = "#e2e8f0"

IMG_DIR = Path(__file__).resolve().parents[1] / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 · Needle animation
# Three stages:
#   Stage 1 — Black Box:        trust score 30 (no explanation)
#   Stage 2 — Explained:        trust score 72 (per-prediction SHAP)
#   Stage 3 — Feature ranked:   trust score 91 (global importance visible)
# ─────────────────────────────────────────────────────────────────────────────

STAGES = [
    {
        "label":   "Black Box",
        "caption": "XGBoost predicts $420k — but no-one can explain why.",
        "score":   30,
        "bar_vals": [0.30, 0.10, 0.05, 0.08],   # opacity-like importance bars (opaque = unknown)
        "colors":   [GREY, GREY, GREY, GREY],
        "known":    False,
    },
    {
        "label":   "SHAP Explained",
        "caption": "TreeSHAP reveals: MedInc+$95k, Location+$40k, Rooms+$15k, Age−$20k.",
        "score":   72,
        "bar_vals": [0.95, 0.40, 0.15, 0.20],
        "colors":   [GREEN, GREEN, GREEN, RED],
        "known":    True,
    },
    {
        "label":   "Feature Ranked",
        "caption": "Global mean|SHAP|: MedInc dominates — model trusted, regulatory-compliant.",
        "score":   91,
        "bar_vals": [0.95, 0.40, 0.20, 0.15],
        "colors":   [GREEN, GREEN, AMBER, AMBER],
        "known":    True,
    },
]

FEAT_NAMES = ["MedInc", "Location", "HouseAge", "AveRooms"]
FRAMES_PER_STAGE = 40
N_STAGES        = len(STAGES)
TOTAL_FRAMES    = FRAMES_PER_STAGE * (N_STAGES - 1) + 20  # hold last stage briefly


def _ease(t: float) -> float:
    t = max(0.0, min(1.0, float(t)))
    return 0.5 - 0.5 * math.cos(math.pi * t)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _get_interp(frame: int) -> dict:
    """Interpolate between stages based on frame index."""
    seg = min(frame // FRAMES_PER_STAGE, N_STAGES - 2)
    local_t = _ease((frame % FRAMES_PER_STAGE) / max(FRAMES_PER_STAGE - 1, 1))
    s0 = STAGES[seg]
    s1 = STAGES[min(seg + 1, N_STAGES - 1)]

    score    = _lerp(s0["score"],  s1["score"],  local_t)
    bar_vals = [_lerp(a, b, local_t) for a, b in zip(s0["bar_vals"], s1["bar_vals"])]
    colors   = s0["colors"] if local_t < 0.5 else s1["colors"]
    label    = s0["label"]  if local_t < 0.4 else s1["label"]
    caption  = s0["caption"] if local_t < 0.5 else s1["caption"]
    known    = s1["known"] if local_t > 0.5 else s0["known"]
    return dict(score=score, bar_vals=bar_vals, colors=colors,
                label=label, caption=caption, known=known)


def _draw_needle(ax_gauge, ax_bars, ax_title, state: dict) -> None:
    """Render one animation frame."""
    score   = state["score"]
    known   = state["known"]
    label   = state["label"]
    caption = state["caption"]
    bvals   = state["bar_vals"]
    colors  = state["colors"]

    # ── Title strip ──────────────────────────────────────────────────────────
    for ax in (ax_gauge, ax_bars, ax_title):
        ax.set_facecolor(BG)

    ax_title.clear()
    ax_title.set_facecolor(BG)
    ax_title.set_axis_off()
    ax_title.text(0.5, 0.65, "Ch.4 · SHAP Interpretability",
                  ha="center", va="center", fontsize=13, fontweight="bold",
                  color=LIGHT, transform=ax_title.transAxes)
    ax_title.text(0.5, 0.15, caption,
                  ha="center", va="center", fontsize=8.5, color=LIGHT,
                  style="italic", transform=ax_title.transAxes,
                  wrap=True)

    # ── Gauge (semi-circle needle) ────────────────────────────────────────────
    ax_gauge.clear()
    ax_gauge.set_facecolor(BG)
    ax_gauge.set_aspect("equal")
    ax_gauge.set_xlim(-1.3, 1.3)
    ax_gauge.set_ylim(-0.15, 1.25)
    ax_gauge.set_axis_off()

    # Background arc
    theta_bg = np.linspace(math.pi, 0, 300)
    ax_gauge.plot(np.cos(theta_bg), np.sin(theta_bg),
                  color=GREY, lw=12, alpha=0.35, solid_capstyle="round")

    # Coloured arc (score-proportional)
    ratio   = score / 100
    arc_col = GREEN if score > 70 else (AMBER if score > 45 else RED)
    theta_f = np.linspace(math.pi, math.pi - ratio * math.pi, 300)
    ax_gauge.plot(np.cos(theta_f), np.sin(theta_f),
                  color=arc_col, lw=12, solid_capstyle="round")

    # Needle
    angle = math.pi - ratio * math.pi
    nx, ny = math.cos(angle) * 0.82, math.sin(angle) * 0.82
    ax_gauge.annotate("",
                      xy=(nx, ny), xytext=(0, 0),
                      arrowprops=dict(arrowstyle="-|>", color=LIGHT,
                                      lw=2.0, mutation_scale=16))

    # Centre circle
    circ = plt.Circle((0, 0), 0.06, color=LIGHT, zorder=5)
    ax_gauge.add_patch(circ)

    # Score label
    ax_gauge.text(0, -0.08, f"{score:.0f} / 100",
                  ha="center", va="top", fontsize=16, fontweight="bold",
                  color=arc_col)
    ax_gauge.text(0, -0.07, "",
                  ha="center", va="top", fontsize=9, color=GREY)
    ax_gauge.text(-1.15, -0.05, "0", ha="center", va="top",
                  fontsize=8, color=GREY)
    ax_gauge.text(1.15, -0.05, "100", ha="center", va="top",
                  fontsize=8, color=GREY)
    ax_gauge.text(0, 1.05, "Trust Score", ha="center", va="bottom",
                  fontsize=10, color=LIGHT)
    ax_gauge.text(0, 0.88, label, ha="center", va="bottom",
                  fontsize=9, fontweight="bold", color=LIGHT)

    # ── Feature-bar subplot ───────────────────────────────────────────────────
    ax_bars.clear()
    ax_bars.set_facecolor(BG)
    ax_bars.set_xlim(-0.05, 1.10)
    ax_bars.set_ylim(-0.5, len(FEAT_NAMES) - 0.5)
    ax_bars.set_axis_off()

    ax_bars.text(0.5, len(FEAT_NAMES) - 0.05,
                 "Feature SHAP Contributions" if known else "Feature Importance",
                 ha="center", va="bottom", fontsize=9, fontweight="bold",
                 color=LIGHT, transform=ax_bars.transData)

    for i, (name, val, col) in enumerate(zip(FEAT_NAMES, bvals, colors)):
        y = len(FEAT_NAMES) - 1 - i
        # Background track
        ax_bars.barh(y, 1.0, color=GREY, alpha=0.15, height=0.55)
        # Value bar
        ax_bars.barh(y, val, color=col, height=0.55, alpha=0.9)
        # Label
        ax_bars.text(-0.03, y, name, ha="right", va="center",
                     fontsize=9, color=LIGHT)
        if known:
            ax_bars.text(val + 0.03, y, f"{val:.2f}",
                         ha="left", va="center", fontsize=8, color=col)
        else:
            ax_bars.text(0.5, y, "?", ha="center", va="center",
                         fontsize=12, color=GREY, fontweight="bold")


def build_needle_gif() -> None:
    fig = plt.figure(figsize=(9, 4.8), facecolor=BG)
    gs  = fig.add_gridspec(3, 2,
                           height_ratios=[0.18, 0.76, 0.06],
                           width_ratios=[1, 1],
                           hspace=0.06, wspace=0.12)

    ax_title = fig.add_subplot(gs[0, :])
    ax_gauge = fig.add_subplot(gs[1, 0])
    ax_bars  = fig.add_subplot(gs[1, 1])

    for ax in (ax_title, ax_gauge, ax_bars):
        ax.set_facecolor(BG)

    artists: list = []

    def _update(frame: int):
        nonlocal artists
        for a in artists:
            try:
                a.remove()
            except Exception:
                pass
        artists = []
        state = _get_interp(frame)
        _draw_needle(ax_gauge, ax_bars, ax_title, state)
        return []

    ani = animation.FuncAnimation(
        fig, _update, frames=TOTAL_FRAMES, interval=65, blit=False
    )

    out = IMG_DIR / "ch04-shap-needle.gif"
    ani.save(str(out), writer="pillow", fps=15, dpi=110)
    plt.close(fig)
    print(f"[OK] {out}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 · Progress-check bar chart
# ─────────────────────────────────────────────────────────────────────────────

CONSTRAINTS = [
    ("#1 Improvement >5%",     100, GREEN),
    ("#2 Diversity",           100, GREEN),
    ("#3 Efficiency <5×",      100, GREEN),
    ("#4 Interpretability",    100, LIGHT_BLU),   # unlocked this chapter
    ("#5 Robustness",          100, GREEN),
]

UNLOCKED_LABEL = "✓ Unlocked this chapter"
PREV_LABEL     = "✓ Achieved in Ch.1–3"


def build_progress_check() -> None:
    fig, ax = plt.subplots(figsize=(8, 3.6), facecolor=BG)
    ax.set_facecolor(BG)

    labels = [c[0] for c in CONSTRAINTS]
    values = [c[1] for c in CONSTRAINTS]
    colors = [c[2] for c in CONSTRAINTS]
    y      = np.arange(len(labels))

    # Background tracks
    ax.barh(y, [100] * len(y), color=GREY, alpha=0.18, height=0.52)

    # Value bars
    bars = ax.barh(y, values, color=colors, height=0.52, alpha=0.92)

    # Labels
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=LIGHT, fontsize=10)
    ax.set_xlim(0, 120)
    ax.set_xticks([])

    for bar, val, col in zip(bars, values, colors):
        ax.text(val + 2, bar.get_y() + bar.get_height() / 2,
                f"{val}%", va="center", fontsize=9,
                color=col, fontweight="bold")

    ax.spines[:].set_visible(False)
    ax.tick_params(left=False, bottom=False)

    ax.set_title("EnsembleAI — All 5 Constraints Met after Ch.4",
                 color=LIGHT, fontsize=12, fontweight="bold", pad=14)

    # Legend
    patch_prev = mpatches.Patch(color=GREEN,     label=PREV_LABEL)
    patch_new  = mpatches.Patch(color=LIGHT_BLU, label=UNLOCKED_LABEL)
    ax.legend(handles=[patch_prev, patch_new],
              loc="lower right", fontsize=8.5,
              facecolor=BG, edgecolor=GREY,
              labelcolor=LIGHT, framealpha=0.7)

    plt.tight_layout()
    out = IMG_DIR / "ch04-shap-progress-check.png"
    fig.savefig(str(out), dpi=130, facecolor=BG)
    plt.close(fig)
    print(f"[OK] {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating Ch.4 SHAP assets …")
    build_needle_gif()
    build_progress_check()
    print("Done.")
