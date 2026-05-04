"""gen_ch01.py — Generate ch01 MDP visuals.

Produces:
  img/ch01-mdps-needle.gif   — agent capability needle: no framework → MDP formalized
  img/ch01-mdps-progress-check.png — progress check bar chart

Run from repo root or from this directory:
    python notes/01-ml/06_reinforcement_learning/ch01_mdps/gen_scripts/gen_ch01.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import numpy as np

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
IMG_DIR = SCRIPT_DIR.parent / "img"
IMG_DIR.mkdir(exist_ok=True)

# ─── Shared style ─────────────────────────────────────────────────────────────
DARK_BG   = "#1a1a2e"
DARK_CARD = "#16213e"
PRIMARY   = "#1e3a8a"
SUCCESS   = "#15803d"
CAUTION   = "#b45309"
DANGER    = "#b91c1c"
INFO      = "#1d4ed8"
LIGHT     = "#e2e8f0"
MUTED     = "#94a3b8"

RNG = np.random.default_rng(42)


# ─── 1. Needle GIF ────────────────────────────────────────────────────────────
def make_needle_gif() -> None:
    """Animate the agent-capability needle from 'No RL Framework' to 'MDP Formalized'."""

    stages = [
        {
            "label": "No RL Framework",
            "value": 5,
            "caption": "We know supervised/unsupervised learning\nbut have no language for sequential decisions.",
        },
        {
            "label": "MDP Defined\n(S, A, P, R, γ)",
            "value": 35,
            "caption": "The 5-tuple formalizes the problem:\nstates, actions, transitions, rewards, discount.",
        },
        {
            "label": "Bellman Equations\nDerived",
            "value": 62,
            "caption": "Recursive value structure revealed:\nV*(s) = max_a Σ P[R + γV*(s')]",
        },
        {
            "label": "MDP Fully\nFormalized",
            "value": 82,
            "caption": "Optimal policy defined mathematically.\nReady for ch02 Dynamic Programming.",
        },
    ]

    NEEDLE_FRAMES_PER_STAGE = 30
    PAUSE_FRAMES = 20
    total_frames = len(stages) * (NEEDLE_FRAMES_PER_STAGE + PAUSE_FRAMES)

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=DARK_BG)
    ax.set_facecolor(DARK_CARD)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.80, bottom=0.25)

    # Gauge arc background
    theta = np.linspace(np.pi, 0, 300)
    arc_r = 1.0
    ax.fill_between(
        arc_r * np.cos(theta), arc_r * np.sin(theta),
        0.75 * np.cos(theta), 0.75 * np.sin(theta),
        color="#1e293b", zorder=1,
    )

    # Colour bands: red → amber → green
    bands = [
        (0,   25,  DANGER),
        (25,  50,  CAUTION),
        (50,  75,  INFO),
        (75, 100,  SUCCESS),
    ]
    for lo, hi, colour in bands:
        t_lo = np.pi * (1 - lo  / 100)
        t_hi = np.pi * (1 - hi  / 100)
        t_band = np.linspace(t_lo, t_hi, 60)
        ax.fill_between(
            arc_r * np.cos(t_band), arc_r * np.sin(t_band),
            0.78 * np.cos(t_band), 0.78 * np.sin(t_band),
            color=colour, alpha=0.85, zorder=2,
        )

    # Tick marks
    for pct in range(0, 101, 10):
        angle = np.pi * (1 - pct / 100)
        r_inner, r_outer = 0.72, 0.78
        ax.plot(
            [r_inner * np.cos(angle), r_outer * np.cos(angle)],
            [r_inner * np.sin(angle), r_outer * np.sin(angle)],
            color=LIGHT, lw=1.2, zorder=3,
        )
        if pct % 25 == 0:
            labels = {0: "None", 25: "Defined", 50: "Equations", 75: "Solved", 100: "Optimal"}
            ax.text(
                0.95 * np.cos(angle), 0.95 * np.sin(angle),
                labels.get(pct, ""),
                ha="center", va="center", fontsize=7, color=MUTED,
                rotation=np.degrees(angle) - 90,
            )

    # Needle (will be updated)
    def angle_for(val):
        return np.pi * (1 - val / 100)

    needle_line, = ax.plot([], [], color=LIGHT, lw=3.0, zorder=5, solid_capstyle="round")
    needle_dot = ax.scatter([], [], color=LIGHT, s=80, zorder=6)
    pivot_dot   = ax.scatter([0], [0], color=MUTED, s=40, zorder=6)

    stage_text  = ax.text(0, -0.55, "", ha="center", va="center",
                          fontsize=12, fontweight="bold", color=LIGHT, zorder=7)
    caption_text = ax.text(0, -0.78, "", ha="center", va="center",
                           fontsize=8.5, color=MUTED, zorder=7, wrap=True,
                           multialignment="center")
    title_text   = ax.text(0, 1.25, "Ch.1 MDPs — Agent Capability Needle",
                           ha="center", va="center", fontsize=12,
                           fontweight="bold", color=LIGHT, zorder=7)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.0, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")

    # Build frame sequence: for each stage, animate needle sweep then pause
    frame_schedule = []
    prev_val = 0
    for si, stg in enumerate(stages):
        target = stg["value"]
        vals = np.linspace(prev_val, target, NEEDLE_FRAMES_PER_STAGE)
        for v in vals:
            frame_schedule.append((v, stg["label"], stg["caption"]))
        for _ in range(PAUSE_FRAMES):
            frame_schedule.append((target, stg["label"], stg["caption"]))
        prev_val = target

    def update(frame_idx):
        val, lbl, cap = frame_schedule[frame_idx]
        ang = angle_for(val)
        nx, ny = 0.68 * np.cos(ang), 0.68 * np.sin(ang)
        needle_line.set_data([0, nx], [0, ny])
        needle_dot.set_offsets([[nx, ny]])
        stage_text.set_text(lbl.replace("\n", " "))
        caption_text.set_text(cap)
        return needle_line, needle_dot, stage_text, caption_text

    ani = animation.FuncAnimation(
        fig, update, frames=len(frame_schedule),
        interval=50, blit=True, repeat=True,
    )

    gif_path = IMG_DIR / "ch01-mdps-needle.gif"
    ani.save(str(gif_path), writer="pillow", fps=20, dpi=100)
    plt.close(fig)
    print(f"Saved: {gif_path}")


# ─── 2. Progress Check PNG ────────────────────────────────────────────────────
def make_progress_png() -> None:
    """Horizontal bar chart showing what ch01 unlocks vs what remains."""

    items = [
        # (label, pct_complete, colour)
        ("MDP formalism\n(S, A, P, R, gamma)",           100, SUCCESS),
        ("Return G_t with\nexplicit arithmetic",          100, SUCCESS),
        ("Bellman eq. for V-pi\n(derived & solved)",      100, SUCCESS),
        ("Q-function Q-pi(s,a)\n(defined & computed)",    100, SUCCESS),
        ("Bellman optimality\nV*, Q*  (defined)",          60, CAUTION),
        ("Algorithm to compute\nV* for large MDPs",         0, DANGER),
        ("Model-free RL\n(no transition model)",            0, DANGER),
        ("Continuous state\nspaces (CartPole)",             0, DANGER),
        ("Exploration strategy\n(EFFICIENCY)",              0, DANGER),
    ]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=DARK_BG)
    ax.set_facecolor(DARK_CARD)
    fig.subplots_adjust(left=0.38, right=0.92, top=0.88, bottom=0.08)

    n = len(items)
    y_pos = np.arange(n)

    for i, (lbl, pct, clr) in enumerate(items):
        # Background track
        ax.barh(i, 100, left=0, height=0.55, color="#1e293b", zorder=1)
        # Filled portion
        if pct > 0:
            ax.barh(i, pct, left=0, height=0.55, color=clr, alpha=0.88, zorder=2)
        # Percentage label
        ax.text(pct + 2, i, f"{pct}%", va="center", ha="left",
                color=LIGHT if pct > 0 else MUTED, fontsize=9, zorder=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([item[0] for item in items],
                       fontsize=9, color=LIGHT)
    ax.set_xlim(0, 115)
    ax.set_xlabel("% complete after Ch.1", color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

    ax.set_title("Ch.1 MDPs — Progress Check", color=LIGHT,
                 fontsize=12, fontweight="bold", pad=12)

    legend_patches = [
        mpatches.Patch(color=SUCCESS, label="Fully unlocked"),
        mpatches.Patch(color=CAUTION, label="Partially defined"),
        mpatches.Patch(color=DANGER,  label="Not yet addressed"),
    ]
    ax.legend(handles=legend_patches, loc="lower right",
              framealpha=0.3, facecolor=DARK_BG, edgecolor="#334155",
              fontsize=8, labelcolor=LIGHT)

    png_path = IMG_DIR / "ch01-mdps-progress-check.png"
    fig.savefig(str(png_path), dpi=130, facecolor=DARK_BG, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    make_needle_gif()
    make_progress_png()
    print("Done.")
