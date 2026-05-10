"""gen_ch04.py — Generate ch04 DQN visuals.

Produces:
  img/ch04-dqn-needle.gif          — agent capability needle: Q-table fails → DQN → ~150 steps
  img/ch04-dqn-progress-check.png  — CartPole progress bar chart across RL track

Run from repo root or from this directory:
    python notes/01-ml/06_reinforcement_learning/ch04_dqn/gen_scripts/gen_ch04.py
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
    """Animate the agent-capability needle showing DQN training stages on CartPole."""

    stages = [
        {
            "label": "Q-Table Attempt\n(fails on CartPole)",
            "value": 15,
            "caption": "Q-table cannot represent continuous states.\n"
                       "100⁴ = 10⁸ bins — most never visited, zero generalization.",
        },
        {
            "label": "DQN: Random Init\n(high ε exploration)",
            "value": 28,
            "caption": "Neural network initialized randomly.\n"
                       "ε=1.0 → pure exploration, buffer filling with diverse transitions.",
        },
        {
            "label": "DQN + Replay Buffer\n(correlation broken)",
            "value": 70,
            "caption": "Experience replay decouples collection from training.\n"
                       "Random minibatches drop correlation from ρ≈0.9 to ρ≈0.0001.",
        },
        {
            "label": "DQN + Target Network\n(~150/200 steps)",
            "value": 88,
            "caption": "Target network freezes ŷ for C=100 steps.\n"
                       "Loss converges. CartPole: ~150/200 avg steps.",
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

    # Colour bands: danger → caution → info → success
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

    # Tick marks and labels
    for pct, lbl in [(0, "0"), (25, "50"), (50, "100"), (75, "150"), (100, "200")]:
        t = np.pi * (1 - pct / 100)
        x_out, y_out = 1.08 * np.cos(t), 1.08 * np.sin(t)
        x_in,  y_in  = 0.74 * np.cos(t), 0.74 * np.sin(t)
        ax.plot([x_in, x_out], [y_in, y_out], color=LIGHT, lw=1.2, zorder=3)
        ax.text(1.20 * np.cos(t), 1.20 * np.sin(t), lbl,
                ha="center", va="center", fontsize=8, color=MUTED)

    ax.text(0, -0.18, "CartPole avg steps / episode", ha="center", va="center",
            fontsize=9, color=MUTED, style="italic")

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-0.45, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")

    # Needle and center dot (mutable)
    needle_line, = ax.plot([], [], color=LIGHT, lw=3.5, solid_capstyle="round", zorder=5)
    center_dot   = ax.scatter([0], [0], s=90, color=LIGHT, zorder=6)

    stage_text = ax.text(0, 1.25, "", ha="center", va="center",
                         fontsize=11, fontweight="bold", color=LIGHT)
    caption_text = fig.text(0.5, 0.06, "", ha="center", va="center",
                            fontsize=9.5, color=MUTED,
                            bbox=dict(boxstyle="round,pad=0.4", fc=DARK_CARD, ec="#334155"))

    def _needle_angle(value: float) -> float:
        """Convert 0–100 value to angle in radians (π at left, 0 at right)."""
        return np.pi * (1.0 - value / 100.0)

    def _get_state(frame: int):
        total_per_stage = NEEDLE_FRAMES_PER_STAGE + PAUSE_FRAMES
        stage_idx = min(frame // total_per_stage, len(stages) - 1)
        local = frame % total_per_stage
        if local < NEEDLE_FRAMES_PER_STAGE:
            prev_val = stages[max(stage_idx - 1, 0)]["value"]
            curr_val = stages[stage_idx]["value"]
            t = local / NEEDLE_FRAMES_PER_STAGE
            t_eased = 0.5 - 0.5 * np.cos(np.pi * t)
            value = prev_val + (curr_val - prev_val) * t_eased
        else:
            value = stages[stage_idx]["value"]
        return value, stage_idx

    def animate(frame: int):
        value, stage_idx = _get_state(frame)
        ang = _needle_angle(value)
        needle_r = 0.88
        needle_line.set_data([0, needle_r * np.cos(ang)], [0, needle_r * np.sin(ang)])
        stage_text.set_text(stages[stage_idx]["label"])
        caption_text.set_text(stages[stage_idx]["caption"])
        return needle_line, stage_text, caption_text

    ani = animation.FuncAnimation(
        fig, animate, frames=total_frames,
        interval=80, blit=False, repeat=True,
    )

    gif_path = IMG_DIR / "ch04-dqn-needle.gif"
    ani.save(gif_path, writer="pillow", fps=12, dpi=100)
    plt.close(fig)
    print(f"wrote {gif_path}")


# ─── 2. Progress Check PNG ─────────────────────────────────────────────────────
def make_progress_check() -> None:
    """Horizontal bar chart showing AgentAI CartPole progress across RL track."""

    # (label, value_out_of_200, color, annotation)
    milestones = [
        ("Random policy",               20,  DANGER,  "~20 steps"),
        ("Q-table (discrete GridWorld)", 0,   DANGER,  "fails\n(continuous)"),
        ("DQN (this chapter)",          150,  CAUTION, "~150/200"),
        ("Goal: ≥195/200",              195,  SUCCESS, "target\n(Ch.5)"),
    ]

    fig, ax = plt.subplots(figsize=(10, 4), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    fig.subplots_adjust(left=0.34, right=0.92, top=0.88, bottom=0.15)

    max_val = 200
    bar_height = 0.52

    for i, (label, value, color, annot) in enumerate(milestones):
        y = i
        # Background bar
        ax.barh(y, max_val, height=bar_height, left=0,
                color="#1e293b", edgecolor="none", zorder=1)
        # Value bar
        if value > 0:
            ax.barh(y, value, height=bar_height, left=0,
                    color=color, edgecolor="none", alpha=0.90, zorder=2)
        # Goal line
        if "≥195" in label:
            ax.axvline(195, color=SUCCESS, lw=1.5, linestyle="--", alpha=0.6, zorder=3)

        # Label on left
        ax.text(-4, y, label, ha="right", va="center",
                fontsize=9.5, color=LIGHT, fontweight="bold" if "DQN" in label else "normal")
        # Annotation on right of bar
        if value > 0:
            xpos = min(value + 4, max_val - 2)
            ax.text(xpos, y, annot, ha="left", va="center",
                    fontsize=8.5, color=LIGHT)
        else:
            ax.text(6, y, annot, ha="left", va="center",
                    fontsize=8.5, color=MUTED, style="italic")

    # Goal marker annotation
    ax.text(195, len(milestones) - 0.55, "195", ha="center", va="bottom",
            fontsize=8, color=SUCCESS, alpha=0.8)

    ax.set_xlim(-2, max_val + 14)
    ax.set_ylim(-0.6, len(milestones) - 0.3)
    ax.set_xlabel("Average steps per episode (CartPole-v1, max=200)",
                  color=MUTED, fontsize=9)
    ax.xaxis.set_tick_params(colors=MUTED)
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#334155")
    ax.set_facecolor(DARK_BG)

    ax.set_title("AgentAI CartPole Progress — After Ch.4 DQN",
                 color=LIGHT, fontsize=12, fontweight="bold", pad=10)

    # Legend patches
    legend_elements = [
        mpatches.Patch(color=DANGER,  label="Fails / random"),
        mpatches.Patch(color=CAUTION, label="DQN result (~150)"),
        mpatches.Patch(color=SUCCESS, label="Goal (≥195)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.15,
              labelcolor=LIGHT, fontsize=8.5, edgecolor="#334155")

    png_path = IMG_DIR / "ch04-dqn-progress-check.png"
    fig.savefig(png_path, dpi=130, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"wrote {png_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    make_needle_gif()
    make_progress_check()
    print("done.")
