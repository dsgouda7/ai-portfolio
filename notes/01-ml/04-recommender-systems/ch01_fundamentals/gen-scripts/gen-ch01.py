"""Generate ch01 Fundamentals visuals.

Produces
--------
../img/ch01-fundamentals-needle.gif
    Animated needle showing HR@10 moving from random baseline (0.6%)
    to popularity baseline (35%). Target zone at 85%.

../img/ch01-fundamentals-progress-check.png
    5-constraint status dashboard for FlixAI after Ch.1.

Usage
-----
    python gen_scripts/gen_ch01.py

Run from anywhere — paths are resolved relative to this file.
"""
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
rng = np.random.default_rng(SEED)

BG = "#1a1a2e"           # dark background
PANEL_BG = "#16213e"     # slightly lighter panel
PRIMARY = "#1e3a8a"      # deep blue
SUCCESS = "#15803d"      # green
CAUTION = "#b45309"      # amber
DANGER = "#b91c1c"       # red
INFO = "#1d4ed8"         # medium blue
TEXT = "#e2e8f0"         # near-white text
MUTED = "#64748b"        # muted grey
GRID = "#2d3748"         # subtle grid lines

IMG_DIR = Path(__file__).resolve().parent.parent / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# ── Needle GIF ───────────────────────────────────────────────────────────────

def _ease_in_out(t: float) -> float:
    """Smooth ease-in/ease-out interpolation (cosine)."""
    t = max(0.0, min(1.0, t))
    return 0.5 - 0.5 * math.cos(math.pi * t)


def _draw_needle_frame(ax, value: float, stages: list[dict]):
    """Draw a single needle frame onto ax."""
    ax.clear()
    ax.set_facecolor(BG)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.25, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Arc background ──────────────────────────────────────────────────────
    # Full arc from 0% (left) to 100% (right), spanning 180°
    theta_full = np.linspace(np.pi, 0, 300)
    r = 0.90
    ax.plot(np.cos(theta_full) * r, np.sin(theta_full) * r,
            color=GRID, lw=8, solid_capstyle="round")

    # Target zone: 85% → 100% highlighted in green
    target_start = np.pi * (1 - 0.85)   # angle for 85%
    theta_target = np.linspace(target_start, 0, 100)
    ax.plot(np.cos(theta_target) * r, np.sin(theta_target) * r,
            color=SUCCESS, lw=8, alpha=0.55, solid_capstyle="round")

    # Danger zone: 0% → 35% highlighted faintly in red
    danger_end = np.pi * (1 - 0.35)
    theta_danger = np.linspace(np.pi, danger_end, 100)
    ax.plot(np.cos(theta_danger) * r, np.sin(theta_danger) * r,
            color=DANGER, lw=8, alpha=0.25, solid_capstyle="round")

    # ── Tick marks ──────────────────────────────────────────────────────────
    for pct, label in [(0, "0%"), (25, "25%"), (50, "50%"),
                       (75, "75%"), (85, "85%"), (100, "100%")]:
        angle = np.pi * (1 - pct / 100)
        r_in = 0.78
        r_out = 0.95
        ax.plot([np.cos(angle) * r_in, np.cos(angle) * r_out],
                [np.sin(angle) * r_in, np.sin(angle) * r_out],
                color=MUTED, lw=1.2)
        r_label = 0.68
        col = SUCCESS if pct == 85 else TEXT
        ax.text(np.cos(angle) * r_label, np.sin(angle) * r_label,
                label, ha="center", va="center", fontsize=7,
                color=col, fontweight="bold" if pct == 85 else "normal")

    # ── Target line ─────────────────────────────────────────────────────────
    angle_85 = np.pi * (1 - 0.85)
    ax.plot([0, np.cos(angle_85) * r * 1.05],
            [0, np.sin(angle_85) * r * 1.05],
            color=SUCCESS, lw=1.5, linestyle="--", alpha=0.8)
    ax.text(np.cos(angle_85) * r * 1.12, np.sin(angle_85) * r * 1.12,
            "TARGET", ha="center", va="bottom", fontsize=7,
            color=SUCCESS, fontweight="bold")

    # ── Needle ──────────────────────────────────────────────────────────────
    angle = np.pi * (1 - value / 100)
    needle_r = 0.82
    ax.annotate(
        "",
        xy=(np.cos(angle) * needle_r, np.sin(angle) * needle_r),
        xytext=(0, 0),
        arrowprops=dict(
            arrowstyle="-|>",
            lw=3.5,
            color=CAUTION,
            mutation_scale=18,
        ),
    )
    # Hub circle
    hub = plt.Circle((0, 0), 0.055, color=CAUTION, zorder=10)
    ax.add_patch(hub)

    # ── Stage labels ────────────────────────────────────────────────────────
    for s in stages:
        s_angle = np.pi * (1 - s["value"] / 100)
        r_dot = r + 0.06
        dot_col = s.get("color", INFO)
        ax.plot(np.cos(s_angle) * r_dot, np.sin(s_angle) * r_dot,
                "o", color=dot_col, markersize=7, zorder=8)
        r_text = r + 0.19
        ax.text(np.cos(s_angle) * r_text, np.sin(s_angle) * r_text,
                s["label"], ha="center", va="center",
                fontsize=7.5, color=dot_col, fontweight="bold")

    # ── Current value display ───────────────────────────────────────────────
    ax.text(0, -0.10, f"{value:.1f}%",
            ha="center", va="center", fontsize=26, fontweight="bold",
            color=CAUTION)
    ax.text(0, -0.22, "HR@10",
            ha="center", va="center", fontsize=11, color=MUTED)

    # ── Title ───────────────────────────────────────────────────────────────
    ax.text(0, 1.18, "FlixAI — Hit Rate @10",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color=TEXT)
    ax.text(0, 1.08, "Ch.1: Random → Popularity Baseline",
            ha="center", va="center", fontsize=9, color=MUTED)


def make_needle_gif():
    """Produce ch01-fundamentals-needle.gif."""
    out_path = IMG_DIR / "ch01-fundamentals-needle.gif"

    # Stage waypoints  [value_pct, label, color]
    stages_info = [
        {"value": 0.6,  "label": "Random\n0.6%",   "color": DANGER},
        {"value": 35.0, "label": "Popularity\n35%", "color": CAUTION},
        {"value": 85.0, "label": "Target\n85%",     "color": SUCCESS},
    ]

    # Animation: hold at 0.6, sweep to 35, brief hold
    HOLD_FRAMES = 18
    SWEEP_FRAMES = 55
    FINAL_HOLD = 25
    total = HOLD_FRAMES + SWEEP_FRAMES + FINAL_HOLD

    fig, ax = plt.subplots(figsize=(5.5, 3.4), facecolor=BG)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

    drawn_stages = [stages_info[0], stages_info[2]]  # show random + target dots

    def _value_at(frame: int) -> float:
        if frame < HOLD_FRAMES:
            return 0.6
        elif frame < HOLD_FRAMES + SWEEP_FRAMES:
            t = (frame - HOLD_FRAMES) / SWEEP_FRAMES
            t_eased = _ease_in_out(t)
            return 0.6 + (35.0 - 0.6) * t_eased
        else:
            return 35.0

    frames_data = [_value_at(f) for f in range(total)]
    # Add popularity dot only after sweep
    stage_lists = []
    for f in range(total):
        if f < HOLD_FRAMES + SWEEP_FRAMES:
            stage_lists.append(drawn_stages)
        else:
            stage_lists.append(stages_info)   # show all 3 dots at end

    def animate(frame: int):
        _draw_needle_frame(ax, frames_data[frame], stage_lists[frame])

    ani = animation.FuncAnimation(
        fig, animate, frames=total, interval=50, blit=False
    )
    writer = animation.PillowWriter(fps=20)
    ani.save(str(out_path), writer=writer, dpi=110)
    plt.close(fig)
    print(f"✅  Saved {out_path}")


# ── Progress Check PNG ───────────────────────────────────────────────────────

def make_progress_check():
    """Produce ch01-fundamentals-progress-check.png."""
    out_path = IMG_DIR / "ch01-fundamentals-progress-check.png"

    constraints = [
        {
            "id": "#1",
            "name": "HIT_RATE",
            "target": ">85% HR@10",
            "status": "[X]  ~35% — 50 pp gap",
            "pct": 35,
            "color": DANGER,
        },
        {
            "id": "#2",
            "name": "COLD_START",
            "target": "New users / items",
            "status": "[~]  Partial — popularity fallback",
            "pct": 20,
            "color": CAUTION,
        },
        {
            "id": "#3",
            "name": "DIVERSITY",
            "target": "Coverage > 15%",
            "status": "[X]  Coverage < 1%",
            "pct": 1,
            "color": DANGER,
        },
        {
            "id": "#4",
            "name": "LATENCY",
            "target": "< 100ms at scale",
            "status": "[OK] Trivial (static list)",
            "pct": 100,
            "color": SUCCESS,
        },
        {
            "id": "#5",
            "name": "EXPLAINABILITY",
            "target": '"Because you watched X"',
            "status": "[X]  No personalization",
            "pct": 5,
            "color": DANGER,
        },
    ]

    fig, axes = plt.subplots(
        len(constraints), 1,
        figsize=(9, 6.5),
        facecolor=BG,
    )
    fig.subplots_adjust(hspace=0.55, left=0.02, right=0.98,
                        top=0.90, bottom=0.06)

    fig.text(
        0.50, 0.96,
        "FlixAI — Constraint Status after Ch.1",
        ha="center", va="center",
        fontsize=14, fontweight="bold", color=TEXT,
    )
    fig.text(
        0.50, 0.92,
        "Popularity baseline: HR@10 ≈ 35%  |  Random baseline: 0.6%  |  Target: >85%",
        ha="center", va="center",
        fontsize=9, color=MUTED,
    )

    bar_max = 100.0

    for ax, c in zip(axes, constraints):
        ax.set_facecolor(PANEL_BG)
        ax.set_xlim(0, bar_max)
        ax.set_ylim(-0.5, 1.5)
        ax.axis("off")

        # Background bar
        ax.barh(
            0.5, bar_max, height=0.7,
            left=0, color=GRID, align="center",
        )
        # Filled bar
        ax.barh(
            0.5, max(c["pct"], 0.8), height=0.7,
            left=0, color=c["color"], align="center", alpha=0.85,
        )
        # Target line at 85% for #1
        if c["id"] == "#1":
            ax.axvline(85, color=SUCCESS, lw=2, linestyle="--", alpha=0.9)
            ax.text(85.5, 1.35, "target 85%",
                    fontsize=7, color=SUCCESS, va="center")

        # Constraint ID + name
        ax.text(-0.5, 0.5,
                f"{c['id']}  {c['name']}",
                ha="right", va="center",
                transform=ax.get_yaxis_transform(),
                fontsize=9.5, fontweight="bold", color=TEXT)

        # Status text inside bar
        ax.text(1.5, 0.5, c["status"],
                ha="left", va="center",
                fontsize=8.5, color=TEXT, fontweight="bold")

        # Target on right
        ax.text(bar_max - 0.5, 0.5, c["target"],
                ha="right", va="center",
                fontsize=7.5, color=MUTED)

        # Percentage chip
        ax.text(c["pct"] + 0.8, 0.5, f"{c['pct']}%",
                ha="left", va="center",
                fontsize=8, color=c["color"], fontweight="bold")

    fig.savefig(str(out_path), dpi=130, bbox_inches="tight",
                facecolor=BG)
    plt.close(fig)
    print(f"✅  Saved {out_path}")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating ch01 Fundamentals visuals …")
    make_needle_gif()
    make_progress_check()
    print("Done.")
