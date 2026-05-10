"""gen_ch02.py
-----------
Generates visual assets for Ch.2 — Collaborative Filtering:

  img/ch02-collaborative-filtering-needle.gif
      HR@10 needle animation: Popularity 35% → User-CF 60% → Item-CF 65%

  img/ch02-collaborative-filtering-progress-check.png
      FlixAI constraint tracker + HR@10 bar chart

Usage:
    python gen_scripts/gen_ch02.py
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

# ─── Paths ────────────────────────────────────────────────────────────────────

OUT_DIR = Path(__file__).parent.parent / "img"

# ─── Constants ────────────────────────────────────────────────────────────────

SEED = 42
RNG = np.random.default_rng(SEED)

# Dark-background colour palette
BG      = "#1a1a2e"
PRIMARY = "#1e3a8a"
SUCCESS = "#15803d"
CAUTION = "#b45309"
DANGER  = "#b91c1c"
INFO    = "#1d4ed8"
LIGHT   = "#e2e8f0"
GREY    = "#475569"
WHITE   = "#f8fafc"

TARGET_HR = 85.0

STAGES = [
    {
        "label":   "Ch.1 Popularity\nBaseline",
        "value":   35.0,
        "caption": "Same top-10 for everyone. No personalisation.",
        "color":   DANGER,
    },
    {
        "label":   "User-Based CF\n(K=30, Pearson)",
        "value":   60.0,
        "caption": "Similar users' ratings drive personalised recommendations.",
        "color":   CAUTION,
    },
    {
        "label":   "Item-Based CF\n(K=20, adj. cosine)",
        "value":   65.0,
        "caption": "Similar items, precomputed offline. Scalable + accurate.",
        "color":   SUCCESS,
    },
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _ease(t: float) -> float:
    """Smooth ease-in-out (cosine interpolation)."""
    t = max(0.0, min(1.0, float(t)))
    return 0.5 - 0.5 * math.cos(math.pi * t)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


# ─── Needle GIF ───────────────────────────────────────────────────────────────

def _draw_gauge(ax: plt.Axes, value: float) -> None:
    """Semicircular gauge showing HR@10 progress."""
    ax.clear()
    ax.set_facecolor(BG)
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.4, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("HR@10", fontsize=13, fontweight="bold", color=LIGHT, pad=6)

    # Track (grey arc)
    theta_full = np.linspace(np.pi, 0.0, 300)
    r = 1.0
    ax.plot(r * np.cos(theta_full), r * np.sin(theta_full),
            color=GREY, linewidth=10, alpha=0.35, solid_capstyle="round")

    # Coloured filled arc up to current value
    pct = value / 100.0
    theta_fill = np.linspace(np.pi, np.pi - pct * np.pi, 300)
    arc_color = SUCCESS if value >= 65 else (CAUTION if value >= 50 else DANGER)
    ax.plot(r * np.cos(theta_fill), r * np.sin(theta_fill),
            color=arc_color, linewidth=10, solid_capstyle="round")

    # Target marker at 85 %
    t_tgt = np.pi - (TARGET_HR / 100.0) * np.pi
    ax.plot([1.12 * np.cos(t_tgt)], [1.12 * np.sin(t_tgt)],
            "v", color=INFO, markersize=11, zorder=5)
    ax.text(1.22 * np.cos(t_tgt), 1.22 * np.sin(t_tgt),
            f"{TARGET_HR:.0f}%\ntarget",
            ha="center", va="bottom", fontsize=7, color=INFO)

    # Needle arrow
    needle_theta = np.pi - pct * np.pi
    nx, ny = 0.88 * np.cos(needle_theta), 0.88 * np.sin(needle_theta)
    ax.annotate(
        "", xy=(nx, ny), xytext=(0.0, -0.05),
        arrowprops=dict(arrowstyle="-|>", color=LIGHT, lw=2.5, mutation_scale=18),
    )
    ax.plot([0], [-0.05], "o", color=LIGHT, markersize=7, zorder=6)

    # Value label
    ax.text(0.0, -0.22, f"{value:.1f}%",
            ha="center", va="center", fontsize=24, fontweight="bold", color=LIGHT)


def _draw_bars(ax: plt.Axes, current_value: float, stage_idx: int) -> None:
    """Bar chart showing method-by-method HR@10 progression."""
    ax.clear()
    ax.set_facecolor(BG)

    labels = [s["label"] for s in STAGES]
    bar_values = []
    bar_colors = []
    for i, s in enumerate(STAGES):
        if i < stage_idx:
            bar_values.append(s["value"])
            bar_colors.append(SUCCESS)
        elif i == stage_idx:
            bar_values.append(current_value)
            bar_colors.append(CAUTION)
        else:
            bar_values.append(0.0)
            bar_colors.append(GREY)

    bars = ax.bar(range(len(STAGES)), bar_values, color=bar_colors,
                  edgecolor=LIGHT, linewidth=0.6, width=0.55, zorder=3)

    ax.axhline(TARGET_HR, color=INFO, linestyle="--", linewidth=1.8,
               alpha=0.8, zorder=4, label=f"Target {TARGET_HR:.0f}%")

    for bar, val in zip(bars, bar_values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2.0, val + 1.5,
                    f"{val:.0f}%", ha="center", va="bottom",
                    fontsize=9, color=LIGHT, fontweight="bold")

    ax.set_xticks(range(len(STAGES)))
    ax.set_xticklabels(labels, fontsize=8, color=LIGHT)
    ax.set_ylim(0, 100)
    ax.set_ylabel("HR@10 (%)", fontsize=9, color=LIGHT)
    ax.set_title("Hit Rate @ 10 — method progression", fontsize=11,
                 fontweight="bold", color=LIGHT, pad=8)
    ax.tick_params(colors=LIGHT)
    for spine in ax.spines.values():
        spine.set_edgecolor(GREY)
    ax.grid(axis="y", alpha=0.12, color=GREY, zorder=0)
    ax.legend(fontsize=8, facecolor=BG, labelcolor=LIGHT, edgecolor=GREY)


def make_needle_gif() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frames_per_transition = 14
    hold_frames = 5
    n_frames = (len(STAGES) - 1) * frames_per_transition + hold_frames

    fig = plt.figure(figsize=(12, 5.5), facecolor=BG)
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[3.8, 0.8],
        width_ratios=[1.05, 1.4],
        hspace=0.28,
        wspace=0.25,
    )
    ax_gauge  = fig.add_subplot(gs[0, 0])
    ax_bar    = fig.add_subplot(gs[0, 1])
    ax_caption = fig.add_subplot(gs[1, :])
    ax_caption.set_facecolor(BG)
    ax_caption.axis("off")

    caption_text = ax_caption.text(
        0.5, 0.5, "",
        ha="center", va="center", fontsize=11, color=LIGHT,
        transform=ax_caption.transAxes,
        bbox=dict(boxstyle="round,pad=0.45", fc=PRIMARY, ec=INFO, alpha=0.85),
    )

    fig.suptitle(
        "Ch.2 — Collaborative Filtering: Personalisation lifts HR@10",
        fontsize=13, fontweight="bold", color=LIGHT, y=1.01,
    )

    def animate(frame: int):
        seg = min(frame // frames_per_transition, len(STAGES) - 2)
        local_t = (frame % frames_per_transition) / max(frames_per_transition - 1, 1)
        local_t = _ease(local_t)

        v0 = STAGES[seg]["value"]
        v1 = STAGES[min(seg + 1, len(STAGES) - 1)]["value"]
        current = _lerp(v0, v1, local_t)

        _draw_gauge(ax_gauge, current)
        _draw_bars(ax_bar, current, seg)

        stage = STAGES[min(seg + 1, len(STAGES) - 1)]
        caption_text.set_text(f"{stage['label'].replace(chr(10), ' ')}: {stage['caption']}")
        return []

    gif_path = OUT_DIR / "ch02-collaborative-filtering-needle.gif"
    ani = animation.FuncAnimation(
        fig, animate, frames=n_frames, interval=250, blit=False, repeat=True,
    )
    ani.save(gif_path, writer="pillow", fps=4, dpi=100)
    plt.close(fig)
    print(f"wrote {gif_path}")


# ─── Progress Check PNG ───────────────────────────────────────────────────────

def make_progress_check() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, (ax_bar, ax_constraints) = plt.subplots(
        1, 2, figsize=(14, 6), facecolor=BG,
    )
    fig.suptitle(
        "Ch.2 Progress Check — Collaborative Filtering",
        fontsize=15, fontweight="bold", color=LIGHT, y=1.02,
    )

    # ── Left panel: HR@10 bar chart ─────────────────────────────────────────
    ax_bar.set_facecolor(BG)
    methods = [s["label"] for s in STAGES] + ["Target"]
    values  = [s["value"] for s in STAGES] + [TARGET_HR]
    colors  = [s["color"] for s in STAGES] + [INFO]

    bars = ax_bar.bar(range(len(methods)), values, color=colors,
                      edgecolor=LIGHT, linewidth=0.7, width=0.55, zorder=3)
    ax_bar.axhline(TARGET_HR, color=INFO, linestyle="--", linewidth=2,
                   alpha=0.75, zorder=4)

    for bar, val in zip(bars, values):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2.0, val + 1.5,
            f"{val:.0f}%", ha="center", va="bottom",
            fontsize=10, color=LIGHT, fontweight="bold",
        )

    ax_bar.set_xticks(range(len(methods)))
    ax_bar.set_xticklabels(methods, fontsize=9, color=LIGHT)
    ax_bar.set_ylim(0, 100)
    ax_bar.set_ylabel("Hit Rate @ 10 (%)", fontsize=11, color=LIGHT)
    ax_bar.set_title("HR@10 by Method", fontsize=13, fontweight="bold",
                     color=LIGHT, pad=10)
    ax_bar.tick_params(colors=LIGHT)
    for spine in ax_bar.spines.values():
        spine.set_edgecolor(GREY)
    ax_bar.grid(axis="y", alpha=0.12, color=GREY, zorder=0)

    # ── Right panel: FlixAI constraint tracker ───────────────────────────────
    ax_constraints.set_facecolor(BG)
    ax_constraints.axis("off")
    ax_constraints.set_title(
        "FlixAI Constraint Tracker", fontsize=13,
        fontweight="bold", color=LIGHT, pad=10,
    )

    constraints = [
        ("#1  ACCURACY  >85% HR@10",  "partial",  "65% so far — 20 pts to go",       CAUTION),
        ("#2  COLD START",            "fail",      "New users see popularity fallback", DANGER),
        ("#3  SCALABILITY  <200ms",   "partial",   "Item-CF: offline precompute OK\nUser-CF: O(n^2) online FAIL", CAUTION),
        ("#4  DIVERSITY",             "pass",      "Different users get different lists", SUCCESS),
        ("#5  EXPLAINABILITY",        "pass",      '"Users like you also liked..."',  SUCCESS),
    ]

    row_h = 0.155
    y_start = 0.90

    for i, (label, status, note, color) in enumerate(constraints):
        y = y_start - i * (row_h + 0.025)
        icon = "[OK]" if status == "pass" else ("[FAIL]" if status == "fail" else "[PART]")

        # Box
        ax_constraints.add_patch(mpatches.FancyBboxPatch(
            (0.02, y - row_h * 0.7), 0.96, row_h,
            boxstyle="round,pad=0.015",
            fc=BG, ec=color, linewidth=1.6,
            transform=ax_constraints.transAxes,
        ))

        ax_constraints.text(
            0.07, y - row_h * 0.12,
            f"{icon}  {label}",
            ha="left", va="center",
            fontsize=9, color=LIGHT, fontweight="bold",
            transform=ax_constraints.transAxes,
        )
        ax_constraints.text(
            0.07, y - row_h * 0.62,
            note,
            ha="left", va="center",
            fontsize=8, color=color,
            transform=ax_constraints.transAxes,
        )

    fig.tight_layout()
    png_path = OUT_DIR / "ch02-collaborative-filtering-progress-check.png"
    fig.savefig(png_path, dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"wrote {png_path}")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    make_needle_gif()
    make_progress_check()
