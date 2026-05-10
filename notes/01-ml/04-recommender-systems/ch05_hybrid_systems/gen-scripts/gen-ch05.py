"""
gen_ch05.py — Generate visual assets for Ch.5 Hybrid Systems
Produces:
  img/ch05-hybrid-systems-needle.gif        — HR@10 gauge: NCF 82% → hybrid 85%
  img/ch05-hybrid-systems-progress-check.png — FlixAI constraint dashboard after Ch.5
"""

import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# ── constants ────────────────────────────────────────────────────────────────
SEED = 42
rng = np.random.default_rng(SEED)

BG      = "#1a1a2e"
PRIMARY = "#1e3a8a"
INFO    = "#1d4ed8"
SUCCESS = "#15803d"
CAUTION = "#b45309"
DANGER  = "#b91c1c"
FG      = "#e2e8f0"
MUTED   = "#94a3b8"
GOLD    = "#f59e0b"

OUT_DIR = pathlib.Path(__file__).parent.parent / "img"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. NEEDLE GIF  — HR@10 gauge: NCF 82% → hybrid 85%
# ─────────────────────────────────────────────────────────────────────────────

def _needle_frame(ax, hr_value: float, phase: str, frame_idx: int, n_frames: int):
    """Draw one HR@10 speedometer frame."""
    ax.clear()
    ax.set_facecolor(BG)

    hr_min, hr_max = 0.0, 100.0

    # Zone boundaries (HR@10 %)
    zone_bounds = [0, 42, 68, 78, 82, 85, 100]
    zone_colors = [DANGER, CAUTION, INFO, CAUTION, INFO, SUCCESS]
    zone_labels = [
        "0%",
        "42%\n(pop.)",
        "68%\n(CF)",
        "78%\n(MF)",
        "82%\n(NCF)",
        "85%\ntarget",
        "100%",
    ]

    r_inner, r_outer = 0.55, 0.95

    for i in range(len(zone_bounds) - 1):
        a_start = np.pi - (zone_bounds[i] / hr_max) * np.pi
        a_end   = np.pi - (zone_bounds[i + 1] / hr_max) * np.pi
        t = np.linspace(a_start, a_end, 60)
        xs_outer = r_outer * np.cos(t)
        ys_outer = r_outer * np.sin(t)
        xs_inner = r_inner * np.cos(t[::-1])
        ys_inner = r_inner * np.sin(t[::-1])
        ax.fill(
            np.concatenate([xs_outer, xs_inner]),
            np.concatenate([ys_outer, ys_inner]),
            color=zone_colors[i], alpha=0.75, zorder=2,
        )

    # Tick marks + labels
    for tb, tl in zip(zone_bounds, zone_labels):
        angle = np.pi - (tb / hr_max) * np.pi
        xi = 0.52 * np.cos(angle)
        yi = 0.52 * np.sin(angle)
        xo = 0.98 * np.cos(angle)
        yo = 0.98 * np.sin(angle)
        ax.plot([xi, xo], [yi, yo], color=FG, lw=1.0, zorder=3)
        xl = 1.13 * np.cos(angle)
        yl = 1.13 * np.sin(angle)
        ax.text(xl, yl, tl, ha="center", va="center", color=FG,
                fontsize=4.5, zorder=4, multialignment="center")

    # Needle
    needle_angle = np.pi - (hr_value / hr_max) * np.pi
    nx = 0.85 * np.cos(needle_angle)
    ny = 0.85 * np.sin(needle_angle)
    ax.annotate(
        "", xy=(nx, ny), xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=2.2, mutation_scale=14),
        zorder=6,
    )
    ax.plot(0, 0, "o", color=GOLD, ms=6, zorder=7)

    # Centre readout
    ax.text(0, 0.22, "HR@10", ha="center", va="center",
            color=MUTED, fontsize=7, zorder=5)
    ax.text(0, 0.10, f"{hr_value:.1f}%", ha="center", va="center",
            color=GOLD, fontsize=14, fontweight="bold", zorder=5)

    # Phase label
    ax.text(0, -0.20, phase, ha="center", va="center",
            color=FG, fontsize=7.5, zorder=5,
            bbox=dict(boxstyle="round,pad=0.25", facecolor=PRIMARY,
                      edgecolor=INFO, alpha=0.85))

    # Title
    ax.text(0, 0.58, "FlixAI — Ranking Quality", ha="center", va="center",
            color=FG, fontsize=8, fontweight="bold", zorder=5)

    # Progress bar at bottom
    bar_y = -0.60
    ax.plot([-0.85, 0.85], [bar_y, bar_y], color=MUTED, lw=2, alpha=0.3, zorder=4)
    progress = min(frame_idx / max(n_frames - 1, 1), 1.0)
    ax.plot([-0.85, -0.85 + 1.70 * progress], [bar_y, bar_y],
            color=INFO, lw=2, alpha=0.8, zorder=5)

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-0.80, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")


def make_needle_gif():
    """Animate HR@10 rising from NCF baseline (82%) to hybrid result (85%)."""
    # Keyframes: (hr_value_%, phase_label, hold_frames)
    keyframes = [
        (82.0, "Ch.4 Neural CF (NCF)\nHR@10 = 82%  Cold start ❌",   6),
        (82.2, "Add genre vectors\n(19-dim multi-hot)",               1),
        (82.5, "Director embeddings\nVilleneuve = Sci-Fi signal",     1),
        (83.0, "Weighted hybrid\nα·CF + (1-α)·CBF",                  2),
        (83.4, "Adaptive α = n/n_warm\nCold items → trust CBF",       1),
        (83.8, "User tower\nage + occupation + genre prefs",          1),
        (84.2, "Item tower\ngenres + year + director",                1),
        (84.5, "End-to-end joint training\nsampled negatives",        1),
        (84.8, "Val HR@10 rising...\nCold start improving",          1),
        (85.0, "Two-Tower Hybrid\nHR@10 = 85% ✅  Cold start ✅",     8),
    ]

    frames_data = []
    for i in range(len(keyframes) - 1):
        hr_start, phase_start, hold = keyframes[i]
        hr_end, _, _ = keyframes[i + 1]
        n_interp = 16
        for j in range(n_interp):
            t = j / n_interp
            # ease in-out cubic
            hr = hr_start + (hr_end - hr_start) * (3 * t**2 - 2 * t**3)
            frames_data.append((hr, phase_start))
        for _ in range(hold):
            frames_data.append((hr_start, phase_start))

    hr_end, phase_end, hold_end = keyframes[-1]
    for _ in range(hold_end):
        frames_data.append((hr_end, phase_end))

    n_frames = len(frames_data)
    pil_frames = []
    for idx, (hr_val, phase) in enumerate(frames_data):
        fig, ax = plt.subplots(figsize=(4.2, 3.2), facecolor=BG)
        _needle_frame(ax, hr_val, phase, idx, n_frames)
        fig.tight_layout(pad=0.3)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        pil_frames.append(Image.fromarray(buf[:, :, :3]))
        plt.close(fig)

    out_path = OUT_DIR / "ch05-hybrid-systems-needle.gif"
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        optimize=False,
        duration=110,
        loop=0,
    )
    print(f"Saved: {out_path}  ({len(pil_frames)} frames)")


# ─────────────────────────────────────────────────────────────────────────────
# 2. PROGRESS CHECK PNG
# ─────────────────────────────────────────────────────────────────────────────

def make_progress_check():
    """Two-panel dashboard: HR@10 bar chart on the left, constraint table on the right."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), facecolor=BG)
    fig.suptitle(
        "Ch.5 — Hybrid Systems: Progress Check",
        color=FG, fontsize=13, fontweight="bold", y=0.97,
    )

    # ── LEFT PANEL: HR@10 progress bar chart ────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(BG)

    milestones = [
        ("Popularity baseline\n(Ch.1)",          42,  DANGER),
        ("Collaborative Filtering\n(Ch.2)",       68,  CAUTION),
        ("Matrix Factorisation\n(Ch.3)",          78,  INFO),
        ("Neural CF  NCF\n(Ch.4)",               82,  INFO),
        ("Hybrid + Two-Tower\n(this chapter)",   85,  SUCCESS),
        ("FlixAI target",                         85,  SUCCESS),
    ]

    labels = [m[0] for m in milestones]
    values = [m[1] for m in milestones]
    colors = [m[2] for m in milestones]
    alphas = [0.85, 0.85, 0.85, 0.85, 0.95, 0.0]  # last is invisible (target line label)

    y_pos = np.arange(len(labels) - 1)  # skip last (target line)
    bars = ax.barh(
        y_pos, values[:-1],
        color=colors[:-1], alpha=0.85, height=0.55,
        edgecolor=BG, linewidth=1.5,
    )

    for bar, val in zip(bars, values[:-1]):
        ax.text(
            bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
            f"{val}%", va="center", ha="left",
            color=FG, fontsize=9, fontweight="bold",
        )

    # Target line
    ax.axvline(85, color=SUCCESS, lw=1.5, ls="--", alpha=0.8)
    ax.text(85.5, len(labels) - 2.4, ">85%\ntarget ✅", color=SUCCESS,
            fontsize=7, va="top")

    # "You are here" star on the winning bar
    ax.plot(85, 4, "*", color=GOLD, ms=14, zorder=6)
    ax.annotate(
        "FlixAI\ntarget\nhit! ✅", xy=(85, 4), xytext=(74, 4.5),
        arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.2),
        color=GOLD, fontsize=7.5, fontweight="bold",
    )

    # Cold-start improvement callout
    ax.text(
        50, -0.3,
        "Cold-start HR@10: 23% (Ch.4) → 78% (Ch.5)",
        color=INFO, fontsize=7.5, style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=PRIMARY, edgecolor=INFO, alpha=0.6),
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels[:-1], color=FG, fontsize=8)
    ax.set_xlabel("HR@10 (%)", color=MUTED, fontsize=9)
    ax.set_xlim(0, 100)
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(MUTED)
        sp.set_alpha(0.3)
    ax.set_title("Hit Rate @ Top-10 — Track Progress", color=FG, fontsize=10,
                 fontweight="bold", pad=8)

    # ── RIGHT PANEL: Constraint status ──────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(BG)
    ax2.axis("off")
    ax2.set_title("FlixAI Constraint Status After Ch.5", color=FG,
                  fontsize=10, fontweight="bold", pad=8)

    constraints = [
        ("#1 ACCURACY",       "HR@10 = 85% ✅",         SUCCESS, "Target crossed!"),
        ("#2 COLD START",     "Cold HR@10 = 78% ✅",    SUCCESS, "Two-tower fixes it"),
        ("#3 SCALE",          "< 200ms serving ✅",      SUCCESS, "Pre-computed embeddings"),
        ("#4 DIVERSITY",      "ILD@10 < 0.4 ❌",         DANGER,  "Genre bubble — Ch.6 fixes"),
        ("#5 EXPLAINABILITY", '"Sci-Fi by Villeneuve" ✅', SUCCESS, "Item tower features"),
    ]

    row_h = 0.16
    y_start = 0.88
    for i, (constraint, status, color, note) in enumerate(constraints):
        y = y_start - i * row_h
        # Constraint name
        ax2.text(0.03, y, constraint, transform=ax2.transAxes,
                 color=FG, fontsize=9, fontweight="bold", va="center")
        # Status badge
        ax2.text(0.52, y, status, transform=ax2.transAxes,
                 color=color, fontsize=8.5, fontweight="bold", va="center")
        # Note
        ax2.text(0.52, y - 0.045, note, transform=ax2.transAxes,
                 color=MUTED, fontsize=7.5, va="center", style="italic")
        # Separator line
        line_y = y - 0.07
        ax2.plot([0.02, 0.98], [line_y, line_y], color=MUTED, alpha=0.15, lw=0.8,
                 transform=ax2.transAxes)

    # Summary box at bottom
    summary_lines = [
        "Ch.5 Result:  85% HR@10  (target ✅)",
        "Cold-start:   78% HR@10  (structural fix ✅)",
        "Still open:   Constraint #4 Diversity → Ch.6",
    ]
    for i, line in enumerate(summary_lines):
        color = SUCCESS if "✅" in line and "open" not in line else (CAUTION if "open" in line else FG)
        ax2.text(0.03, 0.10 - i * 0.055, line, transform=ax2.transAxes,
                 color=color, fontsize=8, va="center",
                 fontweight="bold" if i == 0 else "normal")

    fig.tight_layout(rect=[0, 0.0, 1, 0.95])
    out_path = OUT_DIR / "ch05-hybrid-systems-progress-check.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating Ch.5 Hybrid Systems visuals...")
    make_needle_gif()
    make_progress_check()
    print("Done.")
