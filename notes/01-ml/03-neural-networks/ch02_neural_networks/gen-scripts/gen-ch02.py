"""
gen_ch02.py — Generate visual assets for Ch.2 Neural Networks
Produces:
  img/ch02-neural-networks-needle.gif   — MAE needle: linear $70k → architecture ~$55k potential
  img/ch02-neural-networks-progress-check.png — UnifiedAI constraint dashboard
"""

import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from PIL import Image

# ── constants ────────────────────────────────────────────────────────────────
SEED = 42
rng = np.random.default_rng(SEED)

BG = "#1a1a2e"
PRIMARY   = "#1e3a8a"
INFO      = "#1d4ed8"
SUCCESS   = "#15803d"
CAUTION   = "#b45309"
DANGER    = "#b91c1c"
FG        = "#e2e8f0"
MUTED     = "#94a3b8"
GOLD      = "#f59e0b"

OUT_DIR = pathlib.Path(__file__).parent.parent / "img"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. NEEDLE GIF
# ─────────────────────────────────────────────────────────────────────────────

def _needle_frame(ax, mae_value: float, phase: str, frame_idx: int, n_frames: int):
    """Draw one MAE-needle speedometer frame."""
    ax.clear()
    ax.set_facecolor(BG)

    # Gauge arc (180° semicircle, 0° = left, 180° = right)
    theta = np.linspace(np.pi, 0, 300)
    r_inner, r_outer = 0.55, 0.95

    # Background arc segments: danger → caution → success zones
    # MAE range: 0 (best) → 90k (worst), mapped to angle 0→180°
    mae_min, mae_max = 0, 90
    zone_bounds = [0, 28, 55, 70, 90]  # in $k
    zone_colors = [SUCCESS, INFO, CAUTION, DANGER]

    for i in range(len(zone_bounds) - 1):
        a_start = np.pi - (zone_bounds[i] / mae_max) * np.pi
        a_end   = np.pi - (zone_bounds[i + 1] / mae_max) * np.pi
        t = np.linspace(a_start, a_end, 60)
        xs_outer = r_outer * np.cos(t)
        ys_outer = r_outer * np.sin(t)
        xs_inner = r_inner * np.cos(t[::-1])
        ys_inner = r_inner * np.sin(t[::-1])
        xs = np.concatenate([xs_outer, xs_inner])
        ys = np.concatenate([ys_outer, ys_inner])
        ax.fill(xs, ys, color=zone_colors[i], alpha=0.75, zorder=2)

    # Tick marks + labels
    tick_maes = [0, 28, 40, 55, 70, 90]
    tick_labels = ["$0k", "$28k\n(target)", "$40k", "$55k\n(arch)", "$70k\n(linear)", "$90k"]
    for tm, tl in zip(tick_maes, tick_labels):
        angle = np.pi - (tm / mae_max) * np.pi
        xi = 0.52 * np.cos(angle)
        yi = 0.52 * np.sin(angle)
        xo = 0.98 * np.cos(angle)
        yo = 0.98 * np.sin(angle)
        ax.plot([xi, xo], [yi, yo], color=FG, lw=1.0, zorder=3)
        xl = 1.12 * np.cos(angle)
        yl = 1.12 * np.sin(angle)
        ax.text(xl, yl, tl, ha="center", va="center", color=FG,
                fontsize=5.5, zorder=4, multialignment="center")

    # Needle
    needle_angle = np.pi - (min(mae_value, mae_max) / mae_max) * np.pi
    nx = 0.72 * np.cos(needle_angle)
    ny = 0.72 * np.sin(needle_angle)
    color = DANGER if mae_value > 70 else (CAUTION if mae_value > 55 else (INFO if mae_value > 28 else SUCCESS))
    ax.annotate("", xy=(nx, ny), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=2.2, mutation_scale=14),
                zorder=6)
    ax.plot(0, 0, "o", color=FG, ms=5, zorder=7)

    # Centre text
    ax.text(0, -0.25, f"${mae_value:.0f}k MAE", ha="center", va="center",
            color=color, fontsize=11, fontweight="bold", zorder=5)
    ax.text(0, -0.40, phase, ha="center", va="center",
            color=MUTED, fontsize=7.5, zorder=5)

    # Progress bar at bottom
    progress = frame_idx / max(n_frames - 1, 1)
    bar_y = -0.68
    ax.plot([-0.85, 0.85], [bar_y, bar_y], color=MUTED, lw=2, alpha=0.3, zorder=4)
    ax.plot([-0.85, -0.85 + 1.70 * progress], [bar_y, bar_y],
            color=INFO, lw=2, alpha=0.8, zorder=5)

    # Title
    ax.text(0, 1.05, "UnifiedAI — MAE Progress", ha="center", va="center",
            color=FG, fontsize=9, fontweight="bold", zorder=5)
    ax.text(0, 0.91, "Ch.2: Architecture Blueprint", ha="center", va="center",
            color=MUTED, fontsize=7, zorder=5)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.85, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")


def make_needle_gif():
    # Keyframe sequence: (mae_value_$k, phase_label, hold_frames)
    keyframes = [
        (70.0, "Linear regression baseline", 4),
        (68.5, "Adding hidden layers...", 1),
        (65.0, "8 features + ReLU activation", 2),
        (60.0, "He init + architecture set", 2),
        (57.0, "Forward pass verified", 2),
        (55.0, "Architecture ready (~$55k potential)", 6),
    ]

    # Interpolate between keyframes
    frames_data = []
    for i in range(len(keyframes) - 1):
        mae_start, phase_start, hold_start = keyframes[i]
        mae_end,   phase_end,   _          = keyframes[i + 1]
        n_interp = 12
        for j in range(n_interp):
            t = j / n_interp
            mae = mae_start + (mae_end - mae_start) * (3 * t**2 - 2 * t**3)
            frames_data.append((mae, phase_start))
        for _ in range(hold_start):
            frames_data.append((mae_start, phase_start))
    # Final hold
    mae_end, phase_end, hold_end = keyframes[-1]
    for _ in range(hold_end):
        frames_data.append((mae_end, phase_end))

    n_frames = len(frames_data)
    pil_frames = []
    for idx, (mae_val, phase) in enumerate(frames_data):
        fig, ax = plt.subplots(figsize=(4.2, 3.0), facecolor=BG)
        _needle_frame(ax, mae_val, phase, idx, n_frames)
        fig.tight_layout(pad=0.3)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        pil_frames.append(Image.fromarray(buf[:, :, :3]))
        plt.close(fig)

    out_path = OUT_DIR / "ch02-neural-networks-needle.gif"
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        optimize=False,
        duration=120,
        loop=0,
    )
    print(f"Saved: {out_path}  ({len(pil_frames)} frames)")


# ─────────────────────────────────────────────────────────────────────────────
# 2. PROGRESS CHECK PNG
# ─────────────────────────────────────────────────────────────────────────────

def make_progress_check():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), facecolor=BG)
    fig.suptitle("Ch.2 — Neural Networks: Progress Check",
                 color=FG, fontsize=13, fontweight="bold", y=0.97)

    # ── LEFT PANEL: MAE Progress bar chart ──────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(BG)

    milestones = [
        ("Linear baseline\n(Ch.1 Regression)",   70, DANGER),
        ("Architecture\nready (Ch.2)",            55, CAUTION),
        ("After training\n(Ch.3 target)",         40, INFO),
        ("UnifiedAI goal\n(final target)",        28, SUCCESS),
    ]
    labels = [m[0] for m in milestones]
    values = [m[1] for m in milestones]
    colors = [m[2] for m in milestones]

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=colors, alpha=0.85, height=0.55,
                   edgecolor=BG, linewidth=1.5)

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 1.2, bar.get_y() + bar.get_height() / 2,
                f"${val}k MAE", va="center", ha="left",
                color=FG, fontsize=8.5, fontweight="bold")

    # UnifiedAI target line
    ax.axvline(28, color=SUCCESS, lw=1.5, ls="--", alpha=0.7, label="≤$28k target")
    ax.text(28.5, len(labels) - 0.3, "≤$28k\ntarget", color=SUCCESS,
            fontsize=7, va="top")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, color=FG, fontsize=8)
    ax.set_xlabel("MAE ($k)", color=MUTED, fontsize=9)
    ax.set_xlim(0, 85)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.spines[:].set_color(MUTED)
    ax.spines[:].set_alpha(0.3)
    ax.set_title("MAE Reduction Trajectory", color=FG, fontsize=10,
                 fontweight="bold", pad=8)

    # "You are here" marker
    ax.annotate("← You are\n   here (Ch.2)", xy=(55, 1),
                xytext=(62, 1.5),
                arrowprops=dict(arrowstyle="->", color=CAUTION, lw=1.2),
                color=CAUTION, fontsize=7.5)

    # ── RIGHT PANEL: Constraint status ──────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(BG)
    ax2.axis("off")

    ax2.text(0.5, 0.97, "UnifiedAI Constraints — Ch.2 Status",
             ha="center", va="top", color=FG, fontsize=10,
             fontweight="bold", transform=ax2.transAxes)

    constraints = [
        ("#1 ACCURACY",        "<=28k MAE",    "WIP", "Architecture ready,\n~$55k possible", CAUTION),
        ("#2 GENERALIZATION",  "Unseen districts", "NO", "Regularisation\nneeded (Ch.4)",  DANGER),
        ("#3 MULTI-TASK",      "Dual output head", "WIP", "Arch supports it;\nhead not swapped", CAUTION),
        ("#4 INTERPRETABILITY","Explainable",   "NO", "Attention\nneeded (Ch.10)",         DANGER),
        ("#5 PRODUCTION",      "<100ms infer.", "WIP", "2,689 params -> fast;\npipeline needed", CAUTION),
    ]

    row_h = 0.155
    for i, (name, target, status, note, col) in enumerate(constraints):
        y = 0.88 - i * row_h
        # Status badge
        badge_col = SUCCESS if status == "OK" else (CAUTION if status == "WIP" else DANGER)
        rect = mpatches.FancyBboxPatch((0.02, y - 0.06), 0.08, 0.10,
                                       boxstyle="round,pad=0.01",
                                       facecolor=badge_col, edgecolor="none",
                                       transform=ax2.transAxes, zorder=3)
        ax2.add_patch(rect)
        ax2.text(0.06, y - 0.01, status, ha="center", va="center",
                 color=FG, fontsize=11, transform=ax2.transAxes, zorder=4)

        # Constraint name
        ax2.text(0.13, y + 0.03, name, ha="left", va="center",
                 color=FG, fontsize=8.5, fontweight="bold",
                 transform=ax2.transAxes)
        ax2.text(0.13, y - 0.025, target, ha="left", va="center",
                 color=MUTED, fontsize=7.5, transform=ax2.transAxes)
        # Note
        ax2.text(0.62, y - 0.005, note, ha="left", va="center",
                 color=col, fontsize=7.5, transform=ax2.transAxes,
                 multialignment="left")

    # Legend
    ax2.text(0.5, 0.05, "OK=Achieved  WIP=In progress  NO=Blocked",
             ha="center", va="center", color=MUTED, fontsize=7.5,
             transform=ax2.transAxes)

    # Parameter count callout
    ax2.text(0.5, 0.135,
             "Architecture: 8 → 64 (ReLU) → 32 (ReLU) → 1 (linear)   |   2,689 parameters",
             ha="center", va="center", color=FG, fontsize=7.5,
             transform=ax2.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor=PRIMARY,
                       edgecolor=INFO, alpha=0.7))

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = OUT_DIR / "ch02-neural-networks-progress-check.png"
    fig.savefig(out_path, dpi=140, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating Ch.2 Neural Networks visuals...")
    make_needle_gif()
    make_progress_check()
    print("Done.")
