"""
gen_ch04.py — Generate visual assets for Ch.4 Neural Collaborative Filtering
Produces:
  img/ch04-neural-cf-needle.gif        — HR@10 gauge: MF 78% → NCF 82%
  img/ch04-neural-cf-progress-check.png — FlixAI constraint dashboard after Ch.4
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
# 1. NEEDLE GIF  — HR@10 gauge: MF 78% → NCF 82%
# ─────────────────────────────────────────────────────────────────────────────

def _needle_frame(ax, hr_value: float, phase: str, frame_idx: int, n_frames: int):
    """Draw one HR@10 speedometer frame."""
    ax.clear()
    ax.set_facecolor(BG)

    hr_min, hr_max = 0.0, 100.0

    # Zone boundaries (HR@10 %)
    zone_bounds = [0, 42, 65, 78, 82, 85, 100]
    zone_colors = [DANGER, CAUTION, INFO, CAUTION, INFO, SUCCESS]
    zone_labels = [
        "0%",
        "42%\n(pop.)",
        "65%\n(CF)",
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
    """Animate HR@10 rising from MF baseline (78%) to NCF result (82%)."""
    # Keyframes: (hr_value_%, phase_label, hold_frames)
    keyframes = [
        (78.0, "Ch.3 Matrix Factorisation\nHR@10 = 78%",      6),
        (78.5, "Replace dot product\nwith NeuMF...",           1),
        (79.2, "Embedding tables\n(GMF + MLP separate)",       1),
        (79.8, "GMF path: ⊙ element-wise\nproduct signal",     1),
        (80.3, "MLP path: ReLU layers\nlearn interactions",    1),
        (80.8, "Fuse GMF + MLP\nvia sigmoid",                  1),
        (81.2, "BCE + negative sampling\noptimising...",       1),
        (81.6, "Epoch 10 — val HR rising",                     1),
        (82.0, "Neural CF  (NeuMF)\nHR@10 ≈ 82%",             8),
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

    out_path = OUT_DIR / "ch04-neural-cf-needle.gif"
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
        "Ch.4 — Neural Collaborative Filtering: Progress Check",
        color=FG, fontsize=13, fontweight="bold", y=0.97,
    )

    # ── LEFT PANEL: HR@10 progress bar chart ────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(BG)

    milestones = [
        ("Popularity baseline\n(Ch.1)",             42,  DANGER),
        ("Item-CF\n(Ch.2)",                         65,  CAUTION),
        ("Matrix Factorisation\n(Ch.3)",            78,  INFO),
        ("Neural CF  NeuMF\n(this chapter)",        82,  INFO),
        ("FlixAI target",                           85,  SUCCESS),
    ]
    labels = [m[0] for m in milestones]
    values = [m[1] for m in milestones]
    colors = [m[2] for m in milestones]

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=colors, alpha=0.85, height=0.55,
                   edgecolor=BG, linewidth=1.5)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
            f"{val}%", va="center", ha="left",
            color=FG, fontsize=9, fontweight="bold",
        )

    # Target line
    ax.axvline(85, color=SUCCESS, lw=1.5, ls="--", alpha=0.8)
    ax.text(85.5, len(labels) - 0.4, ">85%\ntarget", color=SUCCESS, fontsize=7, va="top")

    # "You are here" arrow
    ax.annotate(
        "← Ch.4 result", xy=(82, 3), xytext=(70, 3.6),
        arrowprops=dict(arrowstyle="->", color=INFO, lw=1.2),
        color=INFO, fontsize=7.5,
    )

    # Gap callout
    ax.annotate(
        "", xy=(85, 0.5), xytext=(82, 0.5),
        arrowprops=dict(arrowstyle="<->", color=GOLD, lw=1.5),
    )
    ax.text(83.5, 0.7, "3%\ngap", ha="center", va="bottom", color=GOLD, fontsize=7.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, color=FG, fontsize=8)
    ax.set_xlabel("HR@10 (%)", color=MUTED, fontsize=9)
    ax.set_xlim(0, 100)
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(MUTED)
        sp.set_alpha(0.3)
    ax.set_title("Hit Rate @ Top-10 Progress", color=FG, fontsize=10,
                 fontweight="bold", pad=8)

    # ── RIGHT PANEL: Constraint status ──────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(BG)
    ax2.axis("off")

    ax2.text(
        0.5, 0.97, "FlixAI Constraints — Ch.4 Status",
        ha="center", va="top", color=FG, fontsize=10,
        fontweight="bold", transform=ax2.transAxes,
    )

    constraints = [
        (
            "#1 ACCURACY", ">85% HR@10",
            "WIP", "NeuMF → 82%\n3 pts to go — Ch.5 hybrid closes gap", CAUTION,
        ),
        (
            "#2 COLD START", "New users/items",
            "NO", "Embeddings need history\nNew arrivals → random recs (Ch.5)", DANGER,
        ),
        (
            "#3 SCALABILITY", "<200ms, 1M+ ratings",
            "OK", "O(1) embedding lookup;\nMLP overhead manageable at 100k", SUCCESS,
        ),
        (
            "#4 DIVERSITY", "Long-tail content",
            "WIP", "Richer latent space reduces\npopularity bias", CAUTION,
        ),
        (
            "#5 EXPLAINABILITY", "\"Because you liked X\"",
            "NO", "Deep MLP = black box;\nno direct feature attribution", DANGER,
        ),
    ]

    row_h = 0.155
    for i, (name, target, status, note, col) in enumerate(constraints):
        y = 0.86 - i * row_h
        badge_col = SUCCESS if status == "OK" else (CAUTION if status == "WIP" else DANGER)
        rect = mpatches.FancyBboxPatch(
            (0.02, y - 0.055), 0.08, 0.095,
            boxstyle="round,pad=0.01",
            facecolor=badge_col, edgecolor="none",
            transform=ax2.transAxes, zorder=3,
        )
        ax2.add_patch(rect)
        ax2.text(0.06, y - 0.005, status, ha="center", va="center",
                 color=FG, fontsize=9, fontweight="bold",
                 transform=ax2.transAxes, zorder=4)
        ax2.text(0.13, y + 0.025, name, ha="left", va="center",
                 color=FG, fontsize=8.5, fontweight="bold",
                 transform=ax2.transAxes)
        ax2.text(0.13, y - 0.025, target, ha="left", va="center",
                 color=MUTED, fontsize=7.5, transform=ax2.transAxes)
        ax2.text(0.62, y - 0.005, note, ha="left", va="center",
                 color=col, fontsize=7.5, transform=ax2.transAxes,
                 multialignment="left")

    ax2.text(
        0.5, 0.05,
        "OK = Achieved   WIP = In progress   NO = Blocked",
        ha="center", va="center", color=MUTED, fontsize=7.5,
        transform=ax2.transAxes,
    )

    # Model summary callout
    ax2.text(
        0.5, 0.135,
        "Model: NeuMF  |  d=16 GMF + d=16 MLP  |  MLP: 32→16→8→1  |  k=4 negatives",
        ha="center", va="center", color=FG, fontsize=7.5,
        transform=ax2.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=PRIMARY, edgecolor=INFO, alpha=0.7),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = OUT_DIR / "ch04-neural-cf-progress-check.png"
    fig.savefig(out_path, dpi=140, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating Ch.4 Neural Collaborative Filtering visuals...")
    make_needle_gif()
    make_progress_check()
    print("Done.")
