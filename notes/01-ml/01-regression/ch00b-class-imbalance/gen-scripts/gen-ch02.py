"""
gen_ch02.py — Generate visualizations for Ch.2 Class Imbalance.

Deterministic (seed=42). Run from the chapter root directory:

    python gen_scripts/gen_ch02.py

Outputs:
    img/ch02-class-imbalance-needle.gif        — gauge-style needle animation
                                                 showing minority-class recall
                                                 improving across four stages
    img/ch02-class-imbalance-progress-check.png — five-constraint dashboard
                                                   (RealtyML status after Ch.2)

Both outputs use a dark background (#1a1a2e) matching the track palette.
"""

import matplotlib
matplotlib.use("Agg")  # must be set before importing pyplot

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import numpy as np
from pathlib import Path

# ── Palette (canonical track colours) ────────────────────────────────────────
DARK_BG  = "#1a1a2e"   # background for all plots
PRIMARY  = "#1e3a8a"   # deep blue
SUCCESS  = "#15803d"   # green  (constraint achieved)
CAUTION  = "#b45309"   # amber  (partial / in-progress)
DANGER   = "#b91c1c"   # red    (blocked / failing)
INFO     = "#1d4ed8"   # bright blue
WHITE    = "#e8e8f0"   # near-white text
GREY     = "#6b7280"   # muted label text

# ── Output directory (img/ relative to this script's parent) ─────────────────
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "img"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Global RNG ────────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)


# =============================================================================
# 1. Needle Animation
# =============================================================================

def gen_needle_animation() -> None:
    """
    Gauge-style animation: a needle sweeps from ~20% to ~71% recall across
    four rebalancing stages.  Saved as an animated GIF.
    """
    # Four progressive rebalancing stages (label, target recall)
    stages = [
        ("Baseline\n(no rebalancing)",   0.200),
        ("Class Weights\n(balanced)",    0.460),
        ("SMOTE\n(k_neighbors=5)",       0.670),
        ("SMOTE + Threshold\n(τ=0.38)", 0.710),
    ]

    FPS           = 20
    HOLD_FRAMES   = 20   # frames to hold on each final stage value
    INTERP_FRAMES = 30   # frames to animate between stages

    # Build the per-frame sequence: (recall_value, stage_label)
    frame_data = []
    for i, (label, recall) in enumerate(stages):
        if i == 0:
            for _ in range(HOLD_FRAMES):
                frame_data.append((recall, label))
        else:
            prev_recall = stages[i - 1][1]
            for t in np.linspace(0, 1, INTERP_FRAMES):
                frame_data.append((prev_recall + t * (recall - prev_recall), label))
            for _ in range(HOLD_FRAMES):
                frame_data.append((recall, label))

    # ── Figure setup ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-0.35, 1.55)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(
        0, 1.48,
        "Minority-Class Recall — Rebalancing Effect",
        color=WHITE, fontsize=10, fontweight="bold",
        ha="center", va="top",
    )

    # ── Draw the gauge arc ────────────────────────────────────────────────────
    arc_theta = np.linspace(np.pi, 0, 300)
    ax.plot(np.cos(arc_theta), np.sin(arc_theta), color=GREY, lw=2.5, zorder=2)

    # Colour zones: red 0–22%, amber 22–50%, green 50–100%
    zone_defs = [
        (np.pi,         np.deg2rad(140), DANGER),
        (np.deg2rad(140), np.deg2rad(90), CAUTION),
        (np.deg2rad(90),  0,              SUCCESS),
    ]
    for t_from, t_to, colour in zone_defs:
        theta_zone = np.linspace(t_from, t_to, 120)
        ax.plot(
            np.cos(theta_zone), np.sin(theta_zone),
            color=colour, lw=11, alpha=0.30, zorder=1,
        )

    # Tick marks and percentage labels at 0, 20, 40, 60, 80, 100%
    for rv in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        angle = np.pi * (1 - rv)   # 180° → 0° as recall 0 → 1
        ax.plot(
            [0.90 * np.cos(angle), 1.05 * np.cos(angle)],
            [0.90 * np.sin(angle), 1.05 * np.sin(angle)],
            color=WHITE, lw=1.5, zorder=3,
        )
        ax.text(
            1.22 * np.cos(angle), 1.22 * np.sin(angle),
            f"{int(rv * 100)}%",
            color=WHITE, fontsize=7.5, ha="center", va="center", zorder=4,
        )

    # ── Needle components ─────────────────────────────────────────────────────
    needle_line, = ax.plot([], [], lw=3.5, zorder=6, solid_capstyle="round")
    needle_base  = plt.Circle((0, 0), 0.07, color=WHITE, zorder=7)
    ax.add_patch(needle_base)

    # Dynamic text elements
    recall_text = ax.text(
        0, 0.38, "",
        color=WHITE, fontsize=22, fontweight="bold",
        ha="center", va="center", zorder=8,
    )
    stage_text = ax.text(
        0, -0.18, "",
        color=WHITE, fontsize=8.5, ha="center", va="top", zorder=8,
        bbox=dict(facecolor=PRIMARY, edgecolor="none", pad=5, alpha=0.85),
    )

    # Legend patches
    legend_patches = [
        mpatches.Patch(color=DANGER,  label="Failing (<22%)"),
        mpatches.Patch(color=CAUTION, label="Improving (22–50%)"),
        mpatches.Patch(color=SUCCESS, label="Good (≥50%)"),
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower center",
        fontsize=7.5,
        frameon=True,
        facecolor=DARK_BG,
        edgecolor=GREY,
        labelcolor=WHITE,
        ncol=3,
        bbox_to_anchor=(0.5, -0.32),
    )

    # ── Animation update function ─────────────────────────────────────────────
    def update(frame_idx: int):
        rv, label = frame_data[frame_idx]
        angle = np.pi * (1 - rv)
        nx = 0.82 * np.cos(angle)
        ny = 0.82 * np.sin(angle)
        needle_line.set_data([0, nx], [0, ny])

        colour = DANGER if rv < 0.22 else (CAUTION if rv < 0.50 else SUCCESS)
        needle_line.set_color(colour)
        needle_base.set_facecolor(colour)
        recall_text.set_text(f"{rv:.0%}")
        recall_text.set_color(colour)
        stage_text.set_text(label)
        return needle_line, needle_base, recall_text, stage_text

    ani = animation.FuncAnimation(
        fig, update,
        frames=len(frame_data),
        interval=1000 // FPS,
        blit=True,
    )

    out_path = OUTPUT_DIR / "ch02-class-imbalance-needle.gif"
    ani.save(str(out_path), writer="pillow", fps=FPS)
    plt.close(fig)
    print(f"  ✅  {out_path}")


# =============================================================================
# 2. Progress-Check Dashboard
# =============================================================================

def gen_progress_check() -> None:
    """
    Five-constraint status dashboard for RealtyML after Ch.2.
    Dark background; each constraint row colour-coded by status.
    """
    fig, ax = plt.subplots(figsize=(10, 5.8), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.8)

    # ── Header ────────────────────────────────────────────────────────────────
    ax.text(
        5, 6.55,
        "RealtyML — Constraint Dashboard",
        color=WHITE, fontsize=14, fontweight="bold", ha="center", va="center",
    )
    ax.text(
        5, 6.18,
        "After Ch.2: Handling Class Imbalance  ·  Goal: Portland MAE 174k → <95k",
        color=GREY, fontsize=9, ha="center", va="center",
    )

    # ── Constraint rows ───────────────────────────────────────────────────────
    # (label, target, status_text, colour, detail_note)
    constraints = [
        (
            "#1  ACCURACY",
            "MAE < $95k",
            "🟡 Partial",
            CAUTION,
            "High-value MAE 287k → ~144k   (49% ↓)   — overall gap remains",
        ),
        (
            "#2  GENERALIZATION",
            "CA → Portland",
            "🟡 Partial",
            CAUTION,
            "Better minority coverage; covariate shift unaddressed until Ch.3",
        ),
        (
            "#3  DATA QUALITY",
            "Match production dist.",
            "✅  Unlocked",
            SUCCESS,
            "SMOTE: 75/25 → 50/50  ·  stratify=y throughout  ·  seed=42",
        ),
        (
            "#4  AUDITABILITY",
            "Reproducible pipeline",
            "🟡 Partial",
            CAUTION,
            "seed=42 and stratify=y documented; full provenance log pending",
        ),
        (
            "#5  PRODUCTION-READY",
            "No data leakage",
            "🟡 Partial",
            CAUTION,
            "SMOTE-after-split pattern established; monitoring not yet built",
        ),
    ]

    ROW_HEIGHT = 0.88
    Y_START    = 5.55

    for i, (label, target, status, colour, note) in enumerate(constraints):
        y = Y_START - i * ROW_HEIGHT

        # Background band
        band = mpatches.FancyBboxPatch(
            (0.25, y - 0.36), 9.5, 0.74,
            boxstyle="round,pad=0.04",
            facecolor=colour, alpha=0.13,
            edgecolor=colour, linewidth=1.4,
        )
        ax.add_patch(band)

        # Status circle
        ax.add_patch(plt.Circle((0.72, y), 0.20, color=colour, zorder=3))

        # Constraint label and target
        ax.text(
            1.10, y + 0.14, label,
            color=WHITE, fontsize=9, fontweight="bold", va="center",
        )
        ax.text(
            1.10, y - 0.13, target,
            color=GREY, fontsize=8, va="center",
        )

        # Status badge (centred)
        ax.text(
            6.1, y + 0.14, status,
            color=colour, fontsize=9, fontweight="bold",
            ha="center", va="center",
        )
        ax.text(
            6.1, y - 0.13, note,
            color=WHITE, fontsize=7.2,
            ha="center", va="center", alpha=0.88,
        )

    # ── Footer ────────────────────────────────────────────────────────────────
    ax.text(
        5, 0.28,
        "Ch.2 unlocks Constraint #3 DATA QUALITY  ·  "
        "Next: Ch.3 Data Drift → Constraint #2 GENERALIZATION",
        color=GREY, fontsize=7.8, ha="center", va="center", style="italic",
    )

    out_path = OUTPUT_DIR / "ch02-class-imbalance-progress-check.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  ✅  {out_path}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    print("Generating Ch.2 Class Imbalance visuals …")
    gen_needle_animation()
    gen_progress_check()
    print("\nDone. Output in notes/01-ml/00_data_fundamentals/ch02_class_imbalance/img/")
