"""
gen_ch08.py — Generate visual assets for Ch.8 TensorBoard.

Produces:
    ../img/ch08-tensorboard-needle.gif   — Constraint #5 production-readiness meter
    ../img/ch08-tensorboard-progress-check.png — static progress-check summary

Design:
    seed=42, dark background #1a1a2e, accent palette matching authoring-guide.md
    colour palette: primary #1e3a8a, success #15803d, caution #b45309, danger #b91c1c
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
HERE    = Path(__file__).resolve().parent            # gen_scripts/
IMG_DIR = HERE.parent / "img"                        # ch08_tensorboard/img/
IMG_DIR.mkdir(parents=True, exist_ok=True)

ROOT = Path(__file__).resolve().parents[5]           # repo root
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# ── Attempt to use the shared animation renderer if available ─────────────────
try:
    from shared.animation_renderer import render_metric_story  # type: ignore

    STAGES = [
        {
            "label": "No monitoring",
            "value": 78.0,
            "curve": [0.46, 0.12, 0.88, 0.28],
            "caption": (
                "Training blind: the model memorises training data long after "
                "the optimal generalisation point — wasted compute, wrong final weights."
            ),
        },
        {
            "label": "Loss curves",
            "value": 64.0,
            "curve": [0.43, 0.16, 0.90, 0.16],
            "caption": (
                "Scalar logging reveals the exact epoch when the validation gap opens — "
                "overfitting detected 25 epochs early."
            ),
        },
        {
            "label": "Early stopping",
            "value": 56.0,
            "curve": [0.40, 0.19, 0.91, 0.07],
            "caption": (
                "Halting at the best validation checkpoint recovers the optimal weights "
                "before late-epoch memorisation degrades them."
            ),
        },
        {
            "label": "LR schedule tuned",
            "value": 51.0,
            "curve": [0.39, 0.21, 0.92, 0.03],
            "caption": (
                "Scheduler visualisations confirm the best decay strategy, "
                "squeezing out the final MAE improvement."
            ),
        },
    ]

    render_metric_story(
        HERE,
        "ch08-tensorboard-needle",
        "Ch.8 — TensorBoard monitoring moves the validation metric in the right direction",
        "Test MAE ($k)",
        STAGES,
        better="lower",
        style="regression",
    )
    print("needle.gif written via shared renderer.")

except Exception:
    # ── Standalone fallback — pure matplotlib / pillow ────────────────────────
    _generate_standalone()


def _generate_standalone() -> None:
    """Fallback: generate both assets with raw matplotlib + Pillow."""
    import math
    import io
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.animation import FuncAnimation, PillowWriter

    SEED = 42
    rng  = np.random.default_rng(SEED)

    BG       = "#1a1a2e"
    PRIMARY  = "#1e3a8a"
    SUCCESS  = "#15803d"
    CAUTION  = "#b45309"
    DANGER   = "#b91c1c"
    WHITE    = "#e2e8f0"
    ACCENT   = "#3b82f6"

    # ── Stage definitions (label, MAE in $k, caption) ──────────────────────
    STAGES = [
        ("No monitoring",   78.0, "Training blind: model memorises past the optimal checkpoint."),
        ("Loss curves",     64.0, "Scalar logging catches overfitting 25 epochs early."),
        ("Early stopping",  56.0, "Checkpoint at best val loss — optimal weights recovered."),
        ("LR tuned",        51.0, "Scheduler dial confirmed; final MAE improvement squeezed out."),
    ]
    MAE_MAX  = 90.0
    MAE_MIN  = 40.0
    N_STAGES = len(STAGES)
    FPS      = 12
    HOLD     = FPS * 2      # frames to hold each stage
    TRANS    = FPS           # frames to animate needle between stages

    # ── Needle angle helpers ─────────────────────────────────────────────────
    ANGLE_MIN = 220.0   # degrees (left = bad)
    ANGLE_MAX = -40.0   # degrees (right = good)

    def mae_to_angle(mae: float) -> float:
        frac = (MAE_MAX - mae) / (MAE_MAX - MAE_MIN)
        frac = max(0.0, min(1.0, frac))
        return ANGLE_MIN + frac * (ANGLE_MAX - ANGLE_MIN)

    def angle_to_xy(angle_deg: float, r: float = 0.7):
        a = math.radians(angle_deg)
        return math.cos(a), math.sin(a)

    # ── Build frame list ─────────────────────────────────────────────────────
    # Each entry: (stage_idx, interp_t [0..1])
    frames: list[tuple[int, float]] = []
    for si in range(N_STAGES):
        if si == 0:
            frames.extend([(0, 0.0)] * HOLD)
        else:
            for fi in range(TRANS):
                frames.append((si, fi / TRANS))
            frames.extend([(si, 1.0)] * HOLD)

    # ── Animation ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(9, 5), facecolor=BG)
    ax  = fig.add_subplot(111, facecolor=BG)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.35, 1.15)
    ax.set_aspect('equal')
    ax.axis('off')

    # Gauge arc background
    theta = np.linspace(math.radians(ANGLE_MIN), math.radians(ANGLE_MAX), 300)
    ax.plot(np.cos(theta), np.sin(theta), color="#334155", lw=18, solid_capstyle='round', zorder=1)

    # Coloured arc segments (danger → caution → success)
    for t_start, t_end, col in [
        (0.00, 0.33, DANGER),
        (0.33, 0.66, CAUTION),
        (0.66, 1.00, SUCCESS),
    ]:
        a0 = ANGLE_MIN + t_start * (ANGLE_MAX - ANGLE_MIN)
        a1 = ANGLE_MIN + t_end   * (ANGLE_MAX - ANGLE_MIN)
        seg = np.linspace(math.radians(a0), math.radians(a1), 100)
        ax.plot(np.cos(seg), np.sin(seg), color=col, lw=14,
                solid_capstyle='round', zorder=2, alpha=0.45)

    # Stage tick marks
    tick_angles = [mae_to_angle(s[1]) for s in STAGES]
    for ta in tick_angles:
        tx, ty = angle_to_xy(ta, 0.82)
        ax.plot(tx, ty, 'o', ms=7, color=WHITE, zorder=5)

    # Needle line
    start_angle = mae_to_angle(STAGES[0][1])
    sx, sy      = angle_to_xy(start_angle, 0.68)
    needle_line, = ax.plot([0, sx], [0, sy], color=WHITE, lw=3, zorder=10, solid_capstyle='round')
    needle_hub   = ax.plot(0, 0, 'o', ms=10, color=WHITE, zorder=11)[0]

    # Text elements
    title_txt   = ax.text(0, 1.10, "Ch.8 — TensorBoard Monitoring",
                          ha='center', va='top', color=WHITE,
                          fontsize=11, fontweight='bold', zorder=12)
    stage_txt   = ax.text(0, -0.08, STAGES[0][0],
                          ha='center', va='top', color=ACCENT,
                          fontsize=10, fontweight='bold', zorder=12)
    mae_txt     = ax.text(0, -0.20, f"Val MAE: {STAGES[0][1]:.0f}k",
                          ha='center', va='top', color=WHITE,
                          fontsize=9, zorder=12)
    caption_txt = ax.text(0, -0.30, STAGES[0][2],
                          ha='center', va='top', color="#94a3b8",
                          fontsize=7.5, wrap=True, zorder=12,
                          multialignment='center')

    def _update(frame_idx: int):
        si, t = frames[frame_idx]

        if si == 0 or t == 0.0:
            cur_angle = mae_to_angle(STAGES[si][1])
        else:
            prev_a = mae_to_angle(STAGES[si - 1][1])
            next_a = mae_to_angle(STAGES[si][1])
            # ease-in-out cubic interpolation
            t_ease = t * t * (3 - 2 * t)
            cur_angle = prev_a + t_ease * (next_a - prev_a)

        nx, ny = angle_to_xy(cur_angle, 0.68)
        needle_line.set_data([0, nx], [0, ny])

        if t >= 1.0 or si == 0:
            label, mae, caption = STAGES[si]
            stage_txt.set_text(label)
            mae_txt.set_text(f"Val MAE: {mae:.0f}k")
            caption_txt.set_text(caption)

        return needle_line, stage_txt, mae_txt, caption_txt

    ani = FuncAnimation(fig, _update, frames=len(frames), interval=1000 // FPS, blit=True)
    out_gif = IMG_DIR / "ch08-tensorboard-needle.gif"
    ani.save(str(out_gif), writer=PillowWriter(fps=FPS), dpi=120)
    plt.close(fig)
    print(f"Written: {out_gif}")

    # ── Progress-check static PNG ────────────────────────────────────────────
    _generate_progress_check(BG, PRIMARY, SUCCESS, CAUTION, DANGER, WHITE)


def _generate_progress_check(BG, PRIMARY, SUCCESS, CAUTION, DANGER, WHITE):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    ROWS = [
        # (label, status, colour)
        ("Loss scalars (train/val)",           "✅ Done",    SUCCESS),
        ("Weight/gradient histograms",         "✅ Done",    SUCCESS),
        ("Embedding projector (layer3)",       "✅ Done",    SUCCESS),
        ("HParam sweep logging",               "✅ Done",    SUCCESS),
        ("Model checkpointing (best val loss)","✅ Done",    SUCCESS),
        ("Serving latency monitoring",         "❌ Not yet", DANGER),
        ("Model versioning / registry",        "❌ Not yet", DANGER),
        ("A/B testing infrastructure",         "❌ Not yet", DANGER),
        ("Prediction drift detection",         "❌ Not yet", DANGER),
    ]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis('off')

    # Title
    ax.text(0.5, 0.97, "Ch.8 — TensorBoard: Constraint #5 PRODUCTION Status",
            ha='center', va='top', transform=ax.transAxes,
            color=WHITE, fontsize=12, fontweight='bold')
    ax.text(0.5, 0.90, "Monitoring instrumented ✅   |   Serving / registry ❌ (Production track)",
            ha='center', va='top', transform=ax.transAxes,
            color="#94a3b8", fontsize=9)

    ROW_H = 0.072
    TOP   = 0.84

    for i, (label, status, col) in enumerate(ROWS):
        y = TOP - i * ROW_H
        # Row background pill
        rect = mpatches.FancyBboxPatch(
            (0.04, y - 0.025), 0.92, ROW_H * 0.85,
            boxstyle="round,pad=0.01",
            facecolor="#0f172a", edgecolor=col, linewidth=1.2,
            transform=ax.transAxes, clip_on=False, zorder=2
        )
        ax.add_patch(rect)
        ax.text(0.08, y + 0.005, label, ha='left', va='center',
                transform=ax.transAxes, color=WHITE, fontsize=8.5, zorder=3)
        ax.text(0.90, y + 0.005, status, ha='right', va='center',
                transform=ax.transAxes, color=col, fontsize=8.5,
                fontweight='bold', zorder=3)

    out_png = IMG_DIR / "ch08-tensorboard-progress-check.png"
    fig.savefig(str(out_png), dpi=150, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close(fig)
    print(f"Written: {out_png}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # If the shared renderer was imported successfully the gif is already written.
    # Generate the progress-check PNG independently (not produced by render_metric_story).
    try:
        import matplotlib
        BG      = "#1a1a2e"
        PRIMARY = "#1e3a8a"
        SUCCESS = "#15803d"
        CAUTION = "#b45309"
        DANGER  = "#b91c1c"
        WHITE   = "#e2e8f0"
        _generate_progress_check(BG, PRIMARY, SUCCESS, CAUTION, DANGER, WHITE)
    except Exception as exc:
        print(f"Progress-check generation failed: {exc}")
