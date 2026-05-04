"""Generate Ch.3 animations.

Outputs (saved next to this script):
    ch03-feature-importance-needle.gif   — chapter-level needle animation
    feature-scaling-gradient.gif         — gradient descent paths: unscaled vs scaled

Run:  python gen_ch03.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.animation import PillowWriter

HERE = Path(__file__).parent
ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.animation_renderer import render_metric_story

# ── Palette ────────────────────────────────────────────────────────────────
DARK       = "#1f2937"
BLUE       = "#2563eb"
GREEN      = "#16a34a"
AMBER      = "#d97706"
RED        = "#dc2626"
GREY       = "#cbd5e1"
LIGHT      = "#f8fafc"
ORANGE     = "#ea580c"


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Chapter needle animation
# ══════════════════════════════════════════════════════════════════════════════
STAGES = [
    {
        "label": "Raw weights (unscaled)",
        "value": 55.0,
        "curve": [0.18, 0.62, 0.82, 0.0],
        "caption": "Without standardization, weight magnitudes reflect input scale—not importance.",
    },
    {
        "label": "Z-score standardization",
        "value": 55.0,
        "curve": [0.30, 0.40, 0.85, 0.0],
        "caption": "Standardize → gradient steps balance across all features.",
    },
    {
        "label": "Univariate R²",
        "value": 54.0,
        "curve": [0.38, 0.22, 0.88, 0.0],
        "caption": "Rank features by standalone variance explained: MedInc dominates.",
    },
    {
        "label": "Standardized weights",
        "value": 53.5,
        "curve": [0.44, 0.14, 0.90, 0.0],
        "caption": "Partial contributions expose the jointly irreplaceable Lat/Lon pair.",
    },
    {
        "label": "Permutation importance",
        "value": 53.0,
        "curve": [0.48, 0.10, 0.91, 0.0],
        "caption": "Scramble each feature → measure MAE rise → most reliable ranking.",
    },
]


def gen_needle() -> None:
    render_metric_story(
        HERE,
        "ch03-feature-importance-needle",
        "Ch.3 — Feature Scaling, Importance & Multicollinearity",
        "MAE ($k)",
        STAGES,
        better="lower",
        style="regression",
    )
    print("✓  ch03-feature-importance-needle.gif")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Feature-scaling gradient descent animation
#
#   Two-panel story:
#     LEFT  — unscaled features (x₁ in [0,15], x₂ in [0,35000]).
#             The loss contours are extremely elongated ellipses.
#             Gradient descent zig-zags / oscillates down the ravine.
#     RIGHT — standardized features (both ~N(0,1)).
#             Contours are near-circular; gradient descent walks straight in.
#
#   Narrative printed as a text score below both panels.
# ══════════════════════════════════════════════════════════════════════════════
np.random.seed(42)

def _quadratic_loss(w1, w2, A, B):
    """Loss = A·w1² + B·w2² (centred at origin for simplicity)."""
    return A * w1 ** 2 + B * w2 ** 2


def _gd_path(A, B, lr, steps=60):
    w = np.array([1.0, 1.0], dtype=float)
    path = [w.copy()]
    for _ in range(steps):
        grad = np.array([2 * A * w[0], 2 * B * w[1]])
        w = w - lr * grad
        path.append(w.copy())
        if np.linalg.norm(w) < 1e-6:
            break
    return np.array(path)


def gen_scaling_animation() -> None:
    # ── Loss landscape parameters ──────────────────────────────────────────
    # Unscaled: one feature has 2000× greater scale → loss contours are
    # very elongated (A≈1, B≈1e-6 in weight space, but when visualised in
    # raw feature space the gradient along B axis dominates).
    A_raw, B_raw     = 1.0, 0.0005   # elongated ellipse in raw space
    A_std, B_std     = 1.0, 0.80     # near-circular in scaled space

    lr_raw = 0.02
    lr_std = 0.35

    path_raw = _gd_path(A_raw, B_raw, lr_raw, steps=80)
    path_std = _gd_path(A_std, B_std, lr_std, steps=25)

    # ── Contour grids ─────────────────────────────────────────────────────
    w1r = np.linspace(-1.2, 1.2, 300)
    w2r = np.linspace(-1.2, 1.2, 300)
    W1, W2 = np.meshgrid(w1r, w2r)
    Z_raw = _quadratic_loss(W1, W2, A_raw, B_raw)
    Z_std = _quadratic_loss(W1, W2, A_std, B_std)

    # ── Figure ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), facecolor="white")
    fig.suptitle(
        "Why Feature Scaling Matters: Gradient Descent Paths",
        fontsize=13, fontweight="bold", color=DARK, y=1.02,
    )

    titles = ["BEFORE scaling\n(unbalanced contours → zig-zag)",
              "AFTER StandardScaler\n(balanced contours → straight path)"]
    Zs     = [Z_raw, Z_std]
    levels = np.linspace(0.005, 1.15, 18)

    for ax, Z, title in zip(axes, Zs, titles):
        ax.contourf(W1, W2, Z, levels=levels, cmap="Blues", alpha=0.55)
        ax.contour (W1, W2, Z, levels=levels, colors="steelblue",
                    alpha=0.45, linewidths=0.7)
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.25, 1.25)
        ax.set_xlabel("w₁  (feature 1 weight)", fontsize=10)
        ax.set_ylabel("w₂  (feature 2 weight)", fontsize=10)
        ax.set_title(title, fontsize=11, color=DARK)
        ax.axhline(0, color=GREY, lw=0.5)
        ax.axvline(0, color=GREY, lw=0.5)
        ax.plot(0, 0, "*", color="gold", ms=14,
                markeredgecolor=DARK, markeredgewidth=0.8,
                zorder=6, label="minimum")

    line_raw, = axes[0].plot([], [], "-", color=RED, lw=1.8, alpha=0.8)
    dot_raw,  = axes[0].plot([], [], "o", color=RED, ms=7, zorder=7)
    line_std, = axes[1].plot([], [], "-", color=GREEN, lw=1.8, alpha=0.8)
    dot_std,  = axes[1].plot([], [], "o", color=GREEN, ms=7, zorder=7)

    epoch_text = axes[0].text(
        -1.18, 1.15, "", fontsize=9, color=DARK, family="monospace",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.8)
    )

    caption_ax = fig.add_axes([0.05, -0.08, 0.90, 0.07])
    caption_ax.axis("off")
    CAPTIONS = [
        "Epoch 0 — Both models start at (1, 1).  Watch how the paths diverge.",
        "Unscaled: gradient is ~2000× larger along w₂ → huge oscillating steps.",
        "Scaled: contours are circular → gradient points directly at minimum.",
        "Unscaled path zig-zags and wastes iterations; needs a tiny learning rate.",
        "Scaled path converges in a fraction of the steps.",
        "Result: always StandardScale before gradient-based training.",
    ]
    caption_txt = caption_ax.text(
        0.5, 0.5, "", ha="center", va="center", fontsize=10.5, color=DARK,
        wrap=True, bbox=dict(boxstyle="round,pad=0.4", fc=LIGHT, ec=GREY)
    )

    n_raw = len(path_raw)
    n_std = len(path_std)
    TOTAL = max(n_raw, n_std) + 16   # extra hold frames at end

    def _cap_idx(frame):
        return min(int(frame / TOTAL * len(CAPTIONS)), len(CAPTIONS) - 1)

    def _init():
        line_raw.set_data([], [])
        dot_raw.set_data([], [])
        line_std.set_data([], [])
        dot_std.set_data([], [])
        epoch_text.set_text("")
        caption_txt.set_text("")
        return line_raw, dot_raw, line_std, dot_std, epoch_text, caption_txt

    def _update(frame):
        ir = min(frame, n_raw - 1)
        ist = min(frame, n_std - 1)

        line_raw.set_data(path_raw[:ir + 1, 0], path_raw[:ir + 1, 1])
        dot_raw.set_data([path_raw[ir, 0]], [path_raw[ir, 1]])

        line_std.set_data(path_std[:ist + 1, 0], path_std[:ist + 1, 1])
        dot_std.set_data([path_std[ist, 0]], [path_std[ist, 1]])

        epoch_text.set_text(f"Step {frame}")
        caption_txt.set_text(CAPTIONS[_cap_idx(frame)])
        return line_raw, dot_raw, line_std, dot_std, epoch_text, caption_txt

    anim = animation.FuncAnimation(
        fig, _update, frames=TOTAL,
        init_func=_init, blit=True, interval=100,
    )
    plt.tight_layout()

    out_path = HERE / "feature-scaling-gradient.gif"
    anim.save(str(out_path), writer=PillowWriter(fps=10))
    plt.close()
    print("✓  feature-scaling-gradient.gif")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    gen_needle()
    gen_scaling_animation()
    print("\nAll ch03 animations generated.")
