"""Generate the 'eyeball explorer' animation for Ch.01 §3.5.

Narrative: shows a regression line being moved through several manual guesses
(as if a person is dragging it) with the MSE readout updating each frame.
The final frame lands at the optimal fit — teaching the intuition that
gradient descent is just automated eyeballing.

Outputs (relative to this script):
    ../img/eyeball_explorer.gif   — animated GIF (~20 frames)
    ../img/eyeball_explorer.png   — final-frame still

Run:
    python gen_eyeball_animation.py
"""
from __future__ import annotations

import os
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── palette ───────────────────────────────────────────────────────────────────
DARK        = "#1f2937"
BLUE        = "#2563eb"
LIGHT_BLUE  = "#93c5fd"
GREEN       = "#16a34a"
AMBER       = "#d97706"
RED         = "#dc2626"
GREY        = "#cbd5e1"
LIGHT       = "#f8fafc"

HERE    = Path(__file__).parent
OUT_DIR = HERE / ".." / "img"
OUT_DIR.mkdir(parents=True, exist_ok=True)
GIF_PATH = OUT_DIR / "eyeball_explorer.gif"
PNG_PATH = OUT_DIR / "eyeball_explorer.png"

# ── data (original-unit subset for visual clarity) ────────────────────────────
data = fetch_california_housing()
X_full = data.data[:, 0]          # MedInc
y_full = data.target               # MedHouseVal (×$100k)

rng = np.random.default_rng(7)
idx = rng.choice(len(X_full), size=500, replace=False)
X_vis = X_full[idx]
y_vis = y_full[idx]

# ── compute optimal w* and b* analytically (OLS on subset) ───────────────────
X_mean, y_mean = X_vis.mean(), y_vis.mean()
w_opt = np.sum((X_vis - X_mean) * (y_vis - y_mean)) / np.sum((X_vis - X_mean) ** 2)
b_opt = y_mean - w_opt * X_mean

# ── loss surface: MSE(w) with b fixed at b* for clarity ──────────────────────
w_grid = np.linspace(-0.1, w_opt * 2.2, 400)
mse_grid = np.array([np.mean((wi * X_vis + b_opt - y_vis) ** 2) for wi in w_grid])

def mse_at(w: float, b: float) -> float:
    return float(np.mean((w * X_vis + b - y_vis) ** 2))


# ── the "guesses" a human would make — a story arc ───────────────────────────
# Each entry: (w, b, caption)
guesses: list[tuple[float, float, str]] = [
    (0.00,  1.00, "First guess:\nflat line through the middle"),
    (0.20,  0.50, "Looks too flat.\nTilt it up a bit..."),
    (0.45,  0.10, "Still off at the low end.\nRaise the slope some more..."),
    (0.65, -0.20, "Too steep now — high-income\ndistricts are over-predicted"),
    (0.50,  0.10, "Back off a little..."),
    (w_opt * 0.85, b_opt * 1.10, "Getting closer.\nErrors look about equal above and below"),
    (w_opt * 0.95, b_opt * 1.02, "Almost there..."),
    (w_opt,        b_opt,        "✓ Optimal fit\nMSE is minimised"),
]

# Duplicate the last frame a few times so the GIF pauses at the solution
guesses += [(w_opt, b_opt, "✓ Optimal fit\nMSE is minimised")] * 4


def draw_frame(w: float, b: float, caption: str) -> np.ndarray:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor="white")
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.13, wspace=0.35)

    # ── Left panel: scatter + current line ───────────────────────────────────
    ax = axes[0]
    ax.scatter(X_vis, y_vis, s=10, alpha=0.35, color=LIGHT_BLUE, zorder=1)

    xs = np.array([X_vis.min(), X_vis.max()])
    ax.plot(xs, w * xs + b, color=AMBER, linewidth=3, zorder=3, label="Your guess")
    ax.plot(xs, w_opt * xs + b_opt, color=GREEN, linewidth=2,
            linestyle="--", zorder=2, alpha=0.7, label="Optimal fit")

    # draw a few residual lines to the current guess
    sample_idx = rng.choice(len(X_vis), size=40, replace=False)
    for si in sample_idx:
        y_hat_i = w * X_vis[si] + b
        ax.plot([X_vis[si], X_vis[si]], [y_vis[si], y_hat_i],
                color=RED, alpha=0.25, linewidth=0.8, zorder=2)

    current_mse = mse_at(w, b)
    opt_mse     = mse_at(w_opt, b_opt)
    ax.set_title(
        f"Trying: w = {w:.3f},  b = {b:.3f}\n"
        f"MSE = {current_mse:.4f}  (optimal = {opt_mse:.4f})",
        fontsize=11, color=DARK, pad=8,
    )
    ax.set_xlabel("MedInc (×$10k)", fontsize=10)
    ax.set_ylabel("MedHouseVal (×$100k)", fontsize=10)
    ax.legend(fontsize=9, frameon=True)
    ax.set_xlim(X_vis.min() - 0.2, X_vis.max() + 0.2)
    ax.set_ylim(-0.1, 5.5)
    ax.grid(alpha=0.18)

    # caption box
    ax.text(
        0.5, -0.14, caption,
        transform=ax.transAxes, ha="center", va="top",
        fontsize=9.5, color=DARK, style="italic",
        bbox=dict(boxstyle="round,pad=0.4", fc=LIGHT, ec=GREY),
    )

    # ── Right panel: MSE(w) parabola with current position ───────────────────
    ax2 = axes[1]
    ax2.plot(w_grid, mse_grid, color=BLUE, linewidth=2.5, label="MSE(w)  [b fixed at b*]")
    ax2.axvline(w_opt, color=GREEN, linewidth=1.5, linestyle="--", alpha=0.7, label=f"w* = {w_opt:.3f}")

    # dot for current guess
    ax2.scatter([w], [mse_at(w, b_opt)], s=140, color=AMBER,
                zorder=5, label=f"Current w = {w:.3f}")

    # gradient arrow at current w (finite-difference direction indicator)
    dw      = 0.001
    grad_w  = (mse_at(w + dw, b_opt) - mse_at(w - dw, b_opt)) / (2 * dw)
    arrow_dx = -0.12 * np.sign(grad_w)          # small arrow pointing downhill
    arrow_dy = grad_w * arrow_dx
    ax2.annotate(
        "", xy=(w + arrow_dx, mse_at(w, b_opt) + arrow_dy),
        xytext=(w, mse_at(w, b_opt)),
        arrowprops=dict(arrowstyle="->", color=RED, lw=2),
    )
    ax2.text(
        w + arrow_dx + (0.04 if arrow_dx > 0 else -0.04),
        mse_at(w, b_opt) + arrow_dy,
        f"gradient\n= {grad_w:.2f}",
        fontsize=8, color=RED, ha="center",
    )

    ax2.set_title("Loss surface: MSE vs w", fontsize=11, color=DARK, pad=8)
    ax2.set_xlabel("w  (slope)", fontsize=10)
    ax2.set_ylabel("MSE", fontsize=10)
    ax2.legend(fontsize=8.5, frameon=True)
    ax2.grid(alpha=0.18)

    # ── super title ──────────────────────────────────────────────────────────
    fig.suptitle(
        "Can you eyeball the best fit?  — Moving the line by hand vs gradient descent",
        fontsize=12, fontweight="bold", color=DARK, y=0.99,
    )

    # render to numpy array
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    frame = np.asarray(buf)[:, :, :3]   # drop alpha → RGB
    plt.close(fig)
    return frame


# ── render all frames ─────────────────────────────────────────────────────────
print("Rendering eyeball_explorer frames…")
frames = [draw_frame(w, b, caption) for w, b, caption in guesses]

# ── save GIF (hold last frame longer) ────────────────────────────────────────
durations = [0.6] * (len(frames) - 4) + [1.5] * 4   # pause on the solution
imageio.mimsave(str(GIF_PATH), frames, duration=durations, loop=0)
print(f"wrote {GIF_PATH}")

# ── save final-frame still ───────────────────────────────────────────────────
fig_still, axes_still = plt.subplots(1, 2, figsize=(13, 5.5), facecolor="white")
fig_still.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.13, wspace=0.35)

ax = axes_still[0]
ax.scatter(X_vis, y_vis, s=10, alpha=0.35, color=LIGHT_BLUE)
xs = np.array([X_vis.min(), X_vis.max()])
ax.plot(xs, w_opt * xs + b_opt, color=GREEN, linewidth=3, label=f"Optimal: w={w_opt:.3f}, b={b_opt:.3f}")
ax.set_title(f"Optimal fit  —  MSE = {mse_at(w_opt, b_opt):.4f}", fontsize=11, color=DARK, pad=8)
ax.set_xlabel("MedInc (×$10k)", fontsize=10)
ax.set_ylabel("MedHouseVal (×$100k)", fontsize=10)
ax.legend(fontsize=9)
ax.set_xlim(X_vis.min() - 0.2, X_vis.max() + 0.2)
ax.set_ylim(-0.1, 5.5)
ax.grid(alpha=0.2)

ax2 = axes_still[1]
ax2.plot(w_grid, mse_grid, color=BLUE, linewidth=2.5)
ax2.axvline(w_opt, color=GREEN, linewidth=2, linestyle="--")
ax2.scatter([w_opt], [mse_at(w_opt, b_opt)], s=160, color=GREEN, zorder=5,
            label=f"Minimum at w* = {w_opt:.3f}")
ax2.set_title("MSE(w) — minimum found", fontsize=11, color=DARK, pad=8)
ax2.set_xlabel("w  (slope)", fontsize=10)
ax2.set_ylabel("MSE", fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.2)

fig_still.suptitle(
    "Can you eyeball the best fit?  — Moving the line by hand vs gradient descent",
    fontsize=12, fontweight="bold", color=DARK, y=0.99,
)
fig_still.savefig(str(PNG_PATH), dpi=130, bbox_inches="tight")
plt.close(fig_still)
print(f"wrote {PNG_PATH}")
