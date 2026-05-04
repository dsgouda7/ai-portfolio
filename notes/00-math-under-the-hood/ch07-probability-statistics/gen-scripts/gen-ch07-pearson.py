"""
Generate three Pearson/covariance animations for ch07 § 4b.

Story arc: the free-kick knuckleball challenge.
  x = wind speed (m/s) per training session
  y = ball deviation from target line (cm) measured at goal

Outputs (all to ../img/):
  ch07-pearson-covariance.gif   – signed-rectangle covariance build-up
  ch07-pearson-correlation.gif  – four scatter plots at different ρ values
  ch07-covariance-matrix.gif    – 5×5 free-kick feature correlation matrix build

Run from repo root:
    python notes/00-math_under_the_hood/ch07_probability_statistics/gen_scripts/gen_ch07_pearson.py
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
import io

OUT = Path(__file__).resolve().parent.parent / "img"
OUT.mkdir(exist_ok=True)

# ── shared palette (matches rest of ch07) ────────────────────────────────────
BG   = "#1a1a2e"
FG   = "#e2e8f0"
BLUE = "#3b82f6"
RED  = "#ef4444"
GREEN= "#22c55e"
GOLD = "#f59e0b"
GREY = "#64748b"

def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return Image.open(buf).copy()


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  COVARIANCE BUILD-UP
#     x = wind speed (m/s), y = ball deviation from target line (cm)
#     across 5 free-kick training sessions
# ═══════════════════════════════════════════════════════════════════════════════
X = np.array([1., 3., 2., 5., 4.])   # wind speed (m/s)
Y = np.array([8., 18., 20., 32., 22.]) # deviation (cm)
xbar, ybar = X.mean(), Y.mean()        # 3.0, 20.0
dx = X - xbar
dy = Y - ybar
products = dx * dy

LABELS = [f"S{i+1}" for i in range(len(X))]
X_LABEL = "wind speed  (m/s)"
Y_LABEL = "ball deviation  (cm)"

frames_cov = []

for k in range(1, len(X) + 1):
    fig, ax = plt.subplots(figsize=(7, 5.5), facecolor=BG)
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GREY)

    # mean lines
    ax.axvline(xbar, color=GREY, lw=1, ls="--", alpha=0.6)
    ax.axhline(ybar, color=GREY, lw=1, ls="--", alpha=0.6)
    ax.text(xbar + 0.1, 5.5, f"$\\bar{{x}}={xbar:.0f}$",
            color=GREY, fontsize=9)
    ax.text(0.25, ybar + 0.8, f"$\\bar{{y}}={ybar:.0f}$",
            color=GREY, fontsize=9)

    # draw rectangles for sessions 1..k
    for i in range(k):
        colour = BLUE if products[i] >= 0 else RED
        rx = min(X[i], xbar)
        ry = min(Y[i], ybar)
        rw = abs(dx[i])
        rh = abs(dy[i])
        ax.add_patch(mpatches.Rectangle(
            (rx, ry), rw, rh,
            linewidth=1.2, edgecolor=colour,
            facecolor=colour, alpha=0.25
        ))
        ax.plot(X[i], Y[i], "o", color=colour, ms=9, zorder=5)
        ax.annotate(
            f"  {LABELS[i]}\n  Δx={dx[i]:+.0f}, Δy={dy[i]:+.0f}\n  prod={products[i]:+.0f}",
            xy=(X[i], Y[i]), xytext=(X[i] + 0.1, Y[i] - 1.0),
            color=colour, fontsize=7.5
        )

    running_cov = products[:k].mean()
    ax.set_xlim(0, 7)
    ax.set_ylim(3, 38)
    ax.set_xlabel(X_LABEL, color=FG)
    ax.set_ylabel(Y_LABEL, color=FG)
    ax.tick_params(colors=FG)
    ax.set_title(
        f"Covariance build-up — session {k} of {len(X)}\n"
        f"Running Cov(wind, deviation) = {running_cov:.1f}  "
        f"[blue = same direction, red = opposite]",
        color=FG, fontsize=10, pad=8
    )
    fig.tight_layout()
    frames_cov.append(fig_to_pil(fig))
    plt.close(fig)

# duplicate last frame for a pause
frames_cov += [frames_cov[-1]] * 4
frames_cov[0].save(
    OUT / "ch07-pearson-covariance.gif",
    save_all=True, append_images=frames_cov[1:],
    duration=900, loop=0
)
print("✓ ch07-pearson-covariance.gif")


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  FOUR SCATTER PLOTS — different ρ values, all in the free-kick context
# ═══════════════════════════════════════════════════════════════════════════════
rng = np.random.default_rng(42)
N = 120

def make_corr_data(r, n=N, seed=42):
    rng2 = np.random.default_rng(seed)
    cov = [[1, r], [r, 1]]
    data = rng2.multivariate_normal([0, 0], cov, n)
    return data[:, 0], data[:, 1]

configs = [
    (0.97,  "$v_0$ (m/s)  ↔  ball range (m)\n$\\rho \\approx +0.97$ — near-perfect (physics)",  BLUE),
    (0.92,  "wind speed  ↔  ball deviation\n$\\rho \\approx +0.92$ — our worked example",        GREEN),
    (0.03,  "temperature  ↔  goal probability\n$\\rho \\approx 0$ — no linear relationship",     GOLD),
    (-0.55, "rest hours  ↔  launch angle error\n$\\rho \\approx -0.55$ — moderate negative",     RED),
]

frames_scatter = []
for rho, title, colour in configs:
    fig, ax = plt.subplots(figsize=(6, 5.5), facecolor=BG)
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GREY)
    xs, ys = make_corr_data(rho)
    ax.scatter(xs, ys, color=colour, alpha=0.45, s=22, edgecolors="none")
    # best-fit line
    m, b_ = np.polyfit(xs, ys, 1)
    xl = np.linspace(xs.min(), xs.max(), 200)
    ax.plot(xl, m * xl + b_, color=colour, lw=2, alpha=0.9)
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.axvline(0, color=GREY, lw=0.6, ls="--", alpha=0.5)
    ax.axhline(0, color=GREY, lw=0.6, ls="--", alpha=0.5)
    ax.set_xlabel("feature $x$ (standardised)", color=FG)
    ax.set_ylabel("feature / outcome $y$ (standardised)", color=FG)
    ax.tick_params(colors=FG)
    rho_label = f"$\\rho = {rho:+.2f}$"
    ax.text(0.05, 0.93, rho_label, transform=ax.transAxes,
            fontsize=16, color=colour, fontweight="bold",
            va="top")
    ax.set_title(title, color=FG, fontsize=10, pad=8)
    fig.tight_layout()
    pil_frame = fig_to_pil(fig)
    frames_scatter += [pil_frame] * 2
    plt.close(fig)

frames_scatter[0].save(
    OUT / "ch07-pearson-correlation.gif",
    save_all=True, append_images=frames_scatter[1:],
    duration=1800, loop=0
)
print("✓ ch07-pearson-correlation.gif")


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  COVARIANCE MATRIX BUILD — 5×5 free-kick features
#
#  Features: v0 (m/s), theta (deg), wind (m/s), spin (rpm), temperature (°C)
#  Synthetic data generated from a target correlation structure:
#    v0 ↔ theta:  +0.35  (partially planned together)
#    v0 ↔ wind:   −0.10  (wind is external — no correlation)
#    v0 ↔ spin:   +0.42  (kicker adjusts spin with power)
#    v0 ↔ temp:   +0.05
#    theta ↔ wind: +0.20  (keeper adjusts angle in wind)
#    theta ↔ spin: +0.88  (angle and spin tightly coupled — same kick style)
#    theta ↔ temp: −0.08
#    wind ↔ spin:  −0.12
#    wind ↔ temp:  +0.15
#    spin ↔ temp:  +0.07
# ═══════════════════════════════════════════════════════════════════════════════
labels = ["$v_0$ (m/s)", r"$\theta$ (°)", "wind (m/s)", "spin (rpm)", "temp (°C)"]
p = len(labels)

# Build a valid positive-definite correlation matrix from the target values
target_corr = np.array([
    [1.00,  0.35, -0.10,  0.42,  0.05],
    [0.35,  1.00,  0.20,  0.88, -0.08],
    [-0.10,  0.20,  1.00, -0.12,  0.15],
    [0.42,  0.88, -0.12,  1.00,  0.07],
    [0.05, -0.08,  0.15,  0.07,  1.00],
])

rng_mat = np.random.default_rng(99)
raw = rng_mat.multivariate_normal(np.zeros(p), target_corr, size=500)
# Recompute actual correlation from the sample for honest annotation
import pandas as pd
df_kick = pd.DataFrame(raw, columns=labels)
corr = df_kick.corr().values

frames_mat = []
for reveal_up_to in range(p):
    fig, ax = plt.subplots(figsize=(7, 6.5), facecolor=BG)
    ax.set_facecolor(BG)

    display = np.full_like(corr, np.nan)
    for col in range(reveal_up_to + 1):
        display[:, col] = corr[:, col]
        display[col, :] = corr[col, :]   # symmetric

    im = ax.imshow(display, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(p))
    ax.set_yticks(range(p))
    ax.set_xticklabels(labels, rotation=30, ha="right", color=FG, fontsize=9)
    ax.set_yticklabels(labels, color=FG, fontsize=9)

    for r in range(p):
        for c in range(p):
            if not np.isnan(display[r, c]):
                val = display[r, c]
                txt_col = "white" if abs(val) > 0.55 else BG
                ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                        color=txt_col, fontsize=8)

    ax.set_title(
        f"Free-kick feature correlation matrix\n"
        f"(revealing column {reveal_up_to + 1}/{p}: {labels[reveal_up_to]})",
        color=FG, fontsize=10, pad=8
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color=FG)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=FG)
    fig.tight_layout()
    frames_mat.append(fig_to_pil(fig))
    plt.close(fig)

frames_mat += [frames_mat[-1]] * 5
frames_mat[0].save(
    OUT / "ch07-covariance-matrix.gif",
    save_all=True, append_images=frames_mat[1:],
    duration=700, loop=0
)
print("✓ ch07-covariance-matrix.gif")

print("\nAll Pearson/covariance animations written to", OUT)

