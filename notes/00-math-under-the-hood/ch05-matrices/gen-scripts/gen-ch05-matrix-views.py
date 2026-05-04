"""Generate static hero image for Pre-Requisites Ch.5 — Matrices & Matrix Calculus.

Three panels tying the chapter together on the free-throw theme:

  (left)   A 2x2 matrix acting on the unit square — the "transformation"
           picture of a matrix. Shows the original unit square and its image
           under A = [[1.5, 0.5], [0.3, 1.2]]. Determinant = area scale factor.

  (middle) Matrix-vector product as a weighted sum of columns — the
           "column picture" of Ax. For the shot-feature design matrix
           row, we show ŷ = w1·col1 + w2·col2 + w3·col3 accumulating
           graphically.

  (right)  Normal-equations fit to the full free-throw trajectory using
           a design matrix with columns [1, t, t²]. Three noisy samples,
           the closed-form least-squares parabola, and the coefficients
           read off as physics constants (h0, v0y, -g/2).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon
import numpy as np
from pathlib import Path

DARK   = "#2C3E50"
BLUE   = "#2E86C1"
GREEN  = "#27AE60"
PURPLE = "#8E44AD"
ORANGE = "#E67E22"
RED    = "#E74C3C"
GREY   = "#7F8C8D"
GOLD   = "#F39C12"

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.8, 5.2), facecolor="white")
fig.suptitle("Ch.5 — A Matrix Is a Linear Map (and Three Views of It)",
             fontsize=15, fontweight="bold", color=DARK, y=1.00)

# ── Panel 1: 2x2 matrix stretches/rotates the unit square ────────────────
A = np.array([[1.5, 0.5],
              [0.3, 1.2]])
det_A = float(np.linalg.det(A))

square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
image = square @ A.T    # apply A to each row-vector

# Draw original and transformed squares
ax1.add_patch(Polygon(square[:-1], closed=True, facecolor="#D6EAF8",
                      edgecolor=BLUE, lw=2.0, alpha=0.9, label="unit square"))
ax1.add_patch(Polygon(image[:-1], closed=True, facecolor="#FDEBD0",
                      edgecolor=ORANGE, lw=2.0, alpha=0.7, label="A · square"))

# Column vectors of A (where the basis e1, e2 land)
for col_idx, col_col in enumerate([BLUE, RED]):
    vec = A[:, col_idx]
    ax1.add_patch(FancyArrowPatch((0, 0), tuple(vec),
                                   arrowstyle="->", color=col_col,
                                   lw=2.0, mutation_scale=18))
    ax1.text(vec[0] + 0.07, vec[1] + 0.07,
             f"col {col_idx+1} = [{vec[0]:.2f}, {vec[1]:.2f}]",
             fontsize=8.5, color=col_col, fontweight="bold")

ax1.axhline(0, color=GREY, lw=0.5); ax1.axvline(0, color=GREY, lw=0.5)
ax1.set_xlim(-0.4, 2.4); ax1.set_ylim(-0.4, 2.1)
ax1.set_aspect("equal")
ax1.set_title("Transformation view: A warps space",
              fontsize=11.5, fontweight="bold", color=DARK, loc="left")
ax1.grid(True, alpha=0.25)
ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
ax1.text(0.02, 0.04,
         f"A = [[1.5, 0.5], [0.3, 1.2]]\n"
         f"det(A) = {det_A:.3f}  (area scale)",
         transform=ax1.transAxes, fontsize=8.5, color=DARK,
         va="bottom", ha="left", family="monospace",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF8E1",
                   edgecolor=GOLD, linewidth=0.9))

# ── Panel 2: column picture — matrix-vector product as sum of columns ────
# Ax = x1*col1 + x2*col2 + x3*col3
cols = np.array([[2.0, 0.5],
                 [0.5, 1.8],
                 [1.2, -0.4]]).T    # 2x3, three 2-D columns
w = np.array([0.6, 0.9, 0.5])
result = cols @ w

# Draw each weighted column as a tip-to-tail arrow
origin = np.array([0.0, 0.0])
tail = origin.copy()
col_colors = [BLUE, PURPLE, ORANGE]
for j in range(3):
    tip = tail + w[j] * cols[:, j]
    c0, c1 = cols[0, j], cols[1, j]
    ax2.add_patch(FancyArrowPatch(tuple(tail), tuple(tip),
                                   arrowstyle="->", color=col_colors[j],
                                   lw=2.0, mutation_scale=16,
                                   label=f"w{j+1}·col{j+1} = {w[j]}·[{c0:.1f}, {c1:.1f}]"))
    tail = tip

# Final vector
ax2.add_patch(FancyArrowPatch((0, 0), tuple(result),
                               arrowstyle="->", color=DARK,
                               lw=2.6, mutation_scale=18, linestyle="--"))
ax2.text(result[0] * 1.05, result[1] * 1.05 - 0.1,
         f"Ax = [{result[0]:.3f}, {result[1]:.3f}]",
         fontsize=9, color=DARK, fontweight="bold")

ax2.axhline(0, color=GREY, lw=0.5); ax2.axvline(0, color=GREY, lw=0.5)
ax2.set_xlim(-0.4, 3.0); ax2.set_ylim(-0.6, 2.6)
ax2.set_aspect("equal")
ax2.set_title("Column view: Ax = sum of weighted columns",
              fontsize=11.5, fontweight="bold", color=DARK, loc="left")
ax2.grid(True, alpha=0.25)
ax2.legend(loc="lower right", fontsize=8.0, framealpha=0.95)

# ── Panel 3: normal equations fit a parabola to noisy trajectory ─────────
rng = np.random.default_rng(4)
v0y, h0, g = 7.2, 2.10, 9.81
t_samples = np.linspace(0.05, 1.40, 14)
y_true = h0 + v0y * t_samples - 0.5 * g * t_samples ** 2
y_noisy = y_true + rng.normal(0, 0.06, size=t_samples.shape)

# Design matrix with columns [1, t, t^2]
X = np.column_stack([np.ones_like(t_samples), t_samples, t_samples ** 2])
# Normal equations: w = (XᵀX)^{-1} Xᵀ y
w_hat = np.linalg.inv(X.T @ X) @ X.T @ y_noisy

t_plot = np.linspace(0, 1.55, 300)
y_plot = w_hat[0] + w_hat[1] * t_plot + w_hat[2] * t_plot ** 2
y_truth_plot = h0 + v0y * t_plot - 0.5 * g * t_plot ** 2

ax3.axhline(3.05, color=GREEN, lw=1.0, linestyle=":", label="hoop rim (3.05 m)")
ax3.plot(t_plot, y_truth_plot, color=GREY, lw=1.2, alpha=0.7, label="true physics")
ax3.plot(t_plot, y_plot, color=PURPLE, lw=2.2,
         label="normal-equations fit")
ax3.scatter(t_samples, y_noisy, color=DARK, s=32, zorder=5, label="noisy samples")

ax3.set_xlabel("time  t  (s)"); ax3.set_ylabel("height  y  (m)")
ax3.set_title("Normal equations: one matrix solve fits a curve",
              fontsize=11.5, fontweight="bold", color=DARK, loc="left")
ax3.legend(loc="upper right", fontsize=8.2, framealpha=0.95)
ax3.grid(True, alpha=0.25)

recovered = (
    f"ŵ = (XᵀX)⁻¹Xᵀy =\n"
    f"  [ {w_hat[0]:+.3f},  {w_hat[1]:+.3f},  {w_hat[2]:+.3f} ]\n"
    f"physics: h0={h0}, v0y={v0y}, -g/2={-g/2:.3f}"
)
ax3.text(0.02, 0.04, recovered, transform=ax3.transAxes, fontsize=8.2,
         color=DARK, va="bottom", ha="left", family="monospace",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#F4ECF7",
                   edgecolor=PURPLE, linewidth=0.9))

fig.tight_layout(rect=(0, 0, 1, 0.96))
out = Path(__file__).resolve().parent / "ch05-matrix-views.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")
