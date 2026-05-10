"""Generate a static annotated figure showing Xᵀe matrix multiplication breakdown.

Output:
    ../img/xtranspose_breakdown.png

Run:  python gen_ch02_xtranspose_breakdown.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
OUT_DIR = HERE.parent / "img"
OUT_DIR.mkdir(exist_ok=True)

# ── palette ────────────────────────────────────────────────────────────────────
BG        = "#1a1a2e"      # dark navy background
PANEL_BG  = "#16213e"      # slightly lighter navy for panels
BORDER    = "#0f3460"      # panel border colour
WHITE     = "#e8e8f0"      # near-white for text/values
DIM       = "#8888aa"      # dimmed text for labels
BLUE      = "#3b82f6"      # feature 1 (x₁, MedInc_s)
GREEN     = "#10b981"      # feature 2 (x₂, HouseAge_s)
ERROR_CLR = "#f59e0b"      # amber for error vector
RESULT1   = BLUE           # result row 1 inherits feature 1 colour
RESULT2   = GREEN          # result row 2 inherits feature 2 colour
ARROW_CLR = "#475569"      # subtle arrow colour

# ── data ───────────────────────────────────────────────────────────────────────
X = np.array([
    [0.5,  1.0],
    [1.5,  0.0],
    [2.0, -1.0],
])
e = np.array([-1.5, -2.5, -4.0])
Xt_e = X.T @ e  # should be [-12.5, 2.5]

# ─────────────────────────────────────────────────────────────────────────────
# Figure layout
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 7), facecolor=BG)
fig.patch.set_facecolor(BG)

# Reserve space: 3 axes side by side, with a wide axes for X (left),
# narrow for e (middle), narrow for result (right).
# We use GridSpec to give unequal widths.
from matplotlib.gridspec import GridSpec
gs = GridSpec(1, 3, figure=fig, width_ratios=[2.4, 1.2, 2.4],
              left=0.04, right=0.96, bottom=0.15, top=0.82, wspace=0.10)

ax_X  = fig.add_subplot(gs[0])   # X matrix (3×2)
ax_e  = fig.add_subplot(gs[1])   # error vector e (3×1)
ax_R  = fig.add_subplot(gs[2])   # result Xᵀe (2×1)

for ax in (ax_X, ax_e, ax_R):
    ax.set_facecolor(PANEL_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(1.5)

# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────
def cell(ax, x, y, text, color=WHITE, fontsize=13, weight="normal", alpha=1.0,
         ha="center", va="center"):
    """Place a text label at (x, y) in axis-fraction coordinates."""
    ax.text(x, y, text, color=color, fontsize=fontsize, fontweight=weight,
            ha=ha, va=va, alpha=alpha,
            transform=ax.transAxes,
            fontfamily="monospace")

def hline(ax, y, x0=0.05, x1=0.95, color=BORDER, lw=0.8):
    ax.axhline(y, xmin=x0, xmax=x1, color=color, linewidth=lw)

def vline(ax, x, y0=0.05, y1=0.95, color=BORDER, lw=0.8):
    ax.axvline(x, ymin=y0, ymax=y1, color=color, linewidth=lw)

# ─────────────────────────────────────────────────────────────────────────────
# LEFT PANEL — X matrix (3 rows × 2 cols)
#
# We show X as given (not Xᵀ) so readers see the raw data,
# then the title clarifies the transpose role.
# ─────────────────────────────────────────────────────────────────────────────
# Row y-centres (top to bottom: i=1, 2, 3)
row_ys = [0.72, 0.50, 0.28]
# Column x-centres (left: x₁, right: x₂)
col_xs = [0.36, 0.66]

# Panel title
cell(ax_X, 0.50, 0.93, "X  (3 × 2)", color=WHITE, fontsize=12, weight="bold")

# Column headers
cell(ax_X, col_xs[0], 0.83,
     "x₁\n(MedInc_s)",  color=BLUE,  fontsize=9.5, weight="bold")
cell(ax_X, col_xs[1], 0.83,
     "x₂\n(HouseAge_s)", color=GREEN, fontsize=9.5, weight="bold")

# Row labels
for idx, ry in enumerate(row_ys):
    cell(ax_X, 0.12, ry, f"i={idx+1}", color=DIM, fontsize=10)

# Grid lines separating rows
for y in [0.61, 0.39]:
    hline(ax_X, y, x0=0.17, x1=0.93)

# Vertical separator between columns
vline(ax_X, 0.50, y0=0.10, y1=0.90)

# Values — x₁ column in blue, x₂ column in green
x1_vals = [f"{X[i,0]:+.1f}" for i in range(3)]
x2_vals = [f"{X[i,1]:+.1f}" for i in range(3)]

for i, ry in enumerate(row_ys):
    cell(ax_X, col_xs[0], ry, x1_vals[i], color=BLUE,  fontsize=14, weight="bold")
    cell(ax_X, col_xs[1], ry, x2_vals[i], color=GREEN, fontsize=14, weight="bold")

# Outer bracket hint — left bracket
ax_X.text(0.04, 0.50, "[", color=WHITE, fontsize=64, va="center",
          ha="center", transform=ax_X.transAxes, alpha=0.55)
ax_X.text(0.96, 0.50, "]", color=WHITE, fontsize=64, va="center",
          ha="center", transform=ax_X.transAxes, alpha=0.55)

# ─────────────────────────────────────────────────────────────────────────────
# MIDDLE PANEL — error vector e (3×1)
# ─────────────────────────────────────────────────────────────────────────────
cell(ax_e, 0.50, 0.93, "e  (3 × 1)", color=WHITE, fontsize=12, weight="bold")
cell(ax_e, 0.50, 0.83, "ŷ − y\n(Epoch 1,\ninit)",
     color=ERROR_CLR, fontsize=8.5, weight="bold")

e_vals = [f"{v:+.1f}" for v in e]
for i, ry in enumerate(row_ys):
    cell(ax_e, 0.50, ry, e_vals[i], color=ERROR_CLR, fontsize=14, weight="bold")
    # Subtly mark the row alignment with a faint stripe
    if i < 2:
        hline(ax_e, (row_ys[i] + row_ys[i+1]) / 2, x0=0.10, x1=0.90)

# Brackets
ax_e.text(0.05, 0.50, "[", color=WHITE, fontsize=64, va="center",
          ha="center", transform=ax_e.transAxes, alpha=0.55)
ax_e.text(0.95, 0.50, "]", color=WHITE, fontsize=64, va="center",
          ha="center", transform=ax_e.transAxes, alpha=0.55)

# ─────────────────────────────────────────────────────────────────────────────
# RIGHT PANEL — result Xᵀe (2×1) with dot-product annotations
# ─────────────────────────────────────────────────────────────────────────────
# Two result rows: top = ∂L/∂w₁, bottom = ∂L/∂w₂
res_ys = [0.70, 0.30]
res_colors = [BLUE, GREEN]

cell(ax_R, 0.50, 0.93, "Xᵀe  (2 × 1)", color=WHITE, fontsize=12, weight="bold")

# Dot-product arithmetic
dot_lines = [
    [
        "(0.5)(−1.5) + (1.5)(−2.5) + (2.0)(−4.0)",
        "= −12.5",
        "∂L/∂w₁ = (2/3)(−12.5) = −8.33",
    ],
    [
        "(1.0)(−1.5) + (0.0)(−2.5) + (−1.0)(−4.0)",
        "= +2.5",
        "∂L/∂w₂ = (2/3)(+2.5) = +1.67",
    ],
]
result_vals = [f"{Xt_e[0]:+.1f}", f"{Xt_e[1]:+.1f}"]

for idx, ry in enumerate(res_ys):
    col = res_colors[idx]
    dot = dot_lines[idx]

    # Large result value on the left side
    cell(ax_R, 0.14, ry, result_vals[idx],
         color=col, fontsize=18, weight="bold")

    # Arithmetic on the right side — three lines
    cell(ax_R, 0.60, ry + 0.07, dot[0],
         color=WHITE, fontsize=7.5, ha="center")
    cell(ax_R, 0.60, ry - 0.00, dot[1],
         color=col, fontsize=9, weight="bold", ha="center")
    cell(ax_R, 0.60, ry - 0.09, dot[2],
         color=col, fontsize=8, ha="center", alpha=0.85)

    # Divider between rows
    if idx == 0:
        hline(ax_R, (res_ys[0] + res_ys[1]) / 2, x0=0.05, x1=0.95)

# Brackets
ax_R.text(0.02, 0.50, "[", color=WHITE, fontsize=64, va="center",
          ha="center", transform=ax_R.transAxes, alpha=0.55)
ax_R.text(0.23, 0.50, "]", color=WHITE, fontsize=64, va="center",
          ha="center", transform=ax_R.transAxes, alpha=0.55)

# ─────────────────────────────────────────────────────────────────────────────
# Cross-panel arrows connecting Xᵀ rows to result elements
#
# We use fig.transFigure coordinates so arrows can cross sub-plot boundaries.
# ─────────────────────────────────────────────────────────────────────────────
def ax_to_fig(ax, x_ax, y_ax):
    """Convert axis-fraction (x_ax, y_ax) → figure-fraction coordinates."""
    pt = ax.transAxes.transform((x_ax, y_ax))
    return fig.transFigure.inverted().transform(pt)

# Arrow from X col 0 (blue) → result row 0 (blue)
# Source: right edge of ax_X at the mid-row y = 0.50 (represents whole column)
# Dest: left bracket region of ax_R at y matching res_ys[0]
arrow_specs = [
    # (source ax, src_x, src_y,  dest ax, dst_x, dst_y,  colour)
    (ax_X, 0.97, 0.50,  ax_R, 0.03, res_ys[0] / 1.0,  BLUE),
    (ax_X, 0.97, 0.50,  ax_R, 0.03, res_ys[1] / 1.0,  GREEN),
]

# Remap result y from axis space (0-1 of data) to figure coords
# res_ys are already axis-fraction values
arrow_specs = [
    (ax_X, 0.97, 0.50,  ax_R, 0.03, res_ys[0],  BLUE),
    (ax_X, 0.97, 0.50,  ax_R, 0.03, res_ys[1],  GREEN),
]

for (src_ax, sx, sy, dst_ax, dx, dy, col) in arrow_specs:
    x0f, y0f = ax_to_fig(src_ax, sx, sy)
    x1f, y1f = ax_to_fig(dst_ax, dx, dy)

    arrow = FancyArrowPatch(
        posA=(x0f, y0f), posB=(x1f, y1f),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=12,
        color=col,
        alpha=0.55,
        lw=1.5,
        connectionstyle="arc3,rad=0.0",
        clip_on=False,
    )
    fig.add_artist(arrow)

# ─────────────────────────────────────────────────────────────────────────────
# Also draw a small "⊗" / "·" symbol between middle panels to indicate
# the matrix-vector multiply operator
# ─────────────────────────────────────────────────────────────────────────────
# Place operator labels in figure-fraction space between panels
for between_ax_left, between_ax_right, symbol in [
    (ax_X, ax_e, "·"),
    (ax_e, ax_R, "="),
]:
    x_left  = ax_to_fig(between_ax_left,  1.0, 0.50)[0]
    x_right = ax_to_fig(between_ax_right, 0.0, 0.50)[0]
    x_mid   = (x_left + x_right) / 2
    y_mid   = ax_to_fig(between_ax_left, 0.0, 0.50)[1]  # same vertical midpoint

    fig.text(x_mid, y_mid, symbol,
             color=WHITE, fontsize=22, ha="center", va="center",
             transform=fig.transFigure, alpha=0.7)

# ─────────────────────────────────────────────────────────────────────────────
# The "Xᵀ" label clarification — note left panel is X; Xᵀ is implied in the op
# ─────────────────────────────────────────────────────────────────────────────
# Add a note below ax_X clarifying the operation
ax_X.text(0.50, 0.06,
          "Xᵀ used in Xᵀe\n(cols of X become rows of Xᵀ)",
          color=DIM, fontsize=8, ha="center", va="bottom",
          transform=ax_X.transAxes, style="italic")

# ─────────────────────────────────────────────────────────────────────────────
# Colour legend (bottom-left)
# ─────────────────────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(color=BLUE,      label="Feature 1 — x₁ (MedInc_s)"),
    mpatches.Patch(color=GREEN,     label="Feature 2 — x₂ (HouseAge_s)"),
    mpatches.Patch(color=ERROR_CLR, label="Error vector e = ŷ − y"),
]
fig.legend(
    handles=legend_patches,
    loc="lower left",
    bbox_to_anchor=(0.02, 0.01),
    ncol=3,
    framealpha=0.15,
    edgecolor=BORDER,
    fontsize=8.5,
    facecolor=PANEL_BG,
    labelcolor=WHITE,
)

# ─────────────────────────────────────────────────────────────────────────────
# Title and subtitle
# ─────────────────────────────────────────────────────────────────────────────
fig.suptitle(
    "How Xᵀe computes gradient components simultaneously",
    fontsize=14, fontweight="bold", color=WHITE, y=0.96,
)
fig.text(
    0.50, 0.03,
    "Row j of Xᵀ = feature j's column from X — dotted against error → one gradient",
    ha="center", va="bottom",
    fontsize=9, color=DIM, style="italic",
    transform=fig.transFigure,
)

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
out_path = OUT_DIR / "xtranspose_breakdown.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"Saved → {out_path}")
