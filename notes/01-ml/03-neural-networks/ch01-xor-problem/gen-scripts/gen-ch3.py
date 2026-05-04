from pathlib import Path
"""Generate ch3 XOR Problem.png — hand-drawn colored sketch style."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── hand-drawn style ─────────────────────────────────────────────────
plt.xkcd(scale=0.8, length=100, randomness=2)

fig = plt.figure(figsize=(18, 9), facecolor="white")
fig.suptitle("XOR Problem", fontsize=24, fontweight="bold", y=0.96)

# ── Layout: 3 columns ───────────────────────────────────────────────
gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.35, 1],
                      wspace=0.15, left=0.06, right=0.96,
                      top=0.84, bottom=0.17)

ax_input = fig.add_subplot(gs[0, 0])
ax_arrow = fig.add_subplot(gs[0, 1])
ax_trans = fig.add_subplot(gs[0, 2])

# ── Colors ───────────────────────────────────────────────────────────
BLUE      = "#2E86C1"
RED       = "#E74C3C"
PALE_BLUE = "#D6EAF8"
PALE_RED  = "#FADBD8"
DARK      = "#2C3E50"
GOLD      = "#F39C12"
GREEN     = "#27AE60"

# ════════════════════════════════════════════════════════════════════
# PANEL 1 — XOR Input Space
# ════════════════════════════════════════════════════════════════════
ax_input.set_title("Input Space\n(not linearly separable)", fontsize=13,
                   fontweight="bold", pad=10, color=DARK)
ax_input.set_xlabel("$x_1$", fontsize=13, color=DARK)
ax_input.set_ylabel("$x_2$", fontsize=13, color=DARK)
ax_input.set_xlim(-0.4, 1.5)
ax_input.set_ylim(-0.4, 1.5)
ax_input.set_xticks([0, 1])
ax_input.set_yticks([0, 1])
ax_input.set_aspect("equal")

# XOR=0 points (blue circles)
xor0_x = [0, 1]
xor0_y = [0, 1]
ax_input.scatter(xor0_x, xor0_y, s=280, c=BLUE, marker="o", zorder=5,
                 edgecolors="white", linewidth=2, label="XOR = 0")

# XOR=1 points (red X markers)
xor1_x = [0, 1]
xor1_y = [1, 0]
ax_input.scatter(xor1_x, xor1_y, s=280, c=RED, marker="X", zorder=5,
                 edgecolors="white", linewidth=1, label="XOR = 1")

# Annotate each point
labels = {(0, 0): "(0,0)\nXOR=0", (0, 1): "(0,1)\nXOR=1",
          (1, 0): "(1,0)\nXOR=1", (1, 1): "(1,1)\nXOR=0"}
offsets = {(0, 0): (18, -22), (0, 1): (18, 14),
           (1, 0): (-18, -22), (1, 1): (-18, 14)}
for (px, py), txt in labels.items():
    ox, oy = offsets[(px, py)]
    ax_input.annotate(txt, (px, py), textcoords="offset points",
                      xytext=(ox, oy), fontsize=8.5, color=DARK,
                      ha="center", va="center",
                      fontweight="bold")

# Show two failed separator lines
x_line = np.linspace(-0.3, 1.3, 50)
for slope, intercept in [(1, -0.1), (-0.5, 0.9)]:
    y_line = slope * x_line + intercept
    ax_input.plot(x_line, y_line, "--", color="#BDC3C7", lw=2, alpha=0.7, zorder=2)

# Red X near where the failed lines cross
ax_input.plot([0.45, 0.65], [0.55, 0.35], '-', color=RED, lw=3, alpha=0.8, zorder=6)
ax_input.plot([0.45, 0.65], [0.35, 0.55], '-', color=RED, lw=3, alpha=0.8, zorder=6)

# Caption below panel — use data coordinates to avoid clipping
ax_input.text(0.5, -0.32, "No single line separates the classes",
              ha="center", va="top", fontsize=10, color=RED,
              fontweight="bold", style="italic")

ax_input.legend(loc="upper left", fontsize=9, framealpha=0.9)

# ════════════════════════════════════════════════════════════════════
# PANEL 2 — Arrow (Hidden Layer Transform)
# ════════════════════════════════════════════════════════════════════
ax_arrow.set_xlim(0, 1)
ax_arrow.set_ylim(0, 1)
ax_arrow.axis("off")

# Big arrow
ax_arrow.annotate("", xy=(0.85, 0.52), xytext=(0.15, 0.52),
                  xycoords="axes fraction", textcoords="axes fraction",
                  arrowprops=dict(arrowstyle="-|>", color=GOLD,
                                  lw=3.5, mutation_scale=22))

ax_arrow.text(0.5, 0.68, "Hidden Layer\nTransforms Space",
              ha="center", va="center", fontsize=11,
              fontweight="bold", color=DARK,
              transform=ax_arrow.transAxes,
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF9E7",
                        edgecolor=GOLD, lw=1.2))

ax_arrow.text(0.5, 0.35, "$\\mathbf{h} = \\sigma(\\mathbf{W}_1 \\mathbf{x} + \\mathbf{b}_1)$",
              ha="center", va="center", fontsize=12, color=DARK,
              transform=ax_arrow.transAxes)

# Small network sketch
node_r = 0.035
layers_x = [0.25, 0.5, 0.75]
layer_nodes = [[0.05, 0.19], [0.05, 0.12, 0.19], [0.12]]

for li, (lx, nodes_y) in enumerate(zip(layers_x, layer_nodes)):
    for ny in nodes_y:
        circ = plt.Circle((lx, ny), node_r, facecolor=[BLUE, GOLD, RED][li],
                          edgecolor="white", lw=1.2, zorder=4,
                          transform=ax_arrow.transAxes)
        ax_arrow.add_patch(circ)
    if li < len(layers_x) - 1:
        next_nodes = layer_nodes[li + 1]
        nx = layers_x[li + 1]
        for ny1 in nodes_y:
            for ny2 in next_nodes:
                ax_arrow.plot([lx + node_r, nx - node_r],
                              [ny1, ny2], "-", color="#BDC3C7", lw=0.8,
                              transform=ax_arrow.transAxes, zorder=3)

# ════════════════════════════════════════════════════════════════════
# PANEL 3 — Transformed Feature Space
# ════════════════════════════════════════════════════════════════════
# Classic XOR hidden-layer solution:
#   h1 ≈ OR(x1,x2),  h2 ≈ NAND(x1,x2)
# Mapping:
#   (0,0) → h=(0.12, 0.88)  XOR=0  (upper-left)
#   (0,1) → h=(0.73, 0.73)  XOR=1  (middle-right)
#   (1,0) → h=(0.73, 0.73)  XOR=1  (middle-right, offset slightly)
#   (1,1) → h=(0.88, 0.12)  XOR=0  (lower-right)
# Boundary: h1 + h2 = 1  separates them cleanly.

ax_trans.set_title("Transformed Space\n(linearly separable)", fontsize=13,
                   fontweight="bold", pad=10, color=DARK)
ax_trans.set_xlabel("$h_1$  (hidden unit 1)", fontsize=11, color=DARK)
ax_trans.set_ylabel("$h_2$  (hidden unit 2)", fontsize=11, color=DARK)
ax_trans.set_xlim(-0.1, 1.1)
ax_trans.set_ylim(-0.1, 1.1)
ax_trans.set_xticks([0, 0.5, 1.0])
ax_trans.set_yticks([0, 0.5, 1.0])
ax_trans.set_aspect("equal")

# XOR=0 points  (upper-left and lower-right — sum ≈ 1.0)
h_xor0 = np.array([[0.12, 0.88], [0.88, 0.12]])
# XOR=1 points  (both in upper-right — sum > 1.0)
h_xor1 = np.array([[0.70, 0.76], [0.76, 0.70]])

# Decision boundary:  h2 = -h1 + 1.05  (i.e. h1+h2 = 1.05)
db_x = np.linspace(-0.05, 1.1, 100)
db_y = -db_x + 1.05

# Shade the regions
ax_trans.fill_between(db_x, db_y, 1.2, alpha=0.13, color=RED, zorder=1,
                      label="_nolegend_")
ax_trans.fill_between(db_x, -0.2, db_y, alpha=0.13, color=BLUE, zorder=1,
                      label="_nolegend_")

# Decision boundary line
ax_trans.plot(db_x, db_y, "-", color=DARK, lw=2.5, zorder=3)
ax_trans.text(0.50, 0.62, "Decision\nboundary",
              fontsize=9, color=DARK, fontweight="bold",
              ha="center", va="center",
              rotation=-45, rotation_mode="anchor",
              bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                        edgecolor=DARK, lw=0.8, alpha=0.9))

# XOR=0 points
ax_trans.scatter(h_xor0[:, 0], h_xor0[:, 1], s=280, c=BLUE, marker="o",
                 edgecolors="white", linewidth=2, zorder=5, label="XOR = 0")
# XOR=1 points
ax_trans.scatter(h_xor1[:, 0], h_xor1[:, 1], s=280, c=RED, marker="X",
                 edgecolors="white", linewidth=1, zorder=5, label="XOR = 1")

# Annotate transformed points
annot = [
    (h_xor0[0], "h(0,0)", (-18, -14)),
    (h_xor0[1], "h(1,1)", (18, 10)),
    (h_xor1[0], "h(0,1)", (18, 10)),
    (h_xor1[1], "h(1,0)", (-18, -14)),
]
for (hx, hy), lbl, (ox, oy) in annot:
    ax_trans.annotate(lbl, (hx, hy), textcoords="offset points",
                      xytext=(ox, oy), fontsize=9, color=DARK,
                      fontweight="bold", ha="center")

ax_trans.legend(loc="lower left", fontsize=9, framealpha=0.9)

# Checkmark — use plain ASCII since unicode may not render in xkcd mode
ax_trans.text(0.88, 0.88, "OK!", fontsize=14, color=GREEN, fontweight="bold",
              ha="center", va="center", alpha=0.9, transform=ax_trans.transAxes,
              bbox=dict(boxstyle="round,pad=0.15", facecolor="#EAFAF1",
                        edgecolor=GREEN, lw=1))

# ── Shared caption ──────────────────────────────────────────────────
fig.text(0.5, 0.04,
         "A hidden layer with non-linear activations maps the XOR input space "
         "into a new representation where the classes become linearly separable.\n"
         "This is the core motivation for deep learning: learned features solve "
         "problems that raw features cannot.",
         ha="center", va="center", fontsize=10, color="#555",
         style="italic",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8F9F9",
                   edgecolor="#BDC3C7", lw=0.8))

out = str(Path(__file__).resolve().parent.parent / "img" / "ch3 XOR Problem.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out}")
