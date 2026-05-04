from pathlib import Path
"""Generate ch4 Network architecture.png — MLP diagram + activation functions."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── hand-drawn style ─────────────────────────────────────────────────
plt.xkcd(scale=0.8, length=100, randomness=2)

fig = plt.figure(figsize=(18, 10), facecolor="white")
fig.suptitle("Neural Network Architecture", fontsize=22, fontweight="bold", y=0.97)

# ── Layout: top row = MLP, bottom row = 3 activation plots ──────────
gs = fig.add_gridspec(2, 3, hspace=0.50, wspace=0.35,
                      left=0.05, right=0.97, top=0.88, bottom=0.10,
                      height_ratios=[1.3, 1])

ax_mlp = fig.add_subplot(gs[0, :])   # full-width top
ax_relu = fig.add_subplot(gs[1, 0])
ax_sig  = fig.add_subplot(gs[1, 1])
ax_tanh = fig.add_subplot(gs[1, 2])

# ── Colors ───────────────────────────────────────────────────────────
BLUE   = "#2E86C1"
ORANGE = "#E67E22"
GREEN  = "#27AE60"
PURPLE = "#8E44AD"
DARK   = "#2C3E50"
GREY   = "#BDC3C7"

# ════════════════════════════════════════════════════════════════════
# PANEL 1 — MLP Architecture
# ════════════════════════════════════════════════════════════════════
ax_mlp.set_xlim(-0.5, 10.5)
ax_mlp.set_ylim(-1.5, 6.0)
ax_mlp.axis("off")
ax_mlp.set_title("Feedforward Neural Network (MLP)", fontsize=14,
                  fontweight="bold", pad=10, color=DARK)

# Layer definitions: x-position, node y-positions, color, labels, layer name
layers = [
    (1.0,  [1.0, 2.5, 4.0],              BLUE,   ["x1", "x2", "x3"],            "Input\nlayer"),
    (4.0,  [0.25, 1.5, 2.75, 4.0],       ORANGE, ["h1", "h2", "h3", "h4"],      "Hidden\nlayer 1"),
    (7.0,  [0.75, 2.25, 3.75],           ORANGE, ["h1", "h2", "h3"],            "Hidden\nlayer 2"),
    (10.0, [2.5],                          GREEN,  ["y"],                          "Output\nlayer"),
]

node_radius = 0.42

# To render clean text inside nodes, we clear the xkcd path effects
import matplotlib.patheffects as pe

# Draw edges first (behind nodes)
for li in range(len(layers) - 1):
    x1, ys1, _, _, _ = layers[li]
    x2, ys2, _, _, _ = layers[li + 1]
    for y1 in ys1:
        for y2 in ys2:
            ax_mlp.plot([x1 + node_radius, x2 - node_radius], [y1, y2],
                        "-", color=GREY, lw=0.9, alpha=0.6, zorder=1)

# Weight annotation on one edge
ax_mlp.annotate("wij",
                xy=(2.5, 3.25), fontsize=10, color=DARK,
                fontweight="bold", ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor=GREY, lw=0.6, alpha=0.85))

# Draw nodes
for lx, ys, color, labels_list, layer_name in layers:
    for y, lbl in zip(ys, labels_list):
        circle = plt.Circle((lx, y), node_radius, facecolor=color,
                            edgecolor="white", lw=2, zorder=3)
        ax_mlp.add_patch(circle)
        txt = ax_mlp.text(lx, y, lbl, ha="center", va="center",
                    fontsize=11, color="white", fontweight="bold", zorder=4)
        txt.set_path_effects([])
    # Layer label below
    mid_y = np.mean(ys)
    ax_mlp.text(lx, -1.0, layer_name, ha="center", va="top",
                fontsize=9, color=DARK, fontweight="bold")

# Activation annotations between layers
act_info = [
    (layers[0], layers[1], "ReLU"),
    (layers[1], layers[2], "ReLU"),
    (layers[2], layers[3], "Linear"),
]
for (lx1, _, _, _, _), (lx2, _, _, _, _), act_label in act_info:
    mid_x = (lx1 + lx2) / 2
    ax_mlp.text(mid_x, 5.0, act_label, ha="center", va="center",
                fontsize=10, color=PURPLE, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#F5EEF8",
                          edgecolor=PURPLE, lw=0.8))

# Forward pass arrow
ax_mlp.annotate("", xy=(9.5, -0.7), xytext=(1.5, -0.7),
                arrowprops=dict(arrowstyle="-|>", color=DARK, lw=2,
                                mutation_scale=16))
ax_mlp.text(5.5, -0.55, "Forward pass", ha="center", va="bottom",
            fontsize=10, color=DARK, fontweight="bold")

# ════════════════════════════════════════════════════════════════════
# PANEL 2 — Activation Functions
# ════════════════════════════════════════════════════════════════════
x = np.linspace(-4, 4, 200)

# --- ReLU ---
ax_relu.set_title("ReLU", fontsize=12, fontweight="bold", color=BLUE, pad=8)
relu = np.maximum(0, x)
ax_relu.plot(x, relu, color=BLUE, lw=2.5, zorder=3)
ax_relu.axhline(0, color=GREY, lw=0.8, zorder=1)
ax_relu.axvline(0, color=GREY, lw=0.8, zorder=1)
ax_relu.set_xlabel("$z$", fontsize=10, color=DARK)
ax_relu.set_ylabel("$f(z)$", fontsize=10, color=DARK)
ax_relu.set_xlim(-4, 4)
ax_relu.set_ylim(-0.5, 4)
ax_relu.set_xticks([-4, -2, 0, 2, 4])
ax_relu.set_yticks([0, 1, 2, 3, 4])
ax_relu.text(2.5, 3.2, "$f(z) = \\max(0, z)$", fontsize=9, color=BLUE,
             fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="#D6EAF8",
                       edgecolor=BLUE, lw=0.6))
# Mark the kink at origin
ax_relu.plot(0, 0, "o", color=BLUE, markersize=6, zorder=4)

# --- Sigmoid ---
ax_sig.set_title("Sigmoid", fontsize=12, fontweight="bold", color=GREEN, pad=8)
sigmoid = 1 / (1 + np.exp(-x))
ax_sig.plot(x, sigmoid, color=GREEN, lw=2.5, zorder=3)
ax_sig.axhline(0, color=GREY, lw=0.8, zorder=1)
ax_sig.axhline(0.5, color="#E74C3C", lw=1, ls="--", zorder=2, alpha=0.6)
ax_sig.axhline(1, color=GREY, lw=0.8, ls="--", zorder=1, alpha=0.5)
ax_sig.axvline(0, color=GREY, lw=0.8, zorder=1)
ax_sig.set_xlabel("$z$", fontsize=10, color=DARK)
ax_sig.set_ylabel("$\\sigma(z)$", fontsize=10, color=DARK)
ax_sig.set_xlim(-4, 4)
ax_sig.set_ylim(-0.1, 1.1)
ax_sig.set_xticks([-4, -2, 0, 2, 4])
ax_sig.set_yticks([0, 0.5, 1.0])
ax_sig.text(0.5, 0.85, "$\\sigma(z) = \\frac{1}{1+e^{-z}}$", fontsize=9,
            color=GREEN, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#EAFAF1",
                      edgecolor=GREEN, lw=0.6))
ax_sig.plot(0, 0.5, "o", color=GREEN, markersize=6, zorder=4)

# --- Tanh ---
ax_tanh.set_title("Tanh", fontsize=12, fontweight="bold", color=ORANGE, pad=8)
tanh = np.tanh(x)
ax_tanh.plot(x, tanh, color=ORANGE, lw=2.5, zorder=3)
ax_tanh.axhline(0, color=GREY, lw=0.8, zorder=1)
ax_tanh.axhline(1, color=GREY, lw=0.8, ls="--", zorder=1, alpha=0.5)
ax_tanh.axhline(-1, color=GREY, lw=0.8, ls="--", zorder=1, alpha=0.5)
ax_tanh.axvline(0, color=GREY, lw=0.8, zorder=1)
ax_tanh.set_xlabel("$z$", fontsize=10, color=DARK)
ax_tanh.set_ylabel("$f(z)$", fontsize=10, color=DARK)
ax_tanh.set_xlim(-4, 4)
ax_tanh.set_ylim(-1.3, 1.3)
ax_tanh.set_xticks([-4, -2, 0, 2, 4])
ax_tanh.set_yticks([-1, 0, 1])
ax_tanh.text(1.0, -0.8, "$f(z) = \\tanh(z)$", fontsize=9, color=ORANGE,
             fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="#FDEBD0",
                       edgecolor=ORANGE, lw=0.6))
ax_tanh.plot(0, 0, "o", color=ORANGE, markersize=6, zorder=4)

# ── Shared section label for activation row ─────────────────────────
fig.text(0.5, 0.48, "Activation Functions",
         ha="center", va="center", fontsize=14, fontweight="bold", color=DARK)

# ── Shared caption ──────────────────────────────────────────────────
fig.text(0.5, 0.025,
         "A neural network stacks linear transformations with non-linear "
         "activation functions to learn complex representations layer by layer.",
         ha="center", va="center", fontsize=10, color="#555", style="italic",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8F9F9",
                   edgecolor="#BDC3C7", lw=0.8))

out = str(Path(__file__).resolve().parent.parent / "img" / "ch4 Network architecture.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out}")
