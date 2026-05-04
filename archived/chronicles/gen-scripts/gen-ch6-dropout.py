"""Generate Reference/img/ch6-dropout-mask.png — correct dropout schematic.

Depicts a small fully-connected network with random binary masks during training
(some units zeroed) and the full dense network at inference time (all units active).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

BLUE = "#2E86C1"
GREY = "#BDC3C7"
RED = "#E74C3C"
DARK = "#2C3E50"

rng = np.random.default_rng(7)

fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), facecolor="white")
fig.suptitle("Dropout: random mask during training vs. full network at inference",
             fontsize=14, fontweight="bold", color=DARK, y=0.99)


def draw_net(ax, active_mask, title, subtitle):
    layer_sizes = [3, 5, 5, 2]
    layer_xs = [0.1, 0.37, 0.64, 0.91]
    # positions
    positions = []
    for lx, n in zip(layer_xs, layer_sizes):
        ys = np.linspace(0.82, 0.18, n)
        positions.append([(lx, y) for y in ys])
    # draw edges first
    for li in range(len(positions) - 1):
        for i, (x1, y1) in enumerate(positions[li]):
            for j, (x2, y2) in enumerate(positions[li + 1]):
                src_on = active_mask[li][i] if active_mask[li] is not None else True
                dst_on = active_mask[li + 1][j] if active_mask[li + 1] is not None else True
                active = src_on and dst_on
                ax.plot([x1, x2], [y1, y2],
                        color=BLUE if active else GREY,
                        lw=0.9 if active else 0.5,
                        alpha=0.85 if active else 0.35, zorder=1)
    # draw nodes
    for li, layer in enumerate(positions):
        for i, (x, y) in enumerate(layer):
            on = active_mask[li][i] if active_mask[li] is not None else True
            color = BLUE if on else "white"
            edge = DARK if on else RED
            ax.add_patch(plt.Circle((x, y), 0.035,
                                    facecolor=color, edgecolor=edge,
                                    lw=1.8, zorder=3))
            if not on:
                # draw a clear X to indicate dropped
                ax.plot([x - 0.022, x + 0.022], [y - 0.022, y + 0.022],
                        color=RED, lw=1.8, zorder=4)
                ax.plot([x - 0.022, x + 0.022], [y + 0.022, y - 0.022],
                        color=RED, lw=1.8, zorder=4)
    # layer labels
    for lx, lbl in zip(layer_xs, ["Input", "Hidden 1", "Hidden 2", "Output"]):
        ax.text(lx, 0.06, lbl, ha="center", va="center",
                fontsize=10, color=DARK, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", color=DARK, pad=6)
    ax.text(0.5, -0.02, subtitle, ha="center", va="top",
            fontsize=10, color="#555", transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.95)
    ax.set_aspect("equal")
    ax.axis("off")


# Training mask: drop ~40% of hidden units; input and output stay on.
h1 = rng.random(5) > 0.4
h2 = rng.random(5) > 0.4
train_mask = [np.ones(3, dtype=bool), h1, h2, np.ones(2, dtype=bool)]
inference_mask = [np.ones(3, dtype=bool), np.ones(5, dtype=bool),
                  np.ones(5, dtype=bool), np.ones(2, dtype=bool)]

draw_net(axes[0], train_mask,
         "Training (dropout p ≈ 0.4)",
         "random binary mask zeros a fresh subset each mini-batch")
draw_net(axes[1], inference_mask,
         "Inference",
         "full network active; weights scaled by keep-prob (or done at train time)")

# legend
ax_leg = fig.add_axes([0.35, 0.02, 0.3, 0.05]); ax_leg.axis("off")
ax_leg.add_patch(plt.Circle((0.08, 0.5), 0.06, facecolor=BLUE, edgecolor=DARK, lw=1.5))
ax_leg.text(0.15, 0.5, "active unit", va="center", fontsize=9, color=DARK)
ax_leg.add_patch(plt.Circle((0.45, 0.5), 0.06, facecolor="white", edgecolor=RED, lw=1.5))
ax_leg.plot([0.425, 0.475], [0.4, 0.6], color=RED, lw=1.5)
ax_leg.plot([0.425, 0.475], [0.6, 0.4], color=RED, lw=1.5)
ax_leg.text(0.52, 0.5, "dropped unit", va="center", fontsize=9, color=DARK)
ax_leg.set_xlim(0, 1); ax_leg.set_ylim(0, 1)

fig.tight_layout(rect=(0, 0.06, 1, 0.96))

out = Path(__file__).resolve().parent / "ch6-dropout-mask.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")
