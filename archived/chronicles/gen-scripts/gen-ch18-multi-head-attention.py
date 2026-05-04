"""Generate Reference/img/ch18-multi-head-attention.png — multi-head attention schematic.

Two-panel figure:
  (left)   Structural diagram of H parallel attention heads with distinct
           W_Q_h, W_K_h, W_V_h projections, concat, and final W_O.
  (right)  Toy 8x8 attention heatmaps from three "heads" learning different
           relationship patterns (feature-family, geography, income-anchor)
           on the California Housing features.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

DARK   = "#2C3E50"
BLUE   = "#2E86C1"
GREEN  = "#27AE60"
PURPLE = "#8E44AD"
ORANGE = "#E67E22"
ATTN1  = "#0E7490"
ATTN2  = "#3730A3"
GOLD   = "#F39C12"
RED    = "#E74C3C"
GREY   = "#7F8C8D"

fig = plt.figure(figsize=(13.2, 6.4), facecolor="white")
fig.suptitle("Multi-Head Attention — H parallel scaled-dot-product attentions, one concat",
             fontsize=13, fontweight="bold", color=DARK, y=0.995)

# ── LEFT: structural diagram ──────────────────────────────────────────────
ax = fig.add_axes([0.03, 0.08, 0.45, 0.84])
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_color("#E0E4EA")

# Input X
ax.add_patch(mpatches.FancyBboxPatch((3.8, 8.7), 2.4, 0.9,
    boxstyle="round,pad=0.05,rounding_size=0.12",
    facecolor="#FFF8E1", edgecolor=GOLD, linewidth=1.6))
ax.text(5.0, 9.15, "Input  X  ∈ ℝᵀˣᵈ", ha="center", va="center",
        fontsize=10, fontweight="bold", color=DARK)

# Three heads (columns)
head_colors = [ATTN1, PURPLE, ORANGE]
head_labels = ["Head 1", "Head 2", "Head H"]
head_x = [1.0, 4.3, 7.6]
for hx, col, lbl in zip(head_x, head_colors, head_labels):
    # Head container
    ax.add_patch(mpatches.FancyBboxPatch((hx, 2.0), 1.8, 6.0,
        boxstyle="round,pad=0.05,rounding_size=0.12",
        facecolor="none", edgecolor=col, linewidth=1.5, linestyle="--"))
    ax.text(hx + 0.9, 7.82, lbl, ha="center", va="top",
            fontsize=9.5, fontweight="bold", color=col)

    # Three projections: W_Q, W_K, W_V
    labels = [("W_Q", BLUE), ("W_K", GREEN), ("W_V", PURPLE)]
    for k, (lab, lc) in enumerate(labels):
        cx = hx + 0.25 + k * 0.50
        ax.add_patch(mpatches.Rectangle((cx, 6.5), 0.42, 0.55,
            facecolor="white", edgecolor=lc, linewidth=1.2))
        ax.text(cx + 0.21, 6.78, lab, ha="center", va="center",
                fontsize=6.8, fontweight="bold", color=lc)
    ax.text(hx + 0.9, 6.2, "Q · K · V", ha="center", va="center",
            fontsize=7.5, color=DARK, style="italic")

    # Scaled dot-product attention
    ax.add_patch(mpatches.FancyBboxPatch((hx + 0.15, 4.1), 1.5, 1.1,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        facecolor="#E0F2FE", edgecolor=col, linewidth=1.3))
    ax.text(hx + 0.9, 4.85, "softmax(QKᵀ/√dₖ)·V",
            ha="center", va="center", fontsize=7.2, color=DARK)
    ax.text(hx + 0.9, 4.4, "scaled dot-product", ha="center", va="center",
            fontsize=6.3, color=GREY, style="italic")

    # head_h output box
    ax.add_patch(mpatches.Rectangle((hx + 0.3, 2.6), 1.2, 0.7,
        facecolor=col, alpha=0.18, edgecolor=col, linewidth=1.2))
    ax.text(hx + 0.9, 2.95, f"head ∈ ℝᵀˣᵈᵏ",
            ha="center", va="center", fontsize=7.2, color=col,
            fontweight="bold")

    # Arrows inside head
    ax.annotate("", xy=(hx + 0.9, 6.5), xytext=(hx + 0.9, 8.6),
                arrowprops=dict(arrowstyle="->", color=col, lw=1.2))
    ax.annotate("", xy=(hx + 0.9, 5.2), xytext=(hx + 0.9, 6.1),
                arrowprops=dict(arrowstyle="->", color=col, lw=1.2))
    ax.annotate("", xy=(hx + 0.9, 3.3), xytext=(hx + 0.9, 4.1),
                arrowprops=dict(arrowstyle="->", color=col, lw=1.2))

# "..." between head 2 and head H
ax.text(6.55, 5.0, "· · ·", ha="center", va="center",
        fontsize=16, color=GREY)

# Concat
ax.add_patch(mpatches.FancyBboxPatch((1.0, 1.1), 8.0, 0.7,
    boxstyle="round,pad=0.04,rounding_size=0.10",
    facecolor="#F4F6F9", edgecolor=DARK, linewidth=1.3))
ax.text(5.0, 1.45, "Concat(head₁, head₂, …, headₕ)  ∈ ℝᵀˣ⁽ᴴ·ᵈᵛ⁾",
        ha="center", va="center", fontsize=9, fontweight="bold",
        color=DARK)

# Arrows from heads to concat
for hx, col in zip(head_x, head_colors):
    ax.annotate("", xy=(hx + 0.9, 1.82), xytext=(hx + 0.9, 2.55),
                arrowprops=dict(arrowstyle="->", color=col, lw=1.2))

# W_O output projection
ax.add_patch(mpatches.FancyBboxPatch((3.5, 0.15), 3.0, 0.65,
    boxstyle="round,pad=0.04,rounding_size=0.10",
    facecolor="#EAFAF1", edgecolor=GREEN, linewidth=1.5))
ax.text(5.0, 0.47, "· W_O  →  ℝᵀˣᵈ", ha="center", va="center",
        fontsize=9.5, fontweight="bold", color=GREEN)
ax.annotate("", xy=(5.0, 0.8), xytext=(5.0, 1.08),
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.3))

# Input-to-heads fan-out arrows
for hx in head_x:
    ax.annotate("", xy=(hx + 0.9, 8.0), xytext=(5.0, 8.65),
                arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.0,
                                connectionstyle="arc3,rad=0.05"))

ax.set_title("Structure:  X → H heads → concat → W_O",
             fontsize=10.5, color=DARK, loc="left", pad=8)

# ── RIGHT: three head attention heatmaps ──────────────────────────────────
features = ["MedInc","HouseAge","AveRooms","AveBedrms",
            "Pop","AveOccup","Lat","Long"]
rng = np.random.default_rng(7)

def softmax_rows(A):
    A = A - A.max(axis=1, keepdims=True)
    e = np.exp(A)
    return e / e.sum(axis=1, keepdims=True)

# Head 1: feature-family pairing (pairs: 0-0, 1-1, 2-3, 4-5, 6-7)
A1 = rng.normal(0, 0.3, size=(8, 8))
pairs = {0: [0], 1: [1], 2: [2, 3], 3: [2, 3], 4: [4, 5], 5: [4, 5], 6: [6, 7], 7: [6, 7]}
for i, js in pairs.items():
    for j in js:
        A1[i, j] += 3.0
head1 = softmax_rows(A1)

# Head 2: geography anchor (every row puts mass on Lat, Long)
A2 = rng.normal(0, 0.3, size=(8, 8))
for i in range(8):
    A2[i, 6] += 2.0
    A2[i, 7] += 1.8
head2 = softmax_rows(A2)

# Head 3: income anchor (every row puts mass on MedInc)
A3 = rng.normal(0, 0.3, size=(8, 8))
for i in range(8):
    A3[i, 0] += 2.6
head3 = softmax_rows(A3)

heads = [("Head 1 — feature-family", head1, ATTN1),
         ("Head 2 — geography anchor", head2, PURPLE),
         ("Head 3 — income anchor", head3, ORANGE)]

for k, (title, A, accent) in enumerate(heads):
    ax = fig.add_axes([0.52, 0.66 - k * 0.28, 0.45, 0.22])
    im = ax.imshow(A, cmap="viridis", vmin=0, vmax=0.6, aspect="auto")
    ax.set_yticks(range(8))
    ax.set_yticklabels(features, fontsize=6.2, color=DARK)
    # Only the bottom heatmap keeps x-tick labels; the upper two hide them
    # to avoid overlap with the next heatmap's title.
    if k == len(heads) - 1:
        ax.set_xticks(range(8))
        ax.set_xticklabels(features, fontsize=6.2, rotation=30,
                           ha="right", color=DARK)
    else:
        ax.set_xticks([])
    ax.set_title(title, fontsize=9.5, fontweight="bold",
                 color=accent, loc="left", pad=4)
    for spine in ax.spines.values(): spine.set_color(accent); spine.set_linewidth(1.2)
    if k == 0:
        ax.text(9.0, 0, "query", rotation=0, va="center",
                fontsize=6.5, color=GREY, ha="left")
        ax.text(3.5, -1.2, "key", va="bottom", ha="center",
                fontsize=6.5, color=GREY)

fig.text(0.745, 0.05,
         "Different heads specialise in different relationship patterns on the same input.",
         ha="center", va="top", fontsize=8.5, color=DARK, style="italic")

out = Path(__file__).resolve().parent / "ch18-multi-head-attention.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")
