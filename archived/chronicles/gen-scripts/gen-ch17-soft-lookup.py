"""Generate Reference/img/ch17-soft-lookup.png — hard dict vs soft attention lookup.

Two-panel figure illustrating the core mental model of Ch.17:
  (left)   HARD dict: a query matches exactly one key → returns exactly one value.
  (right)  SOFT attention: query compared to all keys → softmax → weighted blend.

The 8 housing features (MedInc, HouseAge, …) are used as a concrete example.
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
ORANGE = "#E67E22"
RED    = "#E74C3C"
PURPLE = "#8E44AD"
GREY   = "#95A5A6"
GOLD   = "#F39C12"
ATTN   = "#0E7490"

features = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
            "Population", "AveOccup", "Latitude", "Longitude"]

# Soft-lookup weights (hand-picked, softmax-normalised — pedagogical)
scores = np.array([3.2, 0.8, 2.1, 1.7, 0.2, 1.0, 0.4, 0.6])
weights = np.exp(scores) / np.exp(scores).sum()

fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4), facecolor="white")
fig.suptitle("Hard dict lookup vs soft attention lookup — one picture, one idea",
             fontsize=13, fontweight="bold", color=DARK, y=0.995)

# ── LEFT: HARD Python dict ────────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor("#FBFCFE")
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_color("#E0E4EA")
ax.set_title("HARD  —  Python dict", fontsize=12, fontweight="bold",
             color=DARK, loc="left", pad=6)

# Query box
ax.add_patch(mpatches.FancyBboxPatch((0.4, 8.2), 2.6, 1.2,
    boxstyle="round,pad=0.06,rounding_size=0.15",
    facecolor="#FFF8E1", edgecolor=GOLD, linewidth=1.6))
ax.text(1.7, 8.8, "query", ha="center", va="center",
        fontsize=10, fontweight="bold", color=DARK)
ax.text(1.7, 8.4, '"MedInc"', ha="center", va="center",
        fontsize=9, color=DARK, family="monospace")

# Dict table
dict_y0 = 1.6
row_h = 0.55
for i, (name, val) in enumerate(zip(features, np.arange(8))):
    y = dict_y0 + (7 - i) * row_h
    fc = "#E3F2FD" if i == 0 else "white"
    ec = BLUE if i == 0 else "#CFD8DC"
    lw = 2.0 if i == 0 else 0.8
    ax.add_patch(mpatches.Rectangle((4.2, y), 4.8, row_h,
        facecolor=fc, edgecolor=ec, linewidth=lw))
    ax.text(5.0, y + row_h / 2, name, ha="left", va="center",
            fontsize=9, color=DARK, family="monospace")
    ax.text(8.6, y + row_h / 2, f"v{i}", ha="right", va="center",
            fontsize=9, color=GREY, family="monospace",
            fontweight="bold" if i == 0 else "normal")

# Arrow from query to matching row
ax.annotate("", xy=(4.2, 5.45), xytext=(3.0, 8.2),
            arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.8,
                            connectionstyle="arc3,rad=0.25"))
ax.text(3.2, 6.9, "exact\nmatch", ha="left", va="center",
        fontsize=9, color=GOLD, fontweight="bold")

# Output
ax.add_patch(mpatches.FancyBboxPatch((4.2, 0.3), 4.8, 0.95,
    boxstyle="round,pad=0.05,rounding_size=0.12",
    facecolor="#E8F5E9", edgecolor=GREEN, linewidth=1.6))
ax.text(6.6, 0.77, "output  =  v0", ha="center", va="center",
        fontsize=10.5, fontweight="bold", color=GREEN, family="monospace")
ax.annotate("", xy=(6.6, 1.28), xytext=(6.6, 1.56),
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.3))

ax.text(0.4, 0.6, "• one key matches\n• one value returned\n• not differentiable",
        fontsize=9, color=DARK, va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F4F6F9",
                  edgecolor="#CFD8DC", linewidth=0.9))

# ── RIGHT: SOFT attention ────────────────────────────────────────────────
ax = axes[1]
ax.set_facecolor("#FBFCFE")
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_color("#E0E4EA")
ax.set_title("SOFT  —  attention", fontsize=12, fontweight="bold",
             color=ATTN, loc="left", pad=6)

# Query box
ax.add_patch(mpatches.FancyBboxPatch((0.4, 8.2), 2.6, 1.2,
    boxstyle="round,pad=0.06,rounding_size=0.15",
    facecolor="#E0F2FE", edgecolor=ATTN, linewidth=1.6))
ax.text(1.7, 8.8, "query  q", ha="center", va="center",
        fontsize=10, fontweight="bold", color=ATTN)
ax.text(1.7, 8.4, "(vector in ℝᵈ)", ha="center", va="center",
        fontsize=8.5, color=DARK)

# Key/value rows with weight bars
row_y0 = 1.6
row_h = 0.55
max_bar_w = 3.4
for i, name in enumerate(features):
    y = row_y0 + (7 - i) * row_h
    # Key cell
    ax.add_patch(mpatches.Rectangle((3.4, y), 1.5, row_h * 0.9,
        facecolor="white", edgecolor="#CFD8DC", linewidth=0.8))
    ax.text(4.15, y + row_h / 2, name, ha="center", va="center",
            fontsize=7.8, color=DARK, family="monospace")
    # Weight bar
    bw = max_bar_w * weights[i]
    ax.add_patch(mpatches.Rectangle((5.05, y + 0.06), bw, row_h * 0.78,
        facecolor=ATTN, alpha=0.25 + 0.70 * weights[i] / weights.max(),
        edgecolor=ATTN, linewidth=0.7))
    ax.text(5.1 + bw + 0.1, y + row_h / 2, f"{weights[i]:.2f}",
            ha="left", va="center", fontsize=8, color=DARK,
            fontweight="bold" if weights[i] > 0.15 else "normal")

# "softmax(q · k_i)" label
ax.text(5.95, 6.55, "weights  wᵢ = softmax(q · kᵢ)",
        ha="center", va="bottom", fontsize=8.5, color=ATTN,
        fontweight="bold")

# Arrow from query to compare step
ax.annotate("", xy=(3.4, 5.45), xytext=(3.0, 8.2),
            arrowprops=dict(arrowstyle="->", color=ATTN, lw=1.6,
                            connectionstyle="arc3,rad=0.25"))
ax.text(2.9, 6.85, "compare to\nALL keys",
        ha="left", va="center", fontsize=8.5, color=ATTN,
        fontweight="bold")

# Output — weighted sum
ax.add_patch(mpatches.FancyBboxPatch((0.4, 0.3), 9.2, 0.95,
    boxstyle="round,pad=0.05,rounding_size=0.12",
    facecolor="#EAFAF1", edgecolor=GREEN, linewidth=1.6))
ax.text(5.0, 0.77,
        r"output  =  $\sum_i w_i \cdot v_i$   (weighted blend of ALL values)",
        ha="center", va="center", fontsize=10.5,
        fontweight="bold", color=GREEN)
ax.annotate("", xy=(5.0, 1.28), xytext=(5.0, 1.56),
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.3))

# Footer caption on soft side
ax.text(0.15, 9.72,
        "• every key scored  • differentiable (softmax)  • output is a BLEND, not a single value",
        fontsize=8.2, color=ATTN,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#E0F2FE",
                  edgecolor=ATTN, linewidth=0.8))

fig.tight_layout(rect=(0, 0, 1, 0.95))
out = Path(__file__).resolve().parent / "ch17-soft-lookup.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")
