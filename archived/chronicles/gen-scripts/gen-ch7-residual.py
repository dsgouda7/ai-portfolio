"""Generate Reference/img/ch7-residual-block.png — correct residual block.

The identity skip connection originates from the block's INPUT x and is added
at the '+' node immediately BEFORE the final activation.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

BLUE = "#2E86C1"
ORANGE = "#E67E22"
GREEN = "#27AE60"
DARK = "#2C3E50"
GREY = "#7F8C8D"

fig, ax = plt.subplots(figsize=(7.5, 9), facecolor="white")
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis("off")
ax.set_title("Residual block — identity skip bypasses the two conv stages",
             fontsize=12, fontweight="bold", color=DARK, pad=10)

def box(xy, w, h, color, label, sub=None, text_color="white"):
    x, y = xy
    ax.add_patch(mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                         boxstyle="round,pad=0.04,rounding_size=0.12",
                                         facecolor=color, edgecolor=DARK, lw=1.4))
    ax.text(x, y + (0.18 if sub else 0), label, ha="center", va="center",
            fontsize=11, fontweight="bold", color=text_color)
    if sub:
        ax.text(x, y - 0.26, sub, ha="center", va="center",
                fontsize=8, color=text_color, alpha=0.9)

def arrow(p1, p2, color=DARK, lw=1.6, ls="-"):
    ax.annotate("", xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle="->", color=color, lw=lw, linestyle=ls,
                                shrinkA=4, shrinkB=4))

# Main path (centre line x=5)
CX = 5.0
# Input x node
ax.add_patch(plt.Circle((CX, 11), 0.35, facecolor="white", edgecolor=DARK, lw=1.6))
ax.text(CX, 11, "x", ha="center", va="center", fontsize=14, fontweight="bold", color=DARK)

# Conv1 + BN + ReLU
box((CX, 9.2), 3.2, 0.9, BLUE, "Conv 3×3", "BN → ReLU")
# Conv2 + BN
box((CX, 7.4), 3.2, 0.9, BLUE, "Conv 3×3", "BN (no ReLU yet)")

# Plus node
ax.add_patch(plt.Circle((CX, 5.6), 0.35, facecolor="white", edgecolor=DARK, lw=1.6))
ax.text(CX, 5.6, "+", ha="center", va="center", fontsize=16, fontweight="bold", color=DARK)

# Final ReLU
box((CX, 4.0), 2.4, 0.8, GREEN, "ReLU")
# Output
ax.add_patch(plt.Circle((CX, 2.4), 0.38, facecolor="white", edgecolor=DARK, lw=1.6))
ax.text(CX, 2.4, "y", ha="center", va="center", fontsize=14, fontweight="bold", color=DARK)

# Main arrows
arrow((CX, 10.65), (CX, 9.68))
arrow((CX, 8.72), (CX, 7.88))
arrow((CX, 6.92), (CX, 5.98))
arrow((CX, 5.22), (CX, 4.42))
arrow((CX, 3.58), (CX, 2.78))

# Identity skip: from x node, out to the side, down, back to +
skip_color = ORANGE
ax.annotate("", xy=(CX + 2.2, 11), xytext=(CX + 0.35, 11),
            arrowprops=dict(arrowstyle="-", color=skip_color, lw=2.2))
ax.annotate("", xy=(CX + 2.2, 5.6), xytext=(CX + 2.2, 11),
            arrowprops=dict(arrowstyle="-", color=skip_color, lw=2.2))
ax.annotate("", xy=(CX + 0.35, 5.6), xytext=(CX + 2.2, 5.6),
            arrowprops=dict(arrowstyle="->", color=skip_color, lw=2.2))
ax.text(CX + 2.4, 8.3, "identity  x\n(skip connection)",
        color=skip_color, fontsize=10, fontweight="bold", va="center")

# Equation annotation
ax.text(CX - 2.5, 5.6, r"$y = \mathrm{ReLU}(\,\mathcal{F}(x) + x\,)$",
        color=DARK, fontsize=11, ha="right", va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F8F9F9",
                  edgecolor=GREY, lw=0.8))

# Caption
ax.text(5, 0.6,
        "The skip originates from x (not from F₁(x)) and is added just before the final ReLU.\n"
        "If shape mismatches, replace identity with a 1×1 conv (projection shortcut).",
        ha="center", va="center", fontsize=9, color="#555")

out = Path(__file__).resolve().parent / "ch7-residual-block.png"
fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")
