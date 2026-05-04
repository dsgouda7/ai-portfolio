"""Generate static hero image for Pre-Requisites Ch.1 — Linear Algebra.

Four-panel figure introducing the line y = w*x + b:
  (top-left)   A family of lines varying only w (slope), b fixed at 0.
  (top-right)  A family of lines varying only b (bias), w fixed at 1.
  (bot-left)   Two vectors in R^2 and their dot product as a weighted sum.
  (bot-right)  Free-throw running example: first 0.2 s of ball rise,
               overlaid with a best-guess linear fit h(t) = w*t + b.
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
RED    = "#E74C3C"
GREY   = "#7F8C8D"
GOLD   = "#F39C12"

fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.2), facecolor="white")
fig.suptitle("Ch.1 — Lines, Weights, and Biases",
             fontsize=15, fontweight="bold", color=DARK, y=0.995)

# ── Top-left: varying slope w ────────────────────────────────────────────
ax = axes[0, 0]
x = np.linspace(-4, 4, 200)
slopes = [-1.5, -0.5, 0.5, 1.5, 3.0]
for w in slopes:
    ax.plot(x, w * x, linewidth=2.0, label=f"w = {w:+.1f}", alpha=0.9)
ax.axhline(0, color=GREY, linewidth=0.6); ax.axvline(0, color=GREY, linewidth=0.6)
ax.set_title("Weight w rotates the line (b = 0)",
             fontsize=11.5, fontweight="bold", color=DARK, loc="left")
ax.set_xlabel("x"); ax.set_ylabel("y = w·x")
ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
ax.grid(True, alpha=0.25); ax.set_xlim(-4, 4); ax.set_ylim(-6, 6)

# ── Top-right: varying bias b ────────────────────────────────────────────
ax = axes[0, 1]
biases = [-3, -1, 0, 1, 3]
for b in biases:
    ax.plot(x, 1.0 * x + b, linewidth=2.0, label=f"b = {b:+d}", alpha=0.9)
ax.axhline(0, color=GREY, linewidth=0.6); ax.axvline(0, color=GREY, linewidth=0.6)
ax.set_title("Bias b shifts the line up/down (w = 1)",
             fontsize=11.5, fontweight="bold", color=DARK, loc="left")
ax.set_xlabel("x"); ax.set_ylabel("y = x + b")
ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
ax.grid(True, alpha=0.25); ax.set_xlim(-4, 4); ax.set_ylim(-6, 6)

# ── Bottom-left: dot product as weighted sum ─────────────────────────────
ax = axes[1, 0]
a = np.array([3.0, 1.0])
b_vec = np.array([2.0, 2.5])
ax.annotate("", xy=a, xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.2))
ax.annotate("", xy=b_vec, xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.2))
ax.text(a[0] + 0.12, a[1] - 0.15, r"$\mathbf{a}=[3,\,1]$",
        fontsize=10.5, color=BLUE, fontweight="bold")
ax.text(b_vec[0] + 0.1, b_vec[1] + 0.1, r"$\mathbf{b}=[2,\,2.5]$",
        fontsize=10.5, color=ORANGE, fontweight="bold")
dot = float(np.dot(a, b_vec))
ax.text(0.5, 0.05,
        r"$\mathbf{a}\cdot\mathbf{b}=3\!\cdot\!2 + 1\!\cdot\!2.5 = "
        + f"{dot:.1f}" + r"$",
        transform=ax.transAxes, fontsize=11, color=DARK,
        ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF8E1",
                  edgecolor=GOLD, linewidth=1.0))
ax.text(0.98, 0.96,
        "Dot product = weighted sum.\nEvery neuron computes one.",
        transform=ax.transAxes, fontsize=9.5, color=DARK,
        va="top", ha="right", style="italic")
ax.set_xlim(-0.5, 4); ax.set_ylim(-0.5, 3.5)
ax.axhline(0, color=GREY, linewidth=0.6); ax.axvline(0, color=GREY, linewidth=0.6)
ax.set_title("Vectors and the dot product",
             fontsize=11.5, fontweight="bold", color=DARK, loc="left")
ax.set_xlabel("x-component"); ax.set_ylabel("y-component")
ax.grid(True, alpha=0.25); ax.set_aspect("equal")

# ── Bottom-right: linear fit to first 0.2 s of free-throw ────────────────
ax = axes[1, 1]
rng = np.random.default_rng(17)
t = np.linspace(0, 0.20, 9)
# Ideal: h(t) = v0y * t - 0.5 * g * t^2 ; for small t, nearly linear in t.
v0y, g = 7.2, 9.81
h_true = v0y * t - 0.5 * g * t ** 2
h_obs = h_true + rng.normal(0, 0.015, size=t.shape)
ax.scatter(t, h_obs, s=55, color=DARK, zorder=5,
           label="Recorded release (first 0.2 s)")

# Overlay two candidate lines
for w, b, c, lab in [(v0y, 0.0, GREEN, r"$h(t)=7.2\,t$  (good fit)"),
                     (5.5, 0.1, RED,   r"$h(t)=5.5\,t + 0.1$  (bad fit)")]:
    ax.plot(t, w * t + b, color=c, linewidth=2.2, label=lab)

ax.set_title("Free-throw, first 0.2 s — fit a line by eye",
             fontsize=11.5, fontweight="bold", color=DARK, loc="left")
ax.set_xlabel("time  t  (s)"); ax.set_ylabel("height  h  (m)")
ax.legend(fontsize=9, loc="upper left", framealpha=0.92)
ax.grid(True, alpha=0.25); ax.set_xlim(-0.005, 0.21)
ax.text(0.98, 0.04,
        "Pick (w, b) to minimise total\nmiss — a Ch.3/Ch.4 question.",
        transform=ax.transAxes, fontsize=9, color=GREY,
        ha="right", va="bottom", style="italic")

fig.tight_layout(rect=(0, 0, 1, 0.97))
out = Path(__file__).resolve().parent / "ch01-lines-and-biases.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")
