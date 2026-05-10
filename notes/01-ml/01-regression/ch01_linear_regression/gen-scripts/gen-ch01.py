"""Generate ch01 Gradient Descent Paths.png — regression scatter + GD paths.

Uses a single marker style for the regression scatter (no binary-classification
reading) and overlays three gradient-descent trajectories (small, good, large η).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BLUE = "#2E86C1"
GREEN = "#27AE60"
ORANGE = "#E67E22"
RED = "#C0392B"
DARK = "#2C3E50"
GREY = "#7F8C8D"

rng = np.random.default_rng(2)

# --- synthetic 1-D regression data ---
n = 80
x = rng.uniform(0, 10, n)
true_w, true_b = 1.8, 1.0
y = true_w * x + true_b + rng.normal(0, 1.6, n)

fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), facecolor="white")
fig.suptitle("Linear regression — data fit and gradient-descent paths",
             fontsize=13, fontweight="bold", color=DARK, y=0.99)

# ── Panel 1: regression scatter + fitted line ────────────────────
ax = axes[0]
ax.scatter(x, y, color=BLUE, alpha=0.65, s=36, edgecolors="white", linewidths=0.6,
           label="training samples")
# closed-form fit
w_hat, b_hat = np.polyfit(x, y, 1)
xx = np.linspace(0, 10, 100)
ax.plot(xx, w_hat * xx + b_hat, color=RED, lw=2.2,
        label=f"fitted: y = {w_hat:.2f}·x + {b_hat:.2f}")
ax.set_title("Continuous target y  (one marker style)",
             fontsize=11, fontweight="bold", color=DARK)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(loc="upper left", fontsize=9, frameon=True)
ax.grid(True, alpha=0.3)

# ── Panel 2: loss landscape + GD paths ───────────────────────────
ax = axes[1]
# Use normalized x so the loss surface is near-isotropic and GD is stable to plot.
x_n = (x - x.mean()) / x.std()
# True params in normalised space
true_w_n = 1.8 * x.std()
true_b_n = 1.8 * x.mean() + 1.0

W = np.linspace(true_w_n - 4, true_w_n + 4, 120)
B = np.linspace(true_b_n - 6, true_b_n + 6, 120)
WW, BB = np.meshgrid(W, B)

def loss(w, b):
    y_hat = w * x_n[:, None, None] + b
    return ((y_hat - y[:, None, None]) ** 2).mean(axis=0)

Z = loss(WW, BB)
ax.contour(WW, BB, Z, levels=18, cmap="Greys", linewidths=0.8, alpha=0.7)
ax.contourf(WW, BB, Z, levels=18, cmap="Blues", alpha=0.35)

def gd(lr, steps=80, start=None):
    if start is None:
        start = (true_w_n - 3.5, true_b_n + 5)
    w, b = start
    path = [(w, b)]
    for _ in range(steps):
        y_hat = w * x_n + b
        err = y_hat - y
        dw = 2 * (err * x_n).mean()
        db = 2 * err.mean()
        w -= lr * dw
        b -= lr * db
        path.append((w, b))
    return np.array(path)

paths = [
    (gd(0.01, start=(true_w_n - 3.2, true_b_n + 5.2)),
     GREEN,  "η = 0.01  (too small — slow)"),
    (gd(0.18, start=(true_w_n + 2.8, true_b_n + 5.0)),
     ORANGE, "η = 0.18  (just right)"),
    (gd(0.97, start=(true_w_n - 0.6, true_b_n + 5.5)),
     RED,    "η = 0.97  (too large — oscillates)"),
]
for p, c, lbl in paths:
    # clip the path to the displayed window so a divergent run stays visible
    p = np.clip(p, [W.min(), B.min()], [W.max(), B.max()])
    ax.plot(p[:, 0], p[:, 1], color=c, lw=2.0, marker="o",
            ms=3.5, markeredgecolor="white", markeredgewidth=0.4, label=lbl)
    ax.plot(p[0, 0], p[0, 1], marker="s", ms=9, color=c,
            markeredgecolor=DARK, markeredgewidth=1.0)

ax.plot(true_w_n, true_b_n, marker="*", ms=20, color="gold",
        markeredgecolor=DARK, markeredgewidth=1.2, label="true (w*, b*)")

ax.set_xlim(W.min(), W.max())
ax.set_ylim(B.min(), B.max())
ax.set_title("MSE loss surface with gradient-descent trajectories\n(x normalised)",
             fontsize=11, fontweight="bold", color=DARK)
ax.set_xlabel("weight  w")
ax.set_ylabel("bias    b")
ax.legend(loc="upper right", fontsize=8, frameon=True)

fig.tight_layout(rect=(0, 0, 1, 0.95))

out = Path(__file__).resolve().parent / "Gradient Descent Paths.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")
