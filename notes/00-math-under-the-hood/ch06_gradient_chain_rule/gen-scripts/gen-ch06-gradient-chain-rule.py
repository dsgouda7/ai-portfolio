"""Ch.6 hero image: gradient field, chain-rule diagram, descent trajectory."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

DARK, BLUE, GREEN, PURPLE = "#2C3E50", "#2E86C1", "#27AE60", "#8E44AD"
ORANGE, RED, GREY, GOLD = "#E67E22", "#E74C3C", "#7F8C8D", "#F39C12"

OUT = Path(__file__).parent / "ch06-gradient-chain-rule.png"

fig, axes = plt.subplots(1, 3, figsize=(14.8, 5.2))
fig.patch.set_facecolor("white")

# ---------- Panel 1: 2-D loss surface + gradient field ----------
ax = axes[0]
X, Y = np.meshgrid(np.linspace(-2.5, 2.5, 200), np.linspace(-2.0, 2.0, 200))
# Tilted ellipsoidal bowl — the canonical quadratic loss.
Z = 0.5 * (1.0 * X**2 + 3.0 * Y**2 + 1.2 * X * Y)
cf = ax.contourf(X, Y, Z, levels=18, cmap="Blues_r", alpha=0.9)
ax.contour(X, Y, Z, levels=18, colors=GREY, linewidths=0.4, alpha=0.6)

# Gradient arrows on a coarse grid (pointing uphill; we negate for descent).
Xc, Yc = np.meshgrid(np.linspace(-2.2, 2.2, 10), np.linspace(-1.8, 1.8, 8))
Gx = 1.0 * Xc + 0.6 * Yc
Gy = 3.0 * Yc + 0.6 * Xc
ax.quiver(Xc, Yc, -Gx, -Gy, color=DARK, alpha=0.75,
          scale=60, width=0.005, headwidth=4)
ax.plot(0, 0, "o", color=RED, markersize=11, zorder=5)
ax.annotate(r"$\theta^\star$", xy=(0, 0), xytext=(0.25, 0.15),
            fontsize=14, color=RED, fontweight="bold")

ax.set_title(r"$-\nabla f(\theta)$: every arrow is a downhill step",
             fontsize=11, color=DARK)
ax.set_xlabel(r"$\theta_1$"); ax.set_ylabel(r"$\theta_2$")
ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.0, 2.0); ax.set_aspect("equal")

# ---------- Panel 2: chain-rule computation graph ----------
ax = axes[1]
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
ax.set_title("Matrix chain rule: dimensions flow left-to-right",
             fontsize=11, color=DARK)

def box(x, y, w, h, text, fc, ec=DARK):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                       fc=fc, ec=ec, lw=1.6)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=11, color=DARK)

# Forward pass
box(0.3, 3.6, 1.6, 1.0, r"$x$" "\n" r"$(n)$",        "#D6EAF8")
box(2.7, 3.6, 1.8, 1.0, r"$u=Wx$" "\n" r"$(m)$",    "#D5F5E3")
box(5.3, 3.6, 1.8, 1.0, r"$h=\sigma(u)$" "\n" r"$(m)$", "#FDEBD0")
box(7.9, 3.6, 1.8, 1.0, r"$L(h)$" "\n" "scalar",    "#FADBD8")

for x0, x1 in [(1.9, 2.7), (4.5, 5.3), (7.1, 7.9)]:
    ax.add_patch(FancyArrowPatch((x0, 4.1), (x1, 4.1),
                                 arrowstyle="->", mutation_scale=14,
                                 color=DARK, lw=1.4))
ax.text(5.0, 5.0, "forward", ha="center", fontsize=10, color=GREEN,
        fontweight="bold")

# Backward pass — gradients/Jacobians flow right-to-left
box(7.9, 1.6, 1.8, 1.0, r"$\frac{\partial L}{\partial h}$" "\n" r"$(m)$", "#FADBD8")
box(5.3, 1.6, 1.8, 1.0, r"$\mathrm{diag}(\sigma')$" "\n" r"$(m{\times}m)$", "#FDEBD0")
box(2.7, 1.6, 1.8, 1.0, r"$W^\top$" "\n" r"$(n{\times}m)$", "#D5F5E3")
box(0.3, 1.6, 1.6, 1.0, r"$\frac{\partial L}{\partial x}$" "\n" r"$(n)$", "#D6EAF8")

for x0, x1 in [(7.9, 7.1), (5.3, 4.5), (2.7, 1.9)]:
    ax.add_patch(FancyArrowPatch((x0, 2.1), (x1, 2.1),
                                 arrowstyle="->", mutation_scale=14,
                                 color=RED, lw=1.4))
ax.text(5.0, 0.8, "backward = multiply Jacobians right-to-left",
        ha="center", fontsize=10, color=RED, fontweight="bold")

# ---------- Panel 3: descent trajectories with & without proper step ----------
ax = axes[2]
ax.contour(X, Y, Z, levels=15, colors=GREY, linewidths=0.5, alpha=0.6)
ax.contourf(X, Y, Z, levels=15, cmap="Blues_r", alpha=0.35)

def descend(theta0, eta, n=25):
    th = np.array(theta0, dtype=float)
    path = [th.copy()]
    for _ in range(n):
        g = np.array([1.0 * th[0] + 0.6 * th[1],
                      3.0 * th[1] + 0.6 * th[0]])
        th = th - eta * g
        path.append(th.copy())
        if np.linalg.norm(th) > 10:
            break
    return np.array(path)

# Good step size — smooth descent
p_good = descend([-2.2, 1.6], eta=0.18, n=40)
ax.plot(p_good[:, 0], p_good[:, 1], "-o", color=GREEN, lw=1.8,
        markersize=3.5, label=r"$\eta=0.18$ (good)")

# Too big — zigzags across the narrow valley
p_big = descend([2.2, 1.6], eta=0.55, n=25)
ax.plot(p_big[:, 0], p_big[:, 1], "-o", color=ORANGE, lw=1.8,
        markersize=3.5, label=r"$\eta=0.55$ (zigzag)")

# Too small — creeps
p_slow = descend([-1.5, -1.6], eta=0.04, n=40)
ax.plot(p_slow[:, 0], p_slow[:, 1], "-o", color=PURPLE, lw=1.8,
        markersize=3.5, label=r"$\eta=0.04$ (slow)")

ax.plot(0, 0, "*", color=RED, markersize=16, zorder=6)
ax.set_title(r"Walk in the direction of $-\nabla f$; step size matters",
             fontsize=11, color=DARK)
ax.set_xlabel(r"$\theta_1$"); ax.set_ylabel(r"$\theta_2$")
ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.0, 2.0); ax.set_aspect("equal")
ax.legend(loc="upper right", fontsize=8, framealpha=0.95)

plt.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}")
