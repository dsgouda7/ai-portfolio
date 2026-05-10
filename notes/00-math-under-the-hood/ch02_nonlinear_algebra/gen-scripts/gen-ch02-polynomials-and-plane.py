"""Generate static hero image for Pre-Requisites Ch.2 — Non-linear Algebra.

Three-panel figure:
  (left)   The full parabolic trajectory of a free throw, with the linear
           model from Ch.1 overlaid to show how badly a line fits a parabola.
  (middle) Sliders-at-rest illustration: one parabola with a, b, c annotated
           next to the geometric role of each coefficient.
  (right)  The feature-expansion trick. Left sub-panel: the 1-D parabola
           y = 3*x^2 - 2*x + 1. Right sub-panel: the same equation in 2-D
           feature space (x1 = x^2, x2 = x) shown as a flat plane — proving
           "non-linear in x = linear in (x1, x2)."
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
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

fig = plt.figure(figsize=(14.4, 5.4), facecolor="white")
fig.suptitle("Ch.2 — Polynomials and the Feature-Expansion Trick",
             fontsize=15, fontweight="bold", color=DARK, y=1.00)

# ── Panel 1: parabolic trajectory + (bad) linear fit ─────────────────────
ax1 = fig.add_subplot(1, 3, 1)
v0, theta = 7.5, np.radians(52)     # release speed & angle
v0x, v0y = v0 * np.cos(theta), v0 * np.sin(theta)
g = 9.81
t = np.linspace(0, 2 * v0y / g, 200)     # flight time
x_t = v0x * t
y_t = v0y * t - 0.5 * g * t ** 2
ax1.plot(x_t, y_t, color=BLUE, lw=2.4, label=r"$y = v_{0y}t - \frac{1}{2}gt^2$  (real arc)")

# Bad linear fit: least-squares of y vs x on the full trajectory
coef_line = np.polyfit(x_t, y_t, 1)
ax1.plot(x_t, np.polyval(coef_line, x_t), color=RED, lw=1.8, linestyle="--",
         label="best line  (still wrong)")

ax1.axhline(0, color=GREY, lw=0.6)
ax1.set_xlabel("horizontal distance  x  (m)")
ax1.set_ylabel("height  y  (m)")
ax1.set_title("A line cannot fit a parabola", fontsize=11.5,
              fontweight="bold", color=DARK, loc="left")
ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
ax1.grid(True, alpha=0.25)
ax1.text(0.02, 0.96,
         "Free throw:\n  release 7.5 m/s @ 52°\n  apex ≈ 1.77 m, range ≈ 5.7 m",
         transform=ax1.transAxes, fontsize=8.5, color=DARK,
         va="top", ha="left", style="italic")

# ── Panel 2: parabola y = a*x^2 + b*x + c with coefficients annotated ────
ax2 = fig.add_subplot(1, 3, 2)
xs = np.linspace(-3, 3, 200)
a_val, b_val, c_val = 0.8, -1.2, 1.0
ys = a_val * xs ** 2 + b_val * xs + c_val

ax2.plot(xs, ys, color=PURPLE, lw=2.5)
ax2.axhline(0, color=GREY, lw=0.6); ax2.axvline(0, color=GREY, lw=0.6)
ax2.scatter([0], [c_val], color=GOLD, s=70, zorder=5,
            label=f"y-intercept: c = {c_val}")
vertex_x = -b_val / (2 * a_val)
vertex_y = a_val * vertex_x ** 2 + b_val * vertex_x + c_val
ax2.scatter([vertex_x], [vertex_y], color=RED, s=70, zorder=5,
            label=f"vertex at x = −b/2a = {vertex_x:.3f}")

ax2.set_title(r"$y = a\,x^2 + b\,x + c$   (a=0.8, b=−1.2, c=1.0)",
              fontsize=11.5, fontweight="bold", color=DARK, loc="left")
ax2.set_xlabel("x"); ax2.set_ylabel("y")
ax2.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
ax2.grid(True, alpha=0.25)
ax2.text(0.02, 0.04,
         "a controls curvature (a>0 opens up)\n"
         "b shifts the vertex sideways\n"
         "c is the y-intercept",
         transform=ax2.transAxes, fontsize=8.5, color=DARK,
         va="bottom", ha="left", style="italic",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF8E1",
                   edgecolor=GOLD, linewidth=0.9))

# ── Panel 3: the feature-expansion trick, 3-D view ───────────────────────
ax3 = fig.add_subplot(1, 3, 3, projection="3d")
a3, b3, c3 = 3.0, -2.0, 1.0

# Grid over (x1, x2) where x1 = x^2, x2 = x, but draw as free variables
x_line = np.linspace(-1.5, 1.5, 120)
x1_curve = x_line ** 2                 # constrained: x1 = x2^2
x2_curve = x_line
y_curve = a3 * x1_curve + b3 * x2_curve + c3
ax3.plot(x1_curve, x2_curve, y_curve, color=BLUE, lw=2.6,
         label="curve in 1-D = parabola", zorder=4)

# Plane: y = a*x1 + b*x2 + c  (flat in feature space)
X1, X2 = np.meshgrid(np.linspace(0, 2.3, 20), np.linspace(-1.5, 1.5, 20))
Y_plane = a3 * X1 + b3 * X2 + c3
ax3.plot_surface(X1, X2, Y_plane, alpha=0.32, color=PURPLE, edgecolor="none")

ax3.set_xlabel(r"$x_1 = x^2$", fontsize=9, labelpad=2)
ax3.set_ylabel(r"$x_2 = x$",   fontsize=9, labelpad=2)
ax3.set_zlabel("y",            fontsize=9, labelpad=2)
ax3.set_title(r"Non-linear in $x$  $\equiv$  linear in $(x_1, x_2)$",
              fontsize=11.5, fontweight="bold", color=DARK, loc="left")
ax3.tick_params(labelsize=7.5)
ax3.view_init(elev=22, azim=-52)
ax3.text2D(0.02, 0.94,
           r"$y = 3\,x^2 - 2\,x + 1$" + "\n" +
           r"$\;= 3\,x_1 - 2\,x_2 + 1$" + "\n" +
           "same numbers, flat plane",
           transform=ax3.transAxes, fontsize=8.5, color=DARK,
           va="top", ha="left",
           bbox=dict(boxstyle="round,pad=0.3", facecolor="#F4ECF7",
                     edgecolor=PURPLE, linewidth=0.9))

fig.tight_layout(rect=(0, 0, 1, 0.96))
out = Path(__file__).resolve().parent / "ch02-polynomials-and-plane.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")
