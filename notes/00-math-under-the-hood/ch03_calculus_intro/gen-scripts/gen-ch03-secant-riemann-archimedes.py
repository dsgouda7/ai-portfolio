"""Generate static hero image for Pre-Requisites Ch.3 — Calculus Intro.

Three-panel figure spanning the two halves of the calculus idea:
  (left)   Secant to tangent: the secant slope between (t0, h(t0)) and (t0+h, h(t0+h))
           collapses to the tangent slope as h -> 0. Uses the free-throw trajectory
           so the apex (zero slope) is the physical motivation.
  (middle) Riemann sum: rectangles under the velocity curve v(t) = v0y - g*t
           accumulate to the displacement. Three overlays with n = 4, 16, 64
           rectangles show the sum converging to the integral.
  (right)  Archimedes' polygon-exhaustion of a circle — inscribed n-gons for
           n = 4, 8, 16, 64 filling the unit circle — the historical
           pre-Newton/Leibniz idea of "approximate curves with straight pieces".
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.6, 5.2), facecolor="white")
fig.suptitle("Ch.3 — Calculus in Two Pictures (and one history lesson)",
             fontsize=15, fontweight="bold", color=DARK, y=1.00)

# ── Panel 1: secant → tangent on the free-throw arc ───────────────────────
v0y, g = 7.2, 9.81
t = np.linspace(0, 1.47, 300)
h = v0y * t - 0.5 * g * t ** 2
ax1.plot(t, h, color=BLUE, lw=2.4, label=r"$h(t) = v_{0y}t - \frac{1}{2}g t^2$")

t0 = 0.35
h0 = v0y * t0 - 0.5 * g * t0 ** 2
ax1.scatter([t0], [h0], color=DARK, s=55, zorder=6)

# Secant lines for three decreasing h values
colors_secant = [RED, ORANGE, GOLD]
for dt, col in zip([0.6, 0.25, 0.08], colors_secant):
    t1 = t0 + dt
    h1 = v0y * t1 - 0.5 * g * t1 ** 2
    slope = (h1 - h0) / dt
    # draw secant across a wider range so lines are visible
    t_line = np.linspace(t0 - 0.1, t1 + 0.05, 50)
    h_line = h0 + slope * (t_line - t0)
    ax1.plot(t_line, h_line, color=col, lw=1.6, alpha=0.9,
             label=f"secant  Δt = {dt:.2f}   slope = {slope:+.2f}")
    ax1.scatter([t1], [h1], color=col, s=25, zorder=5)

# True tangent slope = h'(t0) = v0y - g*t0
tangent_slope = v0y - g * t0
t_tan = np.linspace(t0 - 0.25, t0 + 0.25, 30)
h_tan = h0 + tangent_slope * (t_tan - t0)
ax1.plot(t_tan, h_tan, color=DARK, lw=2.4, linestyle="--",
         label=f"tangent  (Δt → 0)   slope = {tangent_slope:+.2f}")

ax1.set_xlabel("time  t  (s)"); ax1.set_ylabel("height  h  (m)")
ax1.set_title("Secant → tangent as Δt → 0", fontsize=11.5,
              fontweight="bold", color=DARK, loc="left")
ax1.legend(loc="lower center", fontsize=7.8, framealpha=0.95)
ax1.grid(True, alpha=0.25)
ax1.set_xlim(-0.05, 1.55)

# ── Panel 2: Riemann sum of velocity → displacement ───────────────────────
t_end = 0.73   # integrate from 0 to t_end (ball's apex: v = 0 at t = v0y/g)
t_plot = np.linspace(0, t_end, 300)
v = v0y - g * t_plot
ax2.plot(t_plot, v, color=BLUE, lw=2.4, label=r"$v(t) = v_{0y} - g\,t$")
ax2.fill_between(t_plot, 0, v, alpha=0.14, color=BLUE)
ax2.axhline(0, color=GREY, lw=0.6)

# Overlay rectangles for n = 8 (visible); annotate sums for n = 4, 16, 64
# Use LEFT-endpoint rule so the under/over-estimate is visible for
# non-linear fitting later; for linear v(t) left-endpoint biases high.
n_show = 8
edges = np.linspace(0, t_end, n_show + 1)
widths = np.diff(edges)
left_pts = edges[:-1]
heights = v0y - g * left_pts
for xi, w, hi in zip(left_pts, widths, heights):
    ax2.bar(xi, hi, width=w, align="edge", alpha=0.35, color=ORANGE,
            edgecolor=ORANGE, linewidth=0.7)

def riemann_left(n):
    edges = np.linspace(0, t_end, n + 1)
    left_pts = edges[:-1]
    return float(np.sum((v0y - g * left_pts) * np.diff(edges)))

exact = v0y * t_end - 0.5 * g * t_end ** 2     # integral of v(t) from 0 to t_end
sum_text = (
    f"exact integral   = {exact:.4f}\n"
    f"left-rect n=4    = {riemann_left(4):.4f}\n"
    f"left-rect n=16   = {riemann_left(16):.4f}\n"
    f"left-rect n=64   = {riemann_left(64):.4f}"
)
ax2.text(0.03, 0.97, sum_text, transform=ax2.transAxes, fontsize=8.5, color=DARK,
         va="top", ha="left", family="monospace",
         bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF8E1",
                   edgecolor=GOLD, linewidth=0.9))
ax2.set_xlabel("time  t  (s)"); ax2.set_ylabel("velocity  v  (m/s)")
ax2.set_title("Riemann sum  →  integral", fontsize=11.5,
              fontweight="bold", color=DARK, loc="left")
ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
ax2.grid(True, alpha=0.25)

# ── Panel 3: Archimedes — inscribed polygons exhausting the circle ────────
theta = np.linspace(0, 2 * np.pi, 400)
ax3.plot(np.cos(theta), np.sin(theta), color=DARK, lw=2.0, label="unit circle")

poly_defs = [(4, RED, "--"), (8, ORANGE, "--"), (16, GREEN, "-"), (64, PURPLE, "-")]
for n, col, ls in poly_defs:
    ang = np.linspace(0, 2 * np.pi, n + 1)
    xs = np.cos(ang); ys = np.sin(ang)
    area = 0.5 * n * np.sin(2 * np.pi / n)
    ax3.plot(xs, ys, color=col, lw=1.5, linestyle=ls,
             label=f"n = {n:>2}   area ≈ {area:.5f}")

ax3.set_aspect("equal")
ax3.set_xlim(-1.3, 1.3); ax3.set_ylim(-1.3, 1.3)
ax3.set_title("Archimedes: polygons → π", fontsize=11.5,
              fontweight="bold", color=DARK, loc="left")
ax3.legend(loc="upper right", fontsize=7.8, framealpha=0.95)
ax3.grid(True, alpha=0.25)
ax3.text(0.03, 0.04, "true area π = 3.14159...",
         transform=ax3.transAxes, fontsize=8.5, color=DARK,
         va="bottom", ha="left", style="italic",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F8F5",
                   edgecolor=GREEN, linewidth=0.9))

fig.tight_layout(rect=(0, 0, 1, 0.96))
out = Path(__file__).resolve().parent / "ch03-secant-riemann-archimedes.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")
