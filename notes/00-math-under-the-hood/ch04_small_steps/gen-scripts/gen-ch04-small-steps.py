"""Generate static hero image for Pre-Requisites Ch.4 — Small Steps on a Curve.

Three panels telling the iterative-optimisation story on the free-throw problem:

  (left)   Range vs release-angle for a free throw in a vacuum. R(theta)
           = v0^2 sin(2 theta) / g peaks sharply at 45 degrees. We draw
           three descent trajectories starting from 20°, 65°, and 80°, all
           converging to the same optimum — the "nice convex" case.

  (middle) Step-size matters. Same curve, three step sizes:
              eta too small  -> crawls
              eta just right -> converges cleanly
              eta too large  -> overshoots and oscillates
           Plotted as the sequence of iterates on the curve.

  (right)  The non-convex villain. Release into a headwind makes the
           range curve bumpy; a bad starting angle converges to a LOCAL
           maximum, missing the global one. Foreshadows the deep-learning
           optimisation headache.
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

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.8, 5.2), facecolor="white")
fig.suptitle("Ch.4 — Optimise by Walking Downhill, One Small Step at a Time",
             fontsize=15, fontweight="bold", color=DARK, y=1.00)

# ── physics ───────────────────────────────────────────────────────────────
g = 9.81
v0 = 8.0                                # release speed m/s
theta_grid = np.linspace(5, 85, 400)    # degrees
R_vacuum = v0 ** 2 * np.sin(2 * np.radians(theta_grid)) / g    # metres

def R_fn(theta_deg):
    return v0 ** 2 * np.sin(2 * np.radians(theta_deg)) / g

def dR_dtheta(theta_deg):
    # d/dθ [ v0²/g · sin(2θ) ] in degrees-space
    # Use chain rule with 2 * cos(2θ) * (π/180)
    return v0 ** 2 / g * 2 * np.cos(2 * np.radians(theta_deg)) * (np.pi / 180)

# ── Panel 1: three descents converging to 45° ─────────────────────────────
ax1.plot(theta_grid, R_vacuum, color=BLUE, lw=2.4,
         label=r"$R(\theta) = \frac{v_0^2}{g}\sin(2\theta)$")
ax1.axvline(45, color=GREEN, lw=1.0, linestyle=":", alpha=0.7)
ax1.scatter([45], [R_fn(45)], color=GREEN, s=65, zorder=6, label="optimum (45°)")

starts = [(20.0, RED, "start 20°"),
          (65.0, ORANGE, "start 65°"),
          (80.0, PURPLE, "start 80°")]
eta = 35.0
for start, col, lbl in starts:
    theta = start
    traj_x, traj_y = [theta], [R_fn(theta)]
    for _ in range(30):
        theta = theta + eta * dR_dtheta(theta)     # ASCENT: +gradient (maximise)
        theta = np.clip(theta, 5, 85)
        traj_x.append(theta); traj_y.append(R_fn(theta))
    ax1.plot(traj_x, traj_y, color=col, lw=1.4, alpha=0.85, marker="o", markersize=3.5,
             label=lbl)

ax1.set_xlabel("release angle  θ  (degrees)")
ax1.set_ylabel("horizontal range  R  (m)")
ax1.set_title("Convex case — all starts converge", fontsize=11.5,
              fontweight="bold", color=DARK, loc="left")
ax1.legend(loc="lower center", fontsize=8.5, framealpha=0.95)
ax1.grid(True, alpha=0.25)

# ── Panel 2: step-size three-way comparison ───────────────────────────────
ax2.plot(theta_grid, R_vacuum, color=BLUE, lw=2.0, alpha=0.7)
ax2.axvline(45, color=GREEN, lw=1.0, linestyle=":", alpha=0.6)

variants = [
    ("η too small  (2)",   2.0,  RED,    20.0),
    ("η just right (35)",  35.0, GREEN,  20.0),
    ("η too large  (180)", 180.0, ORANGE, 20.0),
]
for lbl, eta_v, col, start in variants:
    theta = start
    xs, ys = [theta], [R_fn(theta)]
    for _ in range(18):
        theta = theta + eta_v * dR_dtheta(theta)
        theta = np.clip(theta, 5, 85)
        xs.append(theta); ys.append(R_fn(theta))
    ax2.plot(xs, ys, color=col, lw=1.4, alpha=0.9, marker="o", markersize=3.2,
             label=lbl)

ax2.set_xlabel("release angle  θ  (degrees)")
ax2.set_ylabel("range  R  (m)")
ax2.set_title("Step size decides everything", fontsize=11.5,
              fontweight="bold", color=DARK, loc="left")
ax2.legend(loc="lower center", fontsize=8.5, framealpha=0.95)
ax2.grid(True, alpha=0.25)

# ── Panel 3: non-convex — local vs global optimum ─────────────────────────
# Synthetic bumpy landscape: base vacuum range minus a wind-penalty bump
def R_windy(theta_deg):
    base = R_fn(theta_deg)
    bump = 1.2 * np.exp(-((theta_deg - 60) / 8) ** 2)
    return base - bump + 0.9 * np.exp(-((theta_deg - 30) / 6) ** 2)

R_bumpy = R_windy(theta_grid)
ax3.plot(theta_grid, R_bumpy, color=PURPLE, lw=2.4, label=r"$R_{\mathrm{wind}}(\theta)$")

# Find global optimum numerically
global_theta = theta_grid[int(np.argmax(R_bumpy))]
ax3.axvline(global_theta, color=GREEN, lw=1.0, linestyle=":", alpha=0.7)
ax3.scatter([global_theta], [R_windy(global_theta)], color=GREEN, s=65, zorder=6,
            label=f"global max (≈{global_theta:.0f}°)")

# Finite-difference gradient for the numerical landscape
def dR_windy(theta_deg, h=0.05):
    return (R_windy(theta_deg + h) - R_windy(theta_deg - h)) / (2 * h)

bad_runs = [
    (18.0, BLUE, "start 18° → global ✓"),
    (72.0, RED,  "start 72° → stuck at local ✗"),
]
for start, col, lbl in bad_runs:
    theta = start
    xs, ys = [theta], [R_windy(theta)]
    for _ in range(40):
        theta = theta + 5.0 * dR_windy(theta)
        theta = np.clip(theta, 5, 85)
        xs.append(theta); ys.append(R_windy(theta))
    ax3.plot(xs, ys, color=col, lw=1.4, alpha=0.9, marker="o", markersize=3.2,
             label=lbl)

ax3.set_xlabel("release angle  θ  (degrees)")
ax3.set_ylabel("range  R  (m)")
ax3.set_title("Non-convex — local vs global", fontsize=11.5,
              fontweight="bold", color=DARK, loc="left")
ax3.legend(loc="lower center", fontsize=8.5, framealpha=0.95)
ax3.grid(True, alpha=0.25)

fig.tight_layout(rect=(0, 0, 1, 0.96))
out = Path(__file__).resolve().parent / "ch04-small-steps.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")
