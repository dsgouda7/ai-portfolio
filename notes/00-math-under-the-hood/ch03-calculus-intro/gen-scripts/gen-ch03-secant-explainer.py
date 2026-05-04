"""Generate the secant-explainer image for Pre-Requisites Ch.3.

A single annotated panel that explains, in one picture, what a *secant* is and
how the second point t0 + dt is constructed:

  - Plot the knuckleball free-kick height curve h(t) = v0y*t - 0.5*g*t**2.
  - Mark the anchor point (t0, h(t0)) on the curve.
  - Construct the second point at (t0 + dt, h(t0 + dt)) by:
      * a vertical dashed line at t = t0 + dt down to the t-axis,
      * a horizontal dashed line from (t0 + dt) on the axis up to the curve.
  - Draw the secant through the two points (Latin "secare" = to cut).
  - Annotate the rise (delta h) and run (delta t) with arrows.
  - Show the slope formula (h(t0+dt) - h(t0)) / dt as a callout.
  - Show two more secants (smaller dt) ghosted in to hint at the limit -> tangent.

Run from this folder:
    python gen_ch03_secant_explainer.py
Output: ch03-secant-explainer.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

DARK   = "#2C3E50"
BLUE   = "#2E86C1"
RED    = "#E74C3C"
ORANGE = "#E67E22"
GREEN  = "#27AE60"
GREY   = "#7F8C8D"
LIGHT  = "#BDC3C7"

# --- the free-kick height curve (vertical component, ball off the turf) ---
v0y, g = 7.2, 9.81
t = np.linspace(0, 1.47, 400)
h = v0y * t - 0.5 * g * t ** 2

# --- anchor point and the "big" dt that we will explain in detail ---
# both points kept on the *rising* side of the apex (apex at t = v0y/g ≈ 0.73 s)
# so the rise (Δh) is visibly positive, not eaten by the parabola turning over.
t0 = 0.18
dt_big = 0.40
h0 = v0y * t0 - 0.5 * g * t0 ** 2
h1 = v0y * (t0 + dt_big) - 0.5 * g * (t0 + dt_big) ** 2

# --- two ghosted secants with smaller dt (preview of the limit) ---
dts_ghost = [0.22, 0.08]

fig, ax = plt.subplots(figsize=(10.5, 6.4), facecolor="white")
fig.suptitle("Ch.3 \u2014 What a secant is, and how $t_0 + \\Delta t$ is constructed",
             fontsize=14, fontweight="bold", color=DARK, y=0.98)

# the curve
ax.plot(t, h, color=BLUE, lw=2.6, label=r"$h(t) = v_{0y}\,t - \frac{1}{2}\,g\,t^2$")

# ghosted secants (smaller dt) - drawn first so the main secant sits on top
for dt_g, alpha in zip(dts_ghost, [0.45, 0.65]):
    h1g = v0y * (t0 + dt_g) - 0.5 * g * (t0 + dt_g) ** 2
    slope_g = (h1g - h0) / dt_g
    # extend the secant a bit beyond both endpoints
    x_line = np.array([t0 - 0.05, t0 + dt_g + 0.05])
    y_line = h0 + slope_g * (x_line - t0)
    ax.plot(x_line, y_line, color=GREY, lw=1.1, ls="--", alpha=alpha)
    ax.plot([t0 + dt_g], [h1g], "o", color=GREY, ms=5, alpha=alpha)

# the two main points
ax.plot([t0], [h0], "o", color=RED, ms=9, zorder=5)
ax.plot([t0 + dt_big], [h1], "o", color=RED, ms=9, zorder=5)

# construction lines for the second point: vertical from axis up, horizontal from curve over
ax.plot([t0 + dt_big, t0 + dt_big], [0, h1], color=ORANGE, lw=1.4, ls=":")
ax.plot([0, t0 + dt_big], [h1, h1], color=ORANGE, lw=1.4, ls=":")
ax.plot([t0, t0], [0, h0], color=ORANGE, lw=1.4, ls=":")
ax.plot([0, t0], [h0, h0], color=ORANGE, lw=1.4, ls=":")

# the secant line itself
slope = (h1 - h0) / dt_big
x_sec = np.array([t0 - 0.18, t0 + dt_big + 0.22])
y_sec = h0 + slope * (x_sec - t0)
ax.plot(x_sec, y_sec, color=RED, lw=2.4,
        label=r"secant: slope $= \frac{h(t_0+\Delta t)-h(t_0)}{\Delta t}$")

# rise / run annotations
# run arrow below both points
y_run = min(h0, h1) - 0.45
ax.annotate("", xy=(t0 + dt_big, y_run), xytext=(t0, y_run),
            arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1.6))
ax.text(t0 + dt_big / 2, y_run - 0.10, r"$\Delta t$  (the run)",
        ha="center", va="top", color=GREEN, fontsize=11, fontweight="bold")

# rise arrow just to the right of the second point
x_rise = t0 + dt_big + 0.05
ax.annotate("", xy=(x_rise, h1), xytext=(x_rise, h0),
            arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1.6))
ax.text(x_rise + 0.02, (h0 + h1) / 2,
        r"$\Delta h$" "\n" r"$= h(t_0{+}\Delta t) - h(t_0)$",
        ha="left", va="center", color=GREEN, fontsize=10, fontweight="bold")

# point labels
ax.annotate(r"$(t_0,\; h(t_0))$", xy=(t0, h0), xytext=(t0 - 0.05, h0 - 0.45),
            color=DARK, fontsize=11, ha="right",
            arrowprops=dict(arrowstyle="-", color=DARK, lw=0.8))
ax.annotate(r"$(t_0+\Delta t,\; h(t_0+\Delta t))$",
            xy=(t0 + dt_big, h1), xytext=(t0 + dt_big - 0.05, h1 + 0.50),
            color=DARK, fontsize=11, ha="center",
            arrowprops=dict(arrowstyle="-", color=DARK, lw=0.8))

# axis tick labels for t0 and t0 + dt
ax.set_xticks([0, t0, t0 + dt_big])
ax.set_xticklabels(["0", r"$t_0$", r"$t_0 + \Delta t$"], fontsize=11)
ax.set_yticks([0, h0, h1])
ax.set_yticklabels(["0", r"$h(t_0)$", r"$h(t_0+\Delta t)$"], fontsize=10)

# explainer callout (top-left, where there is empty sky above the curve)
callout = (
    "How to build the second point:\n"
    "  1. Pick Δt > 0 (the run).\n"
    "  2. Step right to t = t₀ + Δt.\n"
    "  3. Read the curve there: h(t₀ + Δt).\n"
    "  4. Draw the line through both points\n"
    "     — that line is the SECANT\n"
    "     (Latin secare, \"to cut\")."
)
ax.text(0.02, 0.97, callout, transform=ax.transAxes, ha="left", va="top",
        fontsize=10, color=DARK,
        bbox=dict(boxstyle="round,pad=0.5", fc="#FDF6E3", ec=DARK, lw=0.8))

# limit hint along the bottom
ax.text(0.50, 0.04,
        "Shrink Δt → 0 and the secant rotates onto the TANGENT at t₀ — "
        "its slope is the DERIVATIVE h′(t₀).",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=10, color=DARK, style="italic")

ax.set_xlabel("time  $t$  (s)", fontsize=12, color=DARK)
ax.set_ylabel("height  $h$  (m)", fontsize=12, color=DARK)
ax.set_xlim(-0.02, 1.05)
ax.set_ylim(-0.65, 3.6)
ax.grid(True, alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper right", fontsize=10, frameon=False)

fig.tight_layout(rect=(0, 0, 1, 0.96))
out = Path(__file__).with_name("ch03-secant-explainer.png")
fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")
