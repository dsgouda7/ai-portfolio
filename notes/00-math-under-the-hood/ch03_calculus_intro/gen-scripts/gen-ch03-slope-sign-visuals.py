"""Generate two visuals for §4.2.2 "What the Slope's Sign and Size Actually Tell You":

1. ch03-slope-sign-panels.png  — Static 5-panel: five key moments from the table,
   each showing the ball's position, its color-coded tangent line, and an annotation
   of h'(t) with the direction label (rising / peak / falling).

2. ch03-slope-rotation-animation.gif — Animated: the ball moves along the arc while
   the tangent line rotates.  Tangent color changes continuously:
       green  → ball is rising (h' > 0.3)
       yellow → near the apex  (|h'| ≤ 0.3)
       red    → ball is falling (h' < -0.3)
   Direction label ("↗ RISING", "─ PEAK", "↘ FALLING") updates each frame.

Physics: h(t) = 6.5t − 4.905t²   (v0y = 6.5 m/s, g = 9.81 m/s²)
         h'(t) = 6.5 − 9.81t
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent

# ── colour palette ─────────────────────────────────────────────────────────────
DARK   = "#2C3E50"
BLUE   = "#2E86C1"
ORANGE = "#E67E22"
RED    = "#C0392B"
GREEN  = "#27AE60"
YELLOW = "#F1C40F"
GREY   = "#95A5A6"
BG     = "white"

# ── free-kick physics ──────────────────────────────────────────────────────────
v0y, g = 6.5, 9.81
def h(t):    return v0y * t - 0.5 * g * t ** 2
def hdot(t): return v0y - g * t
t_apex = v0y / g          # ≈ 0.663 s
t_land = 2 * v0y / g     # ≈ 1.326 s
h_apex = h(t_apex)

# Dense curve for plotting
t_curve = np.linspace(0, t_land, 400)
h_curve = h(t_curve)

# The five table moments
moments = [
    (0.0,    "t = 0.0 s\n(launch)"),
    (0.2,   "t = 0.2 s"),
    (0.5,   "t = 0.5 s"),
    (t_apex, f"t = {t_apex:.3f} s\n(apex)"),
    (1.0,   "t = 1.0 s"),
]

# ── helper: tangent segment ────────────────────────────────────────────────────
def tangent_segment(t0, half_width=0.18):
    slope = hdot(t0)
    t_pts = np.array([t0 - half_width, t0 + half_width])
    h_pts = h(t0) + slope * (t_pts - t0)
    return t_pts, h_pts

def tangent_color(slope):
    if abs(slope) < 0.35:
        return YELLOW
    return GREEN if slope > 0 else RED

def direction_label(slope):
    if abs(slope) < 0.35:
        return "─  PEAK"
    return "↗  RISING" if slope > 0 else "↘  FALLING"

# ══════════════════════════════════════════════════════════════════════════════
# Visual 1 — Static 5-panel snapshot
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 5, figsize=(18, 5), facecolor=BG,
                         gridspec_kw={"wspace": 0.35})
fig.suptitle("The Slope's Sign and Size at Five Moments of the Free Kick",
             fontsize=15, fontweight="bold", color=DARK, y=1.01)

for ax, (t0, label) in zip(axes, moments):
    slope = hdot(t0)
    h0    = h(t0)
    color = tangent_color(slope)
    dlabel = direction_label(slope)

    # Full trajectory (faint)
    ax.plot(t_curve, h_curve, color=BLUE, lw=1.5, alpha=0.25)

    # Tangent line
    t_tan, h_tan = tangent_segment(t0, half_width=0.22)
    # clip tangent below ground
    h_tan_clipped = np.clip(h_tan, -0.1, h_apex + 0.8)
    ax.plot(t_tan, h_tan_clipped, color=color, lw=3.0, solid_capstyle="round",
            zorder=4)

    # Ball
    ax.plot(t0, h0, "o", color=ORANGE, ms=14, zorder=5,
            markeredgecolor=DARK, markeredgewidth=1.2)

    # Wall / crossbar reference lines (light)
    ax.axhline(1.8,  color=GREY, lw=0.9, ls="--", alpha=0.5, label="_")
    ax.axhline(2.44, color=GREY, lw=0.9, ls=":",  alpha=0.5, label="_")

    # h'(t) annotation box
    sign_str = f"+{slope:.2f}" if slope >= 0 else f"{slope:.2f}"
    ax.text(0.50, 0.04,
            f"$h'(t) = {sign_str}$ m/s\n{dlabel}",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=10.5, color=DARK, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.45", fc="#FAFAFA", ec=color, lw=2.2))

    # Time label at top
    ax.set_title(label, fontsize=11, color=DARK, pad=6)

    ax.set_xlim(0, t_land + 0.05)
    ax.set_ylim(-0.25, h_apex + 0.85)
    ax.set_xlabel("time (s)", fontsize=10, color=DARK)
    if ax is axes[0]:
        ax.set_ylabel("height (m)", fontsize=10, color=DARK)
    ax.grid(True, alpha=0.20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=DARK, labelsize=9)

# Legend strip at bottom
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], color=GREEN,  lw=3, label="Rising  (h′ > 0)"),
    Line2D([0], [0], color=YELLOW, lw=3, label="At peak (h′ ≈ 0)"),
    Line2D([0], [0], color=RED,    lw=3, label="Falling (h′ < 0)"),
    Line2D([0], [0], color=GREY,   lw=1.5, ls="--", label="Wall 1.8 m"),
    Line2D([0], [0], color=GREY,   lw=1.5, ls=":",  label="Crossbar 2.44 m"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=5,
           fontsize=10.5, frameon=False, bbox_to_anchor=(0.5, -0.10))

out1 = HERE / "ch03-slope-sign-panels.png"
fig.savefig(out1, dpi=120, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"wrote {out1}")

# ══════════════════════════════════════════════════════════════════════════════
# Visual 2 — Animated tangent rotation (45 frames, 12 fps)
# ══════════════════════════════════════════════════════════════════════════════
N_FRAMES = 54
t_anim = np.linspace(0.0, t_land * 0.97, N_FRAMES)

fig2, ax2 = plt.subplots(figsize=(10, 6), facecolor=BG)

ax2.set_xlim(-0.04, t_land + 0.08)
ax2.set_ylim(-0.35, h_apex + 0.90)
ax2.set_xlabel("time  $t$  (s)", fontsize=13, color=DARK)
ax2.set_ylabel("height  $h$  (m)", fontsize=13, color=DARK)
ax2.set_title("The Tangent Rotates: Sign Tells Direction, Size Tells Steepness",
              fontsize=13, fontweight="bold", color=DARK)
ax2.grid(True, alpha=0.22)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# Static: full trajectory
ax2.plot(t_curve, h_curve, color=BLUE, lw=1.6, alpha=0.30,
         label=r"$h(t)=6.5t-4.905t^2$")

# Static: apex
ax2.plot(t_apex, h_apex, "x", color=RED, ms=14, mew=2.5, zorder=6,
         label=r"apex: $h'=0$")
ax2.axhline(h_apex, color=RED, lw=0.7, ls=":", alpha=0.45)

# Static: wall and crossbar
ax2.axhline(1.80, color=GREY, lw=1.2, ls="--", alpha=0.55, label="wall  1.8 m")
ax2.axhline(2.44, color=GREY, lw=1.2, ls=":",  alpha=0.55, label="crossbar  2.44 m")

ax2.legend(loc="lower left", fontsize=10.5, frameon=False)

# Animated objects
ball2,    = ax2.plot([], [], "o", color=ORANGE, ms=18, zorder=7,
                     markeredgecolor=DARK, markeredgewidth=1.3)
tan_line, = ax2.plot([], [], "-", lw=3.2, solid_capstyle="round", zorder=5)

info_box = ax2.text(
    0.98, 0.97, "",
    transform=ax2.transAxes, ha="right", va="top",
    fontsize=13, color=DARK, fontweight="bold", family="monospace",
    bbox=dict(boxstyle="round,pad=0.55", fc="#FAFEFE", ec=GREEN, lw=2.2)
)
dir_box = ax2.text(
    0.015, 0.97, "",
    transform=ax2.transAxes, ha="left", va="top",
    fontsize=14, color=DARK, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.50", fc="#FAFAFA", ec=GREY, lw=1.5)
)

# Trace of past ball positions
trail_x, trail_y = [], []
trail_line, = ax2.plot([], [], ".", color=ORANGE, ms=4, alpha=0.40, zorder=4)

def init2():
    ball2.set_data([], [])
    tan_line.set_data([], [])
    trail_line.set_data([], [])
    info_box.set_text("")
    dir_box.set_text("")
    trail_x.clear()
    trail_y.clear()
    return ball2, tan_line, trail_line, info_box, dir_box

def animate2(frame):
    t0 = t_anim[frame]
    h0 = h(t0)
    slope = hdot(t0)
    color = tangent_color(slope)
    dlabel = direction_label(slope)

    # Ball
    ball2.set_data([t0], [h0])

    # Tangent, clipped to plot bounds
    t_tan, h_tan = tangent_segment(t0, half_width=0.24)
    # clip tangent outside vertical plot range
    mask = (h_tan >= -0.35) & (h_tan <= h_apex + 0.90)
    t_plot = np.where(mask, t_tan, np.nan)
    tan_line.set_data(t_plot, h_tan)
    tan_line.set_color(color)

    # Trail
    trail_x.append(t0)
    trail_y.append(h0)
    trail_line.set_data(trail_x, trail_y)

    # Info box
    sign_str = f"+{slope:.2f}" if slope >= 0 else f"{slope:.2f}"
    info_box.set_text(f"t = {t0:.3f} s\nh′(t) = {sign_str} m/s")
    info_box.get_bbox_patch().set_edgecolor(color)

    # Direction label
    dir_box.set_text(dlabel)
    dir_box.get_bbox_patch().set_edgecolor(color)

    return ball2, tan_line, trail_line, info_box, dir_box

anim2 = animation.FuncAnimation(
    fig2, animate2, init_func=init2,
    frames=N_FRAMES, interval=1000 / 12, blit=True, repeat=True
)

out2 = HERE / "ch03-slope-rotation-animation.gif"
print(f"Rendering {N_FRAMES} frames at 12 fps...")
anim2.save(out2, writer="pillow", fps=12, dpi=100)
print(f"wrote {out2}")
plt.close()
