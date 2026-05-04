"""Generate slow-motion free-kick animation showing the derivative at each instant.

The animation shows:
  - The full parabolic trajectory (faint)
  - A ball marker moving along the path
  - A tangent line at the ball's current position (the derivative)
  - A digital readout of h'(t) = v_y(t) updating in real-time
  - Time elapsed counter
  - Apex marker (where h'(t) = 0)

Saved as an animated GIF (15 fps, ~3 second duration, 45 frames).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import numpy as np
from pathlib import Path

DARK   = "#2C3E50"
BLUE   = "#2E86C1"
ORANGE = "#E67E22"
RED    = "#E74C3C"
GREEN  = "#27AE60"
GREY   = "#95A5A6"

# Free-kick physics
v0y, g = 7.2, 9.81
t_max = 2 * v0y / g  # time to land
t = np.linspace(0, t_max, 300)
h = v0y * t - 0.5 * g * t ** 2
v_y = lambda t_val: v0y - g * t_val  # derivative (instantaneous vertical velocity)
t_apex = v0y / g

# Animation: 45 frames over the trajectory
n_frames = 45
t_vals = np.linspace(0, t_max * 0.95, n_frames)  # stop just before ground impact

fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
ax.set_xlim(-0.05, t_max + 0.05)
ax.set_ylim(-0.5, max(h) + 1.0)
ax.set_xlabel("time  $t$  (s)", fontsize=13, color=DARK)
ax.set_ylabel("height  $h$  (m)", fontsize=13, color=DARK)
ax.set_title("Free Kick: Derivative = Instantaneous Vertical Velocity", 
             fontsize=14, fontweight="bold", color=DARK)
ax.grid(True, alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Static: full trajectory (faint)
ax.plot(t, h, color=BLUE, lw=1.2, alpha=0.35, label=r"$h(t) = v_{0y}t - \frac{1}{2}gt^2$")

# Static: apex marker
h_apex = v0y * t_apex - 0.5 * g * t_apex ** 2
ax.plot([t_apex], [h_apex], "x", color=RED, ms=12, mew=2.5, 
        label=r"apex: $h'(t)=0$")
ax.axhline(h_apex, color=RED, lw=0.8, ls=":", alpha=0.5)

# Animated elements (initialized empty)
ball, = ax.plot([], [], "o", color=ORANGE, ms=18, zorder=5)
tangent_line, = ax.plot([], [], color=GREEN, lw=2.8, label="tangent (slope = $h'(t)$)")
time_text = ax.text(0.02, 0.96, "", transform=ax.transAxes, fontsize=12, 
                    va="top", ha="left", color=DARK, family="monospace",
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=DARK, lw=1))
derivative_text = ax.text(0.98, 0.96, "", transform=ax.transAxes, fontsize=13,
                          va="top", ha="right", color=DARK, fontweight="bold",
                          bbox=dict(boxstyle="round,pad=0.5", fc="#FFF9E6", ec=GREEN, lw=2))

ax.legend(loc="lower left", fontsize=11, frameon=False)

def init():
    ball.set_data([], [])
    tangent_line.set_data([], [])
    time_text.set_text("")
    derivative_text.set_text("")
    return ball, tangent_line, time_text, derivative_text

def animate(frame):
    t_now = t_vals[frame]
    h_now = v0y * t_now - 0.5 * g * t_now ** 2
    slope = v_y(t_now)
    
    # Ball position
    ball.set_data([t_now], [h_now])
    
    # Tangent line: extend ±0.20 s in time
    dt_extend = 0.25
    t_tan = np.array([t_now - dt_extend, t_now + dt_extend])
    h_tan = h_now + slope * (t_tan - t_now)
    tangent_line.set_data(t_tan, h_tan)
    
    # Time counter
    time_text.set_text(f"t = {t_now:.3f} s")
    
    # Derivative readout with color coding
    color_code = GREEN if abs(slope) > 0.5 else (ORANGE if abs(slope) > 0.01 else RED)
    derivative_text.set_text(f"h'(t) = {slope:+.2f} m/s")
    derivative_text.get_bbox_patch().set_edgecolor(color_code)
    
    return ball, tangent_line, time_text, derivative_text

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=n_frames,
                              interval=1000/15, blit=True, repeat=True)

out = Path(__file__).with_name("ch03-freekick-derivative-animation.gif")
print(f"Rendering {n_frames} frames at 15 fps... (this takes ~20 seconds)")
anim.save(out, writer="pillow", fps=15, dpi=100)
print(f"wrote {out}")
plt.close()
