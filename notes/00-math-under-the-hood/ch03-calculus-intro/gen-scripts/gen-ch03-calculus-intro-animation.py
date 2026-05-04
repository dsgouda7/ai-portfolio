"""Generate animation showing derivative as slope at different points on a parabola.

Shows the trajectory h(t) = -5t² + 6.5t with tangent lines at key points:
  - t=0.2s (slope=4.5)
  - t=0.4s (slope=2.5)  
  - t=0.65s (slope=0, apex at 1.41m height)

Highlights when the derivative equals zero (maximum height).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from pathlib import Path
from PIL import Image

DARK_BG = "#1a1a2e"
BLUE    = "#4FC3F7"
ORANGE  = "#FF9800"
GREEN   = "#66BB6A"
RED     = "#EF5350"
YELLOW  = "#FFD54F"
WHITE   = "#E0E0E0"

# Trajectory: h(t) = -5t² + 6.5t
# Derivative: h'(t) = -10t + 6.5
def h(t):
    return -5 * t**2 + 6.5 * t

def h_prime(t):
    return -10 * t + 6.5

# Find apex: h'(t) = 0 => t = 6.5/10 = 0.65s
t_apex = 6.5 / 10
h_apex = h(t_apex)

# Time range
t_max = 1.3  # trajectory hits ground at t=1.3s
t_grid = np.linspace(0, t_max, 300)
h_grid = h(t_grid)

# Key points to highlight
key_points = [
    (0.2, 4.5, "t=0.2s, slope=4.5"),
    (0.4, 2.5, "t=0.4s, slope=2.5"),
    (0.65, 0.0, "t=0.65s, slope=0 (apex)")
]

# Create animation with 30 frames
n_frames = 30
t_vals = np.linspace(0, t_max * 0.95, n_frames)

fig, ax = plt.subplots(figsize=(12, 8), facecolor=DARK_BG)
ax.set_facecolor(DARK_BG)
ax.set_xlim(-0.05, t_max + 0.05)
ax.set_ylim(-0.1, h_apex + 0.3)
ax.set_xlabel("time  t  (s)", fontsize=14, color=WHITE, fontweight='bold')
ax.set_ylabel("height  h(t)  (m)", fontsize=14, color=WHITE, fontweight='bold')
ax.set_title("Derivative as Instantaneous Slope\n" + 
             r"$h(t) = -5t^2 + 6.5t$,  $h'(t) = -10t + 6.5$",
             fontsize=16, fontweight="bold", color=WHITE, pad=20)
ax.grid(True, alpha=0.15, color=WHITE)
ax.tick_params(colors=WHITE, labelsize=11)

# Remove spines
for spine in ax.spines.values():
    spine.set_edgecolor(WHITE)
    spine.set_alpha(0.3)

# Static: full trajectory (faint)
ax.plot(t_grid, h_grid, color=BLUE, lw=2.5, alpha=0.4, 
        label=r"$h(t) = -5t^2 + 6.5t$", zorder=1)

# Static: apex marker
ax.plot([t_apex], [h_apex], "*", color=RED, ms=24, mew=2,
        label=f"apex: h'(t)=0 at {h_apex:.2f}m", zorder=6)
ax.axhline(h_apex, color=RED, lw=1.2, ls="--", alpha=0.4)

# Static: key point markers
for t_pt, slope_pt, label_pt in key_points:
    ax.plot([t_pt], [h(t_pt)], "o", color=YELLOW, ms=12, alpha=0.7, zorder=5)

# Animated elements
ball, = ax.plot([], [], "o", color=ORANGE, ms=20, zorder=10)
tangent_line, = ax.plot([], [], color=GREEN, lw=3.5, alpha=0.9, 
                        label="tangent line (slope = h'(t))")
time_text = ax.text(0.02, 0.97, "", transform=ax.transAxes, fontsize=13,
                    va="top", ha="left", color=WHITE, family="monospace",
                    bbox=dict(boxstyle="round,pad=0.6", fc=DARK_BG, 
                             ec=WHITE, lw=1.5, alpha=0.8))
derivative_text = ax.text(0.98, 0.97, "", transform=ax.transAxes, fontsize=14,
                          va="top", ha="right", color=WHITE, fontweight="bold",
                          bbox=dict(boxstyle="round,pad=0.7", fc="#2D3E50",
                                   ec=GREEN, lw=2.5, alpha=0.9))

ax.legend(loc="lower left", fontsize=12, frameon=True, 
         facecolor=DARK_BG, edgecolor=WHITE, framealpha=0.8,
         labelcolor=WHITE)

def init():
    ball.set_data([], [])
    tangent_line.set_data([], [])
    time_text.set_text("")
    derivative_text.set_text("")
    return ball, tangent_line, time_text, derivative_text

def animate(frame):
    t_now = t_vals[frame]
    h_now = h(t_now)
    slope = h_prime(t_now)
    
    # Ball position
    ball.set_data([t_now], [h_now])
    
    # Tangent line: extend ±0.15 s in time
    dt_extend = 0.15
    t_tan = np.array([t_now - dt_extend, t_now + dt_extend])
    t_tan = np.clip(t_tan, 0, t_max)
    h_tan = h_now + slope * (t_tan - t_now)
    tangent_line.set_data(t_tan, h_tan)
    
    # Time counter
    time_text.set_text(f"t = {t_now:.3f} s\nh = {h_now:.3f} m")
    
    # Derivative readout with color coding
    if abs(slope) < 0.2:  # near apex
        color_code = RED
        status = "APEX!"
    elif slope > 2:
        color_code = GREEN
        status = "rising"
    elif slope > 0:
        color_code = YELLOW
        status = "slowing"
    else:
        color_code = ORANGE
        status = "falling"
    
    derivative_text.set_text(f"h'(t) = {slope:+.2f} m/s\n{status}")
    derivative_text.get_bbox_patch().set_edgecolor(color_code)
    
    return ball, tangent_line, time_text, derivative_text

# Create animation
anim = animation.FuncAnimation(fig, animate, init_func=init, 
                              frames=n_frames, interval=100, 
                              blit=True, repeat=True)

# Save as GIF
output_dir = Path(__file__).parent.parent / "img"
output_path = output_dir / "ch03_calculus_intro-animation.gif"
print(f"Saving animation to {output_path}")
anim.save(str(output_path), writer='pillow', fps=10, dpi=100)
print(f"✓ Animation saved: {output_path}")

plt.close()
