"""Generate the secant → tangent limit animation.

Shows 8 frames of Δt shrinking: 0.60, 0.40, 0.25, 0.15, 0.08, 0.04, 0.02, 0.01.
The secant rotates onto the tangent, with the Δt value displayed in the corner.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from pathlib import Path

DARK   = "#2C3E50"
BLUE   = "#2E86C1"
RED    = "#E74C3C"
GREEN  = "#27AE60"
GREY   = "#95A5A6"

v0y, g = 7.2, 9.81
t = np.linspace(0, 1.47, 300)
h = v0y * t - 0.5 * g * t ** 2

# Anchor point
t0 = 0.35
h0 = v0y * t0 - 0.5 * g * t0 ** 2
slope_true = v0y - g * t0  # true derivative

# Shrinking Δt sequence
dt_vals = [0.60, 0.40, 0.25, 0.15, 0.08, 0.04, 0.02, 0.01]
n_frames = len(dt_vals)

fig, ax = plt.subplots(figsize=(10, 6.5), facecolor="white")
ax.set_xlim(-0.02, 1.05)
ax.set_ylim(-0.3, 2.8)
ax.set_xlabel("time  $t$  (s)", fontsize=12, color=DARK)
ax.set_ylabel("height  $h$  (m)", fontsize=12, color=DARK)
ax.set_title(r"The Secant Limit: as $\Delta t \to 0$, the secant becomes the tangent",
             fontsize=13, fontweight="bold", color=DARK)
ax.grid(True, alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Static: curve
ax.plot(t, h, color=BLUE, lw=2.4, label=r"$h(t) = v_{0y}t - \frac{1}{2}gt^2$")

# Static: anchor point
ax.plot([t0], [h0], "o", color=DARK, ms=9, zorder=5)
ax.annotate(r"$(t_0, h(t_0))$", xy=(t0, h0), xytext=(t0 - 0.08, h0 - 0.35),
            fontsize=11, ha="right", color=DARK,
            arrowprops=dict(arrowstyle="-", color=DARK, lw=0.8))

# Animated: second point
point2, = ax.plot([], [], "o", color=RED, ms=9, zorder=5)

# Animated: secant line
secant_line, = ax.plot([], [], color=RED, lw=2.6, label="secant")

# Animated: tangent (only shown in final frame)
tangent_line, = ax.plot([], [], color=GREEN, lw=2.6, ls="--", 
                        label=r"tangent (slope = $h'(t_0)$)", alpha=0)

# Animated: Δt readout
dt_text = ax.text(0.98, 0.97, "", transform=ax.transAxes, fontsize=13,
                  va="top", ha="right", color=DARK, fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=RED, lw=1.5))

ax.legend(loc="upper right", fontsize=10, frameon=False)

def init():
    point2.set_data([], [])
    secant_line.set_data([], [])
    tangent_line.set_data([], [])
    dt_text.set_text("")
    return point2, secant_line, tangent_line, dt_text

def animate(frame):
    dt = dt_vals[frame]
    h1 = v0y * (t0 + dt) - 0.5 * g * (t0 + dt) ** 2
    slope_secant = (h1 - h0) / dt
    
    # Second point
    point2.set_data([t0 + dt], [h1])
    
    # Secant line
    t_sec = np.array([t0 - 0.12, t0 + dt + 0.12])
    h_sec = h0 + slope_secant * (t_sec - t0)
    secant_line.set_data(t_sec, h_sec)
    
    # Tangent (fade in on last frame)
    if frame == n_frames - 1:
        tangent_line.set_alpha(1.0)
        t_tan = np.array([t0 - 0.20, t0 + 0.30])
        h_tan = h0 + slope_true * (t_tan - t0)
        tangent_line.set_data(t_tan, h_tan)
    else:
        tangent_line.set_alpha(0)
    
    # Δt readout
    dt_text.set_text(rf"$\Delta t = {dt:.2f}$ s" + "\n" + 
                     f"secant slope = {slope_secant:+.2f}")
    
    return point2, secant_line, tangent_line, dt_text

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=n_frames,
                              interval=600, blit=True, repeat=True)

out = Path(__file__).with_name("ch03-secant-tangent-animation.gif")
print(f"Rendering {n_frames} frames...")
anim.save(out, writer="pillow", fps=1.67, dpi=120)  # ~0.6 sec per frame
print(f"wrote {out}")
plt.close()
