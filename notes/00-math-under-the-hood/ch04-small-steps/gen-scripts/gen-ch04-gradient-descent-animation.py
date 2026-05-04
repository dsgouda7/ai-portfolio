"""Generate gradient descent animation: ball rolling down a loss curve.

Shows a ball (current θ) descending the range function R(θ) = (v₀²/g) sin(2θ),
which we MINIMIZE as -R (so we find the maximum range). Displays:
  - The loss curve (negative range)
  - Ball position at each iteration
  - Tangent line at current position (the gradient)
  - Iteration counter + current loss value
  - Learning rate η displayed

20 iterations at η = 0.15, starting from θ₀ = 20°.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from pathlib import Path

DARK   = "#2C3E50"
BLUE   = "#2E86C1"
ORANGE = "#E67E22"
RED    = "#E74C3C"
GREEN  = "#27AE60"

# Range formula (we MAXIMIZE this, so we MINIMIZE loss = -R)
v0, g = 25.0, 9.81
R = lambda theta_deg: (v0 ** 2 / g) * np.sin(np.radians(2 * theta_deg))
R_prime = lambda theta_deg: (v0 ** 2 / g) * 2 * np.cos(np.radians(2 * theta_deg)) * (np.pi / 180)

# Loss = -R (we're minimizing, so gradient descent walks downhill)
loss = lambda theta_deg: -R(theta_deg)
loss_prime = lambda theta_deg: -R_prime(theta_deg)

# Gradient descent
theta0 = 20.0
eta = 0.15
n_iters = 20

theta_vals = [theta0]
loss_vals = [loss(theta0)]
for k in range(n_iters):
    grad = loss_prime(theta_vals[-1])
    theta_new = theta_vals[-1] - eta * grad
    theta_vals.append(theta_new)
    loss_vals.append(loss(theta_new))

# Plot setup
theta_grid = np.linspace(5, 85, 300)
loss_grid = loss(theta_grid)

fig, ax = plt.subplots(figsize=(11, 6.5), facecolor="white")
ax.set_xlim(0, 90)
ax.set_ylim(min(loss_grid) - 5, max(loss_grid) + 5)
ax.set_xlabel(r"launch angle  $\theta$  (degrees)", fontsize=12, color=DARK)
ax.set_ylabel(r"loss  $L(\theta) = -R(\theta)$  (m)", fontsize=12, color=DARK)
ax.set_title("Gradient Descent: Ball Rolling Down the Loss Curve", 
             fontsize=14, fontweight="bold", color=DARK)
ax.grid(True, alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Static: loss curve
ax.plot(theta_grid, loss_grid, color=BLUE, lw=2.4, label=r"$L(\theta) = -R(\theta)$")

# Static: minimum marker (at 45°)
theta_opt = 45.0
loss_opt = loss(theta_opt)
ax.plot([theta_opt], [loss_opt], "*", color=GREEN, ms=20, 
        label=r"true minimum ($\theta^* = 45°$)", zorder=6)

# Animated: ball
ball, = ax.plot([], [], "o", color=ORANGE, ms=16, zorder=5)

# Animated: tangent line
tangent_line, = ax.plot([], [], color=RED, lw=2.2, alpha=0.8)

# Animated: trajectory breadcrumbs (fading trail)
trail, = ax.plot([], [], "o-", color=ORANGE, ms=5, alpha=0.4, lw=1)

# Animated: info box
info_text = ax.text(0.02, 0.97, "", transform=ax.transAxes, fontsize=11,
                    va="top", ha="left", color=DARK, family="monospace",
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=DARK, lw=1.2))

ax.legend(loc="upper right", fontsize=10, frameon=False)

def init():
    ball.set_data([], [])
    tangent_line.set_data([], [])
    trail.set_data([], [])
    info_text.set_text("")
    return ball, tangent_line, trail, info_text

def animate(frame):
    theta_now = theta_vals[frame]
    loss_now = loss_vals[frame]
    grad = loss_prime(theta_now)
    
    # Ball position
    ball.set_data([theta_now], [loss_now])
    
    # Tangent line: slope is the gradient
    dtheta = 8.0  # degrees to extend left/right
    theta_tan = np.array([theta_now - dtheta, theta_now + dtheta])
    loss_tan = loss_now + grad * (theta_tan - theta_now)
    tangent_line.set_data(theta_tan, loss_tan)
    
    # Trail (last 5 positions)
    start_idx = max(0, frame - 4)
    trail.set_data(theta_vals[start_idx:frame+1], loss_vals[start_idx:frame+1])
    
    # Info box
    info_text.set_text(
        f"Iteration: {frame}\n"
        f"θ = {theta_now:.2f}°\n"
        f"L(θ) = {loss_now:.2f} m\n"
        f"η = {eta}\n"
        f"∇L = {grad:+.2f}"
    )
    
    return ball, tangent_line, trail, info_text

n_frames = len(theta_vals)
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=n_frames,
                              interval=400, blit=True, repeat=True)

out = Path(__file__).with_name("ch04-gradient-descent-animation.gif")
print(f"Rendering {n_frames} frames...")
anim.save(out, writer="pillow", fps=2.5, dpi=110)
print(f"wrote {out}")
plt.close()
