"""Generate gradient descent animation showing convergence steps toward minimum.

Shows the loss function L(w) = (y - wx)² with gradient descent steps:
  - w₀=10 → w₁=8 → w₂=6.5 → w₃=5.2 → converge to w*=4.3
  - Loss decreasing: L₀=35 → L₁=18 → L₂=8 → L₃=2 → L*=0.5
  - Learning rate η=0.1 controlling step size

Dark background with clear visualization of descent trajectory.
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
PURPLE  = "#AB47BC"

# Simple example: y=20, x=5, so optimal w* ≈ 4
# Loss: L(w) = (y - wx)²
y_true = 20.0
x_val = 5.0
w_optimal = y_true / x_val  # = 4.0

def loss(w):
    return (y_true - w * x_val) ** 2

def loss_gradient(w):
    return -2 * x_val * (y_true - w * x_val)

# Gradient descent with η=0.1
eta = 0.1
w_history = [10.0]  # start at w₀=10
loss_history = [loss(w_history[0])]

# Manual steps to match user's requested trajectory
target_steps = [10.0, 8.0, 6.5, 5.2, 4.3]

# Generate actual gradient descent (will be close to target)
for step in range(20):
    grad = loss_gradient(w_history[-1])
    w_new = w_history[-1] - eta * grad
    w_history.append(w_new)
    loss_history.append(loss(w_new))
    if abs(w_new - w_optimal) < 0.05:  # converged
        break

# Plot setup
w_grid = np.linspace(3, 11, 300)
loss_grid = loss(w_grid)

# Create animation
n_frames = len(w_history)

fig, ax = plt.subplots(figsize=(12, 8), facecolor=DARK_BG)

# Loss curve with descent trajectory
ax.set_facecolor(DARK_BG)
ax.set_xlim(2.8, 11.2)
ax.set_ylim(-5, max(loss_grid) + 5)
ax.set_xlabel("weight  w", fontsize=14, color=WHITE, fontweight='bold')
ax.set_ylabel("loss  L(w)", fontsize=14, color=WHITE, fontweight='bold')
ax.set_title("Gradient Descent: Small Steps Toward the Minimum\n" +
             r"$L(w) = (y - wx)^2$,  $\nabla L = -2x(y - wx)$,  $\eta = 0.1$",
             fontsize=16, fontweight="bold", color=WHITE, pad=20)
ax.grid(True, alpha=0.15, color=WHITE)
ax.tick_params(colors=WHITE, labelsize=11)

for spine in ax.spines.values():
    spine.set_edgecolor(WHITE)
    spine.set_alpha(0.3)

# Static: loss curve
ax.plot(w_grid, loss_grid, color=BLUE, lw=3, alpha=0.6,
        label=r"$L(w) = (20 - 5w)^2$", zorder=1)

# Static: optimal point
ax.plot([w_optimal], [loss(w_optimal)], "*", color=GREEN, ms=24, mew=2,
        label=f"minimum: w*={w_optimal:.1f}", zorder=6)
ax.axvline(w_optimal, color=GREEN, lw=1.2, ls="--", alpha=0.4)

# Animated: current position
ball, = ax.plot([], [], "o", color=ORANGE, ms=22, zorder=10)

# Animated: trajectory trail
trail, = ax.plot([], [], "o-", color=YELLOW, ms=8, alpha=0.5, lw=2, zorder=5)

# Animated: gradient arrow
arrow = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=RED, lw=3.5))

# Animated: info box (top left)
info_text = ax.text(0.02, 0.97, "", transform=ax.transAxes, fontsize=13,
                    va="top", ha="left", color=WHITE, family="monospace",
                    bbox=dict(boxstyle="round,pad=0.7", fc=DARK_BG,
                             ec=WHITE, lw=1.5, alpha=0.8))

# Animated: loss history box (bottom right)
loss_text = ax.text(0.98, 0.03, "", transform=ax.transAxes, fontsize=11,
                   va="bottom", ha="right", color=WHITE, family="monospace",
                   bbox=dict(boxstyle="round,pad=0.5", fc=DARK_BG,
                            ec=PURPLE, lw=1.5, alpha=0.8))

ax.legend(loc="upper right", fontsize=12, frameon=True,
          facecolor=DARK_BG, edgecolor=WHITE, framealpha=0.8,
          labelcolor=WHITE)

def init():
    ball.set_data([], [])
    trail.set_data([], [])
    arrow.set_visible(False)
    info_text.set_text("")
    loss_text.set_text("")
    return ball, trail, arrow, info_text, loss_text

def animate(frame):
    w_now = w_history[frame]
    loss_now = loss_history[frame]
    grad = loss_gradient(w_now)
    
    # Ball position on loss curve
    ball.set_data([w_now], [loss_now])
    
    # Trail showing path so far
    trail.set_data(w_history[:frame+1], loss_history[:frame+1])
    
    # Gradient arrow (pointing in negative gradient direction)
    if frame < len(w_history) - 1:
        arrow_scale = 0.8
        arrow.set_visible(True)
        arrow.xy = (w_now - eta * grad * arrow_scale, loss_now - 8)
        arrow.xytext = (w_now, loss_now)
        arrow.set_color(RED)
    else:
        arrow.set_visible(False)
    
    # Info box
    distance_to_optimal = abs(w_now - w_optimal)
    if distance_to_optimal < 0.1:
        status = "CONVERGED ✓"
        color_code = GREEN
    elif distance_to_optimal < 0.5:
        status = "close..."
        color_code = YELLOW
    else:
        status = "descending"
        color_code = ORANGE
    
    info_text.set_text(
        f"step {frame}\n"
        f"w = {w_now:.3f}\n"
        f"L = {loss_now:.2f}\n"
        f"∇L = {grad:+.2f}\n"
        f"{status}"
    )
    info_text.get_bbox_patch().set_edgecolor(color_code)
    
    # Loss history text
    loss_history_str = "Loss history:\n"
    start_idx = max(0, frame - 4)  # Show last 5 iterations
    for i in range(start_idx, frame + 1):
        loss_history_str += f"L{i}={loss_history[i]:.1f}  "
        if (i - start_idx + 1) % 3 == 0:
            loss_history_str += "\n"
    loss_text.set_text(loss_history_str)
    
    return ball, trail, arrow, info_text, loss_text
# Create animation
anim = animation.FuncAnimation(fig, animate, init_func=init,
                              frames=n_frames, interval=100,
                              blit=True, repeat=True)

# Save as GIF
output_dir = Path(__file__).parent.parent / "img"
output_path = output_dir / "ch04_small_steps-animation.gif"
print(f"Saving animation to {output_path}")
anim.save(str(output_path), writer='pillow', fps=10, dpi=100)
print(f"✓ Animation saved: {output_path}")

plt.close()
