"""
Generate animated GIF showing feature lookup latency reduction.

Shows needle movement from 380ms (Direct DB) to 8ms (Feature Store).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Wedge, FancyBboxPatch
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

plt.style.use('dark_background')

# Create figure
fig, ax = plt.subplots(figsize=(12, 8), facecolor='#1a1a2e')
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-0.3, 1.3)
ax.set_aspect('equal')
ax.axis('off')

colors = {
    'danger': '#b91c1c',
    'caution': '#b45309',
    'success': '#15803d',
    'bg': '#1a1a2e'
}

# Draw gauge background
gauge_colors = ['#15803d', '#15803d', '#b45309', '#b91c1c']  # Green, yellow, red
gauge_angles = [0, 60, 120, 180]

for i in range(len(gauge_colors)):
    theta1 = gauge_angles[i]
    theta2 = gauge_angles[i+1] if i < len(gauge_angles)-1 else 180
    wedge = Wedge((0, 0), 1, theta1, theta2, 
                  facecolor=gauge_colors[i], 
                  edgecolor='white', 
                  linewidth=2,
                  alpha=0.6)
    ax.add_patch(wedge)

# Add tick marks and labels
max_latency = 500
ticks = [0, 10, 50, 100, 200, 380, 500]
for tick in ticks:
    angle_rad = np.radians(180 - (tick / max_latency) * 180)
    x_start = 0.85 * np.cos(angle_rad)
    y_start = 0.85 * np.sin(angle_rad)
    x_end = 1.0 * np.cos(angle_rad)
    y_end = 1.0 * np.sin(angle_rad)
    
    ax.plot([x_start, x_end], [y_start, y_end], 
           color='white', linewidth=2)
    
    x_label = 1.15 * np.cos(angle_rad)
    y_label = 1.15 * np.sin(angle_rad)
    ax.text(x_label, y_label, f'{tick}ms',
           ha='center', va='center', fontsize=10, color='white', fontweight='bold')

# Title
title_text = ax.text(0, 1.15, 'Feature Lookup Latency', 
                     ha='center', fontsize=16, fontweight='bold', color='white')

# Current value display
value_text = ax.text(0, -0.15, '', ha='center', fontsize=20, 
                    fontweight='bold', color='white')

# Method label
method_text = ax.text(0, 0.4, '', ha='center', fontsize=14, 
                     fontweight='bold', color='white')

# Needle (will be updated in animation)
needle_line, = ax.plot([], [], color='white', linewidth=4, zorder=10)
needle_dot = plt.Circle((0, 0), 0.05, color='white', zorder=11)
ax.add_patch(needle_dot)

# Target line (10ms)
target_latency = 10
target_angle_rad = np.radians(180 - (target_latency / max_latency) * 180)
target_x = 1.0 * np.cos(target_angle_rad)
target_y = 1.0 * np.sin(target_angle_rad)
ax.plot([0, target_x], [0, target_y], color='yellow', 
       linewidth=2, linestyle='--', alpha=0.7, label='10ms target')
ax.text(target_x * 0.7, target_y * 0.7 + 0.15, '10ms\ntarget',
       ha='center', fontsize=9, color='yellow',
       bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['bg'], edgecolor='yellow'))

def update_needle(latency, method, color):
    """Update needle position."""
    angle_rad = np.radians(180 - (latency / max_latency) * 180)
    x = 0.8 * np.cos(angle_rad)
    y = 0.8 * np.sin(angle_rad)
    
    needle_line.set_data([0, x], [0, y])
    needle_line.set_color(color)
    
    value_text.set_text(f'{latency:.0f} ms')
    value_text.set_color(color)
    
    method_text.set_text(method)
    method_text.set_color(color)
    
    return needle_line, value_text, method_text

def animate(frame):
    """Animation function."""
    # Animation stages
    if frame < 30:  # Initial: Direct DB (380ms)
        latency = 380
        method = 'Direct DB (PostgreSQL)'
        color = colors['danger']
    
    elif frame < 60:  # Transition down
        progress = (frame - 30) / 30
        latency = 380 - (380 - 8) * progress
        method = 'Migrating to Feature Store...'
        color = colors['caution']
    
    elif frame < 90:  # Final: Feature Store (8ms)
        latency = 8
        method = 'Feature Store (Redis)'
        color = colors['success']
    
    else:  # Hold at final
        latency = 8
        method = 'Feature Store (Redis) ✅'
        color = colors['success']
    
    return update_needle(latency, method, color)

# Create animation
anim = FuncAnimation(fig, animate, frames=120, interval=50, blit=True)

# Add summary text
summary_box = FancyBboxPatch(
    (-0.9, -0.28), 1.8, 0.15,
    boxstyle="round,pad=0.02",
    facecolor=colors['success'],
    edgecolor='white',
    linewidth=2,
    alpha=0.9
)
ax.add_patch(summary_box)

ax.text(0, -0.245, '97% reduction: 380ms → 8ms',
       ha='center', fontsize=12, color='white', fontweight='bold')
ax.text(0, -0.27, 'Precomputed features in Redis vs on-the-fly PostgreSQL aggregations',
       ha='center', fontsize=9, color='white')

plt.tight_layout()

# Save animation
writer = PillowWriter(fps=20)
anim.save('../img/ch08-feature-store-latency.gif', writer=writer)
print("✓ Generated: ch08-feature-store-latency.gif")

plt.close()
