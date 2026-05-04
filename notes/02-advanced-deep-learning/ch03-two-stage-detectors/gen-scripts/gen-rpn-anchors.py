"""
Generate RPN anchor boxes visualization.

Shows the 9 anchor boxes at multiple scales and aspect ratios used by
the Region Proposal Network at each spatial location.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Dark theme
plt.style.use('dark_background')

fig, ax = plt.subplots(figsize=(12, 8), facecolor='#1a1a2e')
ax.set_facecolor('#1a1a2e')

# Define anchor scales and ratios
scales = [128, 256, 512]  # pixels
ratios = [0.5, 1.0, 2.0]  # height/width ratios (1:2, 1:1, 2:1)

# Center point
cx, cy = 300, 300

# Colors for different scales
colors = ['#b91c1c', '#b45309', '#15803d']  # red, orange, green

# Draw anchors
for i, scale in enumerate(scales):
    for j, ratio in enumerate(ratios):
        # Compute width and height
        area = scale * scale
        h = np.sqrt(area * ratio)
        w = area / h
        
        # Top-left corner
        x1 = cx - w / 2
        y1 = cy - h / 2
        
        # Draw rectangle
        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=2,
            edgecolor=colors[i],
            facecolor='none',
            linestyle='--' if j == 0 else ('-.' if j == 1 else '-')
        )
        ax.add_patch(rect)

# Draw center point
ax.plot(cx, cy, 'o', color='white', markersize=10, zorder=10)
ax.text(cx, cy - 30, 'Anchor Center\n(i, j) = (32, 24)', 
       ha='center', va='top', fontsize=11, color='white',
       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

# Legend
legend_elements = [
    patches.Patch(facecolor='none', edgecolor=colors[0], linewidth=2, label='Scale 128² (small)'),
    patches.Patch(facecolor='none', edgecolor=colors[1], linewidth=2, label='Scale 256² (medium)'),
    patches.Patch(facecolor='none', edgecolor=colors[2], linewidth=2, label='Scale 512² (large)'),
    patches.Patch(facecolor='none', edgecolor='white', linewidth=2, linestyle='--', label='Ratio 1:2 (tall)'),
    patches.Patch(facecolor='none', edgecolor='white', linewidth=2, linestyle='-.', label='Ratio 1:1 (square)'),
    patches.Patch(facecolor='none', edgecolor='white', linewidth=2, linestyle='-', label='Ratio 2:1 (wide)')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)

# Title and labels
ax.set_title('RPN Anchor Boxes at One Spatial Location\n(9 anchors: 3 scales × 3 aspect ratios)', 
            fontsize=14, weight='bold', pad=20)
ax.set_xlabel('X coordinate (pixels)', fontsize=11)
ax.set_ylabel('Y coordinate (pixels)', fontsize=11)

# Set axis limits
ax.set_xlim(-50, 650)
ax.set_ylim(-50, 650)
ax.set_aspect('equal')
ax.grid(True, alpha=0.2)

# Add text box with anchor count
info_text = (
    "Total anchors per image:\n"
    "  Feature map: 64×48 locations\n"
    "  Anchors per location: 9\n"
    "  Total: 64×48×9 = 27,648\n"
    "  After NMS: ~300 proposals"
)
ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
       fontsize=10, va='top', ha='left',
       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7, edgecolor='#00d9ff', linewidth=2))

plt.tight_layout()
plt.savefig('../img/ch03-rpn-anchors.png', dpi=150, facecolor='#1a1a2e')
print("✓ Saved: ch03-rpn-anchors.png")

plt.close()
