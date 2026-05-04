"""
Generate data efficiency comparison chart.
Shows mAP vs number of labeled images for:
- From scratch training (Ch.4 YOLO baseline)
- SimCLR pretrained + fine-tuning
"""

import numpy as np
import matplotlib.pyplot as plt

# Dark theme setup
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#1a1a2e'
plt.rcParams['axes.facecolor'] = '#1a1a2e'

# Data points
label_counts = np.array([100, 500, 1000, 2000, 5000, 10000])
map_from_scratch = np.array([42, 58, 72, 76, 81, 85])
map_pretrained = np.array([62, 74, 84, 87, 89, 91])

# Create figure
fig, ax = plt.subplots(figsize=(14, 8))

# Plot curves
ax.plot(label_counts, map_from_scratch, marker='o', markersize=10, 
        linewidth=3, label='From Scratch (Ch.4 YOLO)', color='#b45309')
ax.plot(label_counts, map_pretrained, marker='s', markersize=10, 
        linewidth=3, label='SimCLR Pretrained + Fine-tuned', color='#15803d')

# Target line
ax.axhline(y=85, color='#b91c1c', linestyle='--', linewidth=2, 
          label='Target: 85% mAP (Constraint #1)', alpha=0.7)

# Highlight key point: SimCLR at 1k labels
ax.scatter([1000], [84], s=500, color='#15803d', marker='*', 
          zorder=5, edgecolors='white', linewidths=2.5, 
          label='SimCLR @ 1k labels: 84% mAP')

# Annotations
ax.annotate('10× cost reduction!\n($50k → $5k)', 
           xy=(1000, 84), xytext=(2500, 79),
           arrowprops=dict(arrowstyle='->', color='white', lw=2),
           fontsize=12, color='white', weight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#15803d', alpha=0.8))

ax.annotate('From scratch needs\n10k labels to hit 85%', 
           xy=(10000, 85), xytext=(6000, 88),
           arrowprops=dict(arrowstyle='->', color='white', lw=2),
           fontsize=11, color='white',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#b45309', alpha=0.8))

# Formatting
ax.set_xscale('log')
ax.set_xlabel('Number of Labeled Training Images (log scale)', fontsize=14, weight='bold')
ax.set_ylabel('mAP@0.5 (%)', fontsize=14, weight='bold')
ax.set_title('Data Efficiency: SimCLR Self-Supervised Pretraining vs From Scratch', 
            fontsize=16, weight='bold', pad=20)
ax.legend(fontsize=12, loc='lower right', framealpha=0.9)
ax.grid(True, alpha=0.3, which='both', linestyle='--')
ax.set_ylim(35, 95)

# Add constraint badge
constraint_text = "✅ Constraint #5\nData Efficiency\n<1,000 labels"
ax.text(0.02, 0.98, constraint_text, transform=ax.transAxes,
       fontsize=11, verticalalignment='top', weight='bold',
       bbox=dict(boxstyle='round,pad=0.8', facecolor='#15803d', alpha=0.9),
       color='white')

plt.tight_layout()
plt.savefig('../img/ch07-data-efficiency.png', dpi=150, 
           bbox_inches='tight', facecolor='#1a1a2e')
print("Saved: ch07-data-efficiency.png")
plt.close()
