"""
Generate detection mAP comparison chart.

Compares Faster R-CNN performance against baselines (ResNet classifier, 
one-stage detectors from Ch.4).
"""

import matplotlib.pyplot as plt
import numpy as np

# Dark theme
plt.style.use('dark_background')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='#1a1a2e')
for ax in [ax1, ax2]:
    ax.set_facecolor('#1a1a2e')

# ─────────────────────────────────────────────────────────────
# Chart 1: mAP@0.5 Comparison
# ─────────────────────────────────────────────────────────────
methods = ['ResNet-50\nClassifier', 'Faster R-CNN\n(ResNet-50)', 
           'YOLOv5s\n(Ch.4)', 'RetinaNet\n(Ch.4)']
map_scores = [0.0, 86.3, 82.1, 84.5]  # ResNet can't do detection
colors = ['#666', '#15803d', '#1d4ed8', '#7c3aed']

bars = ax1.bar(range(len(methods)), map_scores, color=colors, alpha=0.8, width=0.6)

# Highlight Faster R-CNN (current chapter)
bars[1].set_edgecolor('#00d9ff')
bars[1].set_linewidth(3)

# Add threshold line
ax1.axhline(85, color='#b91c1c', linestyle='--', linewidth=2, label='Target: 85% mAP')

# Add value labels
for i, (bar, score) in enumerate(zip(bars, map_scores)):
    if score > 0:
        label = f'{score:.1f}%'
        ax1.text(bar.get_x() + bar.get_width()/2, score + 1.5, label,
                ha='center', va='bottom', fontsize=11, weight='bold', color='white')
    else:
        ax1.text(bar.get_x() + bar.get_width()/2, 5, 'N/A\n(classification\nonly)',
                ha='center', va='bottom', fontsize=9, color='#666', style='italic')

ax1.set_ylabel('mAP@0.5 (%)', fontsize=12, weight='bold')
ax1.set_title('Detection Accuracy Comparison\n(Retail Shelf Dataset)', 
             fontsize=13, weight='bold')
ax1.set_xticks(range(len(methods)))
ax1.set_xticklabels(methods, fontsize=10)
ax1.set_ylim(0, 95)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.2, axis='y')

# ─────────────────────────────────────────────────────────────
# Chart 2: Inference Time vs Accuracy Trade-off
# ─────────────────────────────────────────────────────────────
detectors = [
    {'name': 'Faster R-CNN\n(ResNet-50)', 'map': 86.3, 'time': 180, 'color': '#15803d', 'marker': 'o', 'size': 150},
    {'name': 'Faster R-CNN\n(ResNet-101)', 'map': 89.1, 'time': 250, 'color': '#15803d', 'marker': '^', 'size': 120},
    {'name': 'YOLOv5s', 'map': 82.1, 'time': 15, 'color': '#1d4ed8', 'marker': 's', 'size': 120},
    {'name': 'YOLOv5m', 'map': 84.8, 'time': 28, 'color': '#1d4ed8', 'marker': 'D', 'size': 120},
    {'name': 'RetinaNet', 'map': 84.5, 'time': 95, 'color': '#7c3aed', 'marker': 'p', 'size': 120}
]

for det in detectors:
    ax2.scatter(det['time'], det['map'], 
               color=det['color'], marker=det['marker'], 
               s=det['size'], alpha=0.8, edgecolors='white', linewidths=1.5,
               label=det['name'])

# Highlight current chapter detector
ax2.scatter(180, 86.3, s=300, facecolors='none', 
           edgecolors='#00d9ff', linewidths=3, zorder=10)

# Constraint lines
ax2.axhline(85, color='#b91c1c', linestyle='--', linewidth=2, alpha=0.7, label='Target mAP: 85%')
ax2.axvline(50, color='#b45309', linestyle='--', linewidth=2, alpha=0.7, label='Target latency: 50ms')

# Shade "ideal" region (fast + accurate)
ax2.fill_between([0, 50], 85, 100, alpha=0.1, color='#15803d', label='Ideal zone')

ax2.set_xlabel('Inference Time (ms per frame)', fontsize=12, weight='bold')
ax2.set_ylabel('mAP@0.5 (%)', fontsize=12, weight='bold')
ax2.set_title('Speed vs Accuracy Trade-off\n(NVIDIA RTX 3090)', 
             fontsize=13, weight='bold')
ax2.set_xlim(0, 300)
ax2.set_ylim(80, 92)
ax2.legend(loc='lower right', fontsize=9, framealpha=0.9)
ax2.grid(True, alpha=0.2)

# Add annotation
ax2.annotate('Two-stage:\nHigh accuracy\nSlow inference', 
            xy=(180, 86.3), xytext=(220, 90),
            arrowprops=dict(arrowstyle='->', color='white', lw=1.5),
            fontsize=9, color='white', ha='left',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

plt.tight_layout()
plt.savefig('../img/ch03-map-comparison.png', dpi=150, facecolor='#1a1a2e')
print("✓ Saved: ch03-map-comparison.png")

plt.close()
