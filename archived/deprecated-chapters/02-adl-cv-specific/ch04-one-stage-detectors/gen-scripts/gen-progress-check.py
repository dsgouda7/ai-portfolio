"""
Generate progress check visualization for ProductionCV constraints.

Shows which constraints are achieved after Ch.4 (one-stage detectors).
"""

import matplotlib.pyplot as plt
import numpy as np

# Dark theme
plt.style.use('dark_background')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor='#1a1a2e')
for ax in [ax1, ax2]:
    ax.set_facecolor('#1a1a2e')

# ─────────────────────────────────────────────────────────────
# Chart 1: Constraint Progress Bars
# ─────────────────────────────────────────────────────────────
constraints = [
    {'name': 'Detection\nAccuracy', 'target': 85, 'ch3': 86.3, 'ch4': 82.1, 'unit': '%'},
    {'name': 'Segmentation\nQuality', 'target': 70, 'ch3': 0, 'ch4': 0, 'unit': '%'},
    {'name': 'Inference\nLatency', 'target': 50, 'ch3': 180, 'ch4': 18, 'unit': 'ms', 'invert': True},
    {'name': 'Model\nSize', 'target': 100, 'ch3': 167, 'ch4': 14, 'unit': 'MB', 'invert': True},
    {'name': 'Data\nEfficiency', 'target': 1000, 'ch3': 1000, 'ch4': 1000, 'unit': 'imgs', 'invert': True}
]

def normalize(val, target, invert=False):
    """Normalize to 0-100 scale (100 = target achieved)."""
    if val == 0:
        return 0
    if invert:
        return min(100, (target / val) * 100)
    else:
        return min(100, (val / target) * 100)

x = np.arange(len(constraints))
width = 0.25

# Compute normalized values
targets = [100] * len(constraints)
ch3_norm = [normalize(c['ch3'], c['target'], c.get('invert', False)) for c in constraints]
ch4_norm = [normalize(c['ch4'], c['target'], c.get('invert', False)) for c in constraints]

# Plot bars
bars1 = ax1.bar(x - width, targets, width, label='Target (100%)',
               color='#444', alpha=0.4)
bars2 = ax1.bar(x, ch3_norm, width, label='Ch.3 (Faster R-CNN)',
               color='#7c3aed', alpha=0.8)
bars3 = ax1.bar(x + width, ch4_norm, width, label='Ch.4 (YOLOv5)',
               color='#15803d', alpha=0.8)

# Highlight achieved constraints (≥95% of target)
for i, val in enumerate(ch4_norm):
    if val >= 95:
        bars3[i].set_edgecolor('#00d9ff')
        bars3[i].set_linewidth(3)

# Add value labels
for i, c in enumerate(constraints):
    # Ch.4 actual value
    label = f"{c['ch4']:.1f}{c['unit']}" if c['ch4'] < 100 else f"{int(c['ch4'])}{c['unit']}"
    y_pos = ch4_norm[i] + 3
    color = '#00d9ff' if ch4_norm[i] >= 95 else 'white'
    marker = '✓ ' if ch4_norm[i] >= 95 else ''
    ax1.text(i + width, y_pos, f"{marker}{label}",
            ha='center', va='bottom', fontsize=10, weight='bold', color=color)

ax1.set_ylabel('Progress (%)', fontsize=12, weight='bold')
ax1.set_title('ProductionCV Constraint Progress\n(Ch.3 vs Ch.4)', 
             fontsize=14, weight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([c['name'] for c in constraints], fontsize=11)
ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)
ax1.set_ylim(0, 120)
ax1.grid(True, alpha=0.2, axis='y')

# Achievement summary
achieved_ch3 = sum(1 for v in ch3_norm if v >= 95)
achieved_ch4 = sum(1 for v in ch4_norm if v >= 95)

summary_text = (
    f"Constraints Achieved:\n"
    f"  Ch.3: {achieved_ch3}/5\n"
    f"  Ch.4: {achieved_ch4}/5\n\n"
    f"New in Ch.4:\n"
    f"✓ #3: Latency (<50ms)\n"
    f"✓ #4: Size (<100MB)\n\n"
    f"Speed gain: 10× faster"
)
ax1.text(0.98, 0.97, summary_text, transform=ax1.transAxes,
        fontsize=10, va='top', ha='right',
        bbox=dict(boxstyle='round', facecolor='black', alpha=0.9,
                 edgecolor='#15803d', linewidth=2))

# ─────────────────────────────────────────────────────────────
# Chart 2: Speed vs Accuracy Trade-off
# ─────────────────────────────────────────────────────────────
detectors = [
    {'name': 'Faster R-CNN\nResNet-50', 'map': 86.3, 'time': 180, 
     'color': '#7c3aed', 'marker': 'o', 'size': 150},
    {'name': 'Faster R-CNN\nResNet-101', 'map': 89.1, 'time': 250,
     'color': '#7c3aed', 'marker': '^', 'size': 120},
    {'name': 'YOLOv5s', 'map': 82.1, 'time': 18,
     'color': '#15803d', 'marker': 's', 'size': 200},
    {'name': 'YOLOv5m', 'map': 84.8, 'time': 28,
     'color': '#15803d', 'marker': 'D', 'size': 150},
    {'name': 'YOLOv5l', 'map': 86.2, 'time': 45,
     'color': '#15803d', 'marker': 'p', 'size': 120},
    {'name': 'RetinaNet\nResNet-50', 'map': 84.5, 'time': 95,
     'color': '#b45309', 'marker': 'h', 'size': 120},
]

# Plot detectors
for det in detectors:
    ax2.scatter(det['time'], det['map'],
               color=det['color'], marker=det['marker'],
               s=det['size'], alpha=0.8, edgecolors='white', linewidths=1.5,
               label=det['name'])

# Highlight YOLOv5s (current chapter star)
ax2.scatter(18, 82.1, s=400, facecolors='none',
           edgecolors='#00d9ff', linewidths=3, zorder=10)

# Constraint target lines
ax2.axhline(85, color='#b91c1c', linestyle='--', linewidth=2, alpha=0.7,
           label='Target mAP: 85%')
ax2.axvline(50, color='#b91c1c', linestyle='--', linewidth=2, alpha=0.7,
           label='Target latency: 50ms')

# Shade ideal region
ax2.fill_between([0, 50], 85, 100, alpha=0.1, color='#15803d')
ax2.text(25, 91, 'Ideal Zone\n(Fast + Accurate)', ha='center', va='center',
        fontsize=11, color='#15803d', weight='bold',
        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7,
                 edgecolor='#15803d', linewidth=2))

# Annotations
ax2.annotate('Two-stage:\nHigh accuracy\nSlow (180ms)',
            xy=(180, 86.3), xytext=(210, 88.5),
            arrowprops=dict(arrowstyle='->', color='white', lw=1.5),
            fontsize=10, color='white', ha='left',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

ax2.annotate('YOLOv5s:\n10× faster!\nSmall accuracy drop',
            xy=(18, 82.1), xytext=(60, 83),
            arrowprops=dict(arrowstyle='->', color='#00d9ff', lw=2),
            fontsize=10, color='#00d9ff', weight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.8,
                     edgecolor='#00d9ff', linewidth=2))

ax2.set_xlabel('Inference Time (ms per frame)', fontsize=12, weight='bold')
ax2.set_ylabel('mAP@0.5 (%)', fontsize=12, weight='bold')
ax2.set_title('Speed vs Accuracy Trade-off\n(NVIDIA RTX 3090)',
             fontsize=14, weight='bold')
ax2.set_xlim(0, 280)
ax2.set_ylim(80, 92)
ax2.legend(loc='lower right', fontsize=9, framealpha=0.9, ncol=2)
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('../img/ch04-progress-check.png', dpi=150, facecolor='#1a1a2e')
print("✓ Saved: ch04-progress-check.png")

plt.close()
