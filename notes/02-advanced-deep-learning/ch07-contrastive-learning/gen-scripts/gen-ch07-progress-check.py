"""
Generate progress check dashboard for Ch.7.
Shows constraint achievements after contrastive learning.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Dark theme setup
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#1a1a2e'
plt.rcParams['axes.facecolor'] = '#1a1a2e'

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4)

# Main title
fig.suptitle('Ch.7 Progress Check: Contrastive Learning Unlocks Data Efficiency', 
            fontsize=20, weight='bold', y=0.98)

# ========== Constraint Status Dashboard ==========
ax1 = fig.add_subplot(gs[0, :])
ax1.axis('off')

constraints = [
    ("1. Detection Accuracy", "85%", "#15803d", "✅ (Ch.4)"),
    ("2. Segmentation Quality", "71%", "#15803d", "✅ (Ch.6)"),
    ("3. Inference Latency", "95ms", "#b45309", "⚡ In progress"),
    ("4. Model Size", "25 MB", "#b45309", "⚡ In progress"),
    ("5. Data Efficiency", "<1k labels", "#15803d", "✅ ACHIEVED (Ch.7)!")
]

y_start = 0.8
for i, (name, value, color, status) in enumerate(constraints):
    y_pos = y_start - i * 0.18
    
    # Constraint box
    rect = mpatches.FancyBboxPatch((0.05, y_pos - 0.06), 0.9, 0.12,
                                  boxstyle="round,pad=0.01",
                                  edgecolor=color, facecolor=color,
                                  alpha=0.3, linewidth=2,
                                  transform=ax1.transAxes)
    ax1.add_patch(rect)
    
    # Text
    ax1.text(0.08, y_pos, name, fontsize=13, weight='bold', 
            transform=ax1.transAxes, va='center', color='white')
    ax1.text(0.5, y_pos, value, fontsize=13, weight='bold',
            transform=ax1.transAxes, va='center', color=color)
    ax1.text(0.75, y_pos, status, fontsize=12,
            transform=ax1.transAxes, va='center', color='white')

# ========== Data Efficiency Curve ==========
ax2 = fig.add_subplot(gs[1, :2])

label_counts = np.array([100, 500, 1000, 2000, 5000, 10000])
map_from_scratch = np.array([42, 58, 72, 76, 81, 85])
map_pretrained = np.array([62, 74, 84, 87, 89, 91])

ax2.plot(label_counts, map_from_scratch, marker='o', markersize=8,
        linewidth=2.5, label='From Scratch', color='#b45309')
ax2.plot(label_counts, map_pretrained, marker='s', markersize=8,
        linewidth=2.5, label='SimCLR Pretrained', color='#15803d')
ax2.axhline(y=85, color='#b91c1c', linestyle='--', linewidth=2, alpha=0.6)
ax2.scatter([1000], [84], s=300, color='#15803d', marker='*',
           zorder=5, edgecolors='white', linewidths=2)

ax2.set_xscale('log')
ax2.set_xlabel('Labeled Images', fontsize=12, weight='bold')
ax2.set_ylabel('mAP@0.5 (%)', fontsize=12, weight='bold')
ax2.set_title('Data Efficiency Gain', fontsize=14, weight='bold')
ax2.legend(fontsize=11, loc='lower right')
ax2.grid(True, alpha=0.3, which='both')

# ========== Cost Comparison ==========
ax3 = fig.add_subplot(gs[1, 2])

strategies = ['From Scratch\n(10k labels)', 'SimCLR\n(1k labels)']
costs = [50000, 5000]
colors_bar = ['#b45309', '#15803d']
bars = ax3.bar(strategies, costs, color=colors_bar, edgecolor='white', linewidth=2)

# Add value labels on bars
for bar, cost in zip(bars, costs):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'${cost/1000:.0f}k',
            ha='center', va='bottom', fontsize=13, weight='bold', color='white')

ax3.set_ylabel('Labeling Cost ($)', fontsize=12, weight='bold')
ax3.set_title('Cost Reduction: 10×', fontsize=14, weight='bold')
ax3.set_ylim(0, 60000)
ax3.grid(True, alpha=0.3, axis='y')

# ========== Key Metrics Table ==========
ax4 = fig.add_subplot(gs[2, :])
ax4.axis('off')

metrics_data = [
    ["Metric", "Baseline (Ch.4)", "After Ch.7", "Improvement"],
    ["mAP @ 1k labels", "72%", "84%", "+12% 🎯"],
    ["Labels for 85% mAP", "10,000", "~1,500", "6.7× reduction 🎯"],
    ["Labeling cost", "$50k", "$7.5k", "$42.5k saved 💰"],
    ["Pretraining time", "N/A", "3 days (4× V100)", "One-time cost"],
    ["Unlabeled images", "0 (wasted)", "50k (leveraged)", "50k → useful! 🚀"]
]

# Draw table
col_widths = [0.25, 0.25, 0.25, 0.25]
row_height = 0.15

for i, row in enumerate(metrics_data):
    y_pos = 0.9 - i * row_height
    
    for j, (cell, width) in enumerate(zip(row, col_widths)):
        x_pos = sum(col_widths[:j])
        
        # Header styling
        if i == 0:
            bbox_props = dict(boxstyle='round,pad=0.5', facecolor='#1e3a8a', alpha=0.8, edgecolor='white', linewidth=1.5)
            weight = 'bold'
        else:
            bbox_props = dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e', alpha=0.6, edgecolor='gray', linewidth=1)
            weight = 'normal'
        
        ax4.text(x_pos + width/2, y_pos, cell,
                ha='center', va='center', fontsize=11, weight=weight,
                transform=ax4.transAxes,
                bbox=bbox_props, color='white')

plt.tight_layout()
plt.savefig('../img/ch07-progress-check.png', dpi=150,
           bbox_inches='tight', facecolor='#1a1a2e')
print("Saved: ch07-progress-check.png")
plt.close()
