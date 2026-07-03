"""
Generate comparison chart: SimCLR vs DINO vs MAE.
"""

import matplotlib.pyplot as plt
import numpy as np

# Dark theme setup
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#1a1a2e'
plt.rcParams['axes.facecolor'] = '#1a1a2e'

# Comparison metrics
methods = ['From\nScratch', 'SimCLR\n(Ch.7)', 'DINO\n(Ch.8)', 'MAE\n(Ch.8)']
map_scores = [72, 84, 86, 87]
colors = ['#b45309', '#3b82f6', '#15803d', '#15803d']

# Training characteristics
requires_negatives = ['N/A', 'Yes\n(510 per anchor)', 'No', 'No']
batch_sizes = ['N/A', '256\n(4096 ideal)', '256', '256']
key_innovations = [
    'Supervised\nbaseline',
    'Contrastive\nloss',
    'Self-distillation\n+ attention',
    'Masked\nreconstruction'
]

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Panel 1: mAP comparison (main result)
ax1 = fig.add_subplot(gs[0, :])
bars = ax1.bar(methods, map_scores, color=colors, edgecolor='white', linewidth=2, width=0.6)

# Add value labels
for bar, score in zip(bars, map_scores):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{score}%',
            ha='center', va='bottom', fontsize=15, weight='bold', color='white')

# Target line
ax1.axhline(y=85, color='#b91c1c', linestyle='--', linewidth=2.5, 
           label='Target: 85% mAP', alpha=0.8)

# Highlight DINO and MAE exceeding target
ax1.annotate('Target\nEXCEEDED!', xy=(2, 86), xytext=(2.5, 88),
            arrowprops=dict(arrowstyle='->', color='white', lw=2),
            fontsize=13, color='#15803d', weight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#15803d', alpha=0.8))

ax1.set_ylabel('mAP@0.5 (%) with 1k Labeled Images', fontsize=14, weight='bold')
ax1.set_title('Detection Accuracy Comparison: Self-Supervised Methods Exceed Target', 
             fontsize=16, weight='bold', pad=20)
ax1.legend(fontsize=13, loc='upper left')
ax1.set_ylim(60, 92)
ax1.grid(True, alpha=0.3, axis='y')

# Panel 2: Training characteristics table
ax2 = fig.add_subplot(gs[1, 0])
ax2.axis('off')

table_data = [
    ['Method', 'Requires\nNegatives?', 'Batch Size', 'Key Innovation'],
    ['SimCLR', requires_negatives[1], batch_sizes[1], key_innovations[1]],
    ['DINO', requires_negatives[2], batch_sizes[2], key_innovations[2]],
    ['MAE', requires_negatives[3], batch_sizes[3], key_innovations[3]]
]

col_widths = [0.2, 0.3, 0.25, 0.25]
row_height = 0.2

for i, row in enumerate(table_data):
    y_pos = 0.9 - i * row_height
    
    for j, (cell, width) in enumerate(zip(row, col_widths)):
        x_pos = sum(col_widths[:j])
        
        # Header styling
        if i == 0:
            bbox_props = dict(boxstyle='round,pad=0.5', facecolor='#1e3a8a', 
                            alpha=0.9, edgecolor='white', linewidth=1.5)
            weight = 'bold'
            fontsize = 11
        else:
            bbox_props = dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e', 
                            alpha=0.7, edgecolor='gray', linewidth=1)
            weight = 'normal'
            fontsize = 10
        
        ax2.text(x_pos + width/2, y_pos, cell,
                ha='center', va='center', fontsize=fontsize, weight=weight,
                transform=ax2.transAxes,
                bbox=bbox_props, color='white')

ax2.text(0.5, 0.05, 'Key Insight: DINO & MAE eliminate negative sampling → simpler, more efficient',
        ha='center', va='center', fontsize=12, weight='bold',
        transform=ax2.transAxes,
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#15803d', alpha=0.9),
        color='white')

# Panel 3: Constraint achievements
ax3 = fig.add_subplot(gs[1, 1])
ax3.axis('off')

constraints = [
    '1. Detection Accuracy\nmAP@0.5 ≥ 85%',
    '2. Segmentation Quality\nIoU ≥ 70%',
    '3. Inference Latency\n<50ms',
    '4. Model Size\n<100 MB',
    '5. Data Efficiency\n<1k labels'
]

status_ch7 = ['❌ (84%)', '✅', '❌', '❌', '✅']
status_ch8 = ['✅ (86%)', '✅', '❌', '❌', '✅']

table_constraint = [
    ['Constraint', 'After Ch.7\n(SimCLR)', 'After Ch.8\n(DINO)'],
]

for constraint, s7, s8 in zip(constraints, status_ch7, status_ch8):
    table_constraint.append([constraint, s7, s8])

col_widths_c = [0.45, 0.275, 0.275]
row_height_c = 0.15

for i, row in enumerate(table_constraint):
    y_pos = 0.95 - i * row_height_c
    
    for j, (cell, width) in enumerate(zip(row, col_widths_c)):
        x_pos = sum(col_widths_c[:j])
        
        # Header styling
        if i == 0:
            bbox_props = dict(boxstyle='round,pad=0.4', facecolor='#1e3a8a', 
                            alpha=0.9, edgecolor='white', linewidth=1.5)
            weight = 'bold'
            fontsize = 10
        else:
            bbox_props = dict(boxstyle='round,pad=0.4', facecolor='#1a1a2e', 
                            alpha=0.7, edgecolor='gray', linewidth=1)
            weight = 'normal'
            fontsize = 9
        
        # Color code status
        if '✅' in str(cell) and i > 0:
            bbox_props['facecolor'] = '#15803d'
            bbox_props['alpha'] = 0.4
        
        ax3.text(x_pos + width/2, y_pos, cell,
                ha='center', va='center', fontsize=fontsize, weight=weight,
                transform=ax3.transAxes,
                bbox=bbox_props, color='white')

# Overall title
fig.suptitle('Ch.8 Self-Supervised Vision: DINO & MAE', 
            fontsize=18, weight='bold', y=0.98)

plt.tight_layout()
plt.savefig('../img/ch08-comparison.png', dpi=150, 
           bbox_inches='tight', facecolor='#1a1a2e')
print("Saved: ch08-comparison.png")
plt.close()
