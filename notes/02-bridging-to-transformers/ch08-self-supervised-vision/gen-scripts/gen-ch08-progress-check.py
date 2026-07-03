"""
Generate progress check dashboard for Ch.8.
Shows constraint achievements after DINO/MAE.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Dark theme setup
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#1a1a2e'
plt.rcParams['axes.facecolor'] = '#1a1a2e'

fig = plt.figure(figsize=(18, 11))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

# Main title
fig.suptitle('Ch.8 Progress Check: DINO & MAE Exceed Detection Target', 
            fontsize=20, weight='bold', y=0.98)

# ========== Constraint Status ==========
ax1 = fig.add_subplot(gs[0, :])
ax1.axis('off')

constraints = [
    ("1. Detection Accuracy", "86%", "#15803d", "✅ ACHIEVED (Ch.8)!"),
    ("2. Segmentation Quality", "71%", "#15803d", "✅ (Ch.6)"),
    ("3. Inference Latency", "95ms", "#b45309", "⚡ In progress (Ch.9-10)"),
    ("4. Model Size", "25 MB", "#b45309", "⚡ In progress (Ch.9-10)"),
    ("5. Data Efficiency", "<1k labels", "#15803d", "✅ (Ch.7-8)")
]

y_start = 0.85
for i, (name, value, color, status) in enumerate(constraints):
    y_pos = y_start - i * 0.17
    
    rect = mpatches.FancyBboxPatch((0.05, y_pos - 0.055), 0.9, 0.11,
                                  boxstyle="round,pad=0.01",
                                  edgecolor=color, facecolor=color,
                                  alpha=0.3, linewidth=2.5,
                                  transform=ax1.transAxes)
    ax1.add_patch(rect)
    
    ax1.text(0.08, y_pos, name, fontsize=14, weight='bold', 
            transform=ax1.transAxes, va='center', color='white')
    ax1.text(0.45, y_pos, value, fontsize=14, weight='bold',
            transform=ax1.transAxes, va='center', color=color)
    ax1.text(0.7, y_pos, status, fontsize=13,
            transform=ax1.transAxes, va='center', color='white')

# ========== mAP Evolution Chart ==========
ax2 = fig.add_subplot(gs[1, 0])

chapters = ['From\nScratch', 'Ch.4\nYOLO', 'Ch.7\nSimCLR', 'Ch.8\nDINO', 'Ch.8\nMAE']
map_scores = [72, 85, 84, 86, 87]
label_counts = [1000, 10000, 1000, 1000, 1000]
colors_bar = ['#b45309', '#b45309', '#3b82f6', '#15803d', '#15803d']

x_pos = np.arange(len(chapters))
bars = ax2.bar(x_pos, map_scores, color=colors_bar, edgecolor='white', linewidth=2)

for bar, score, labels in zip(bars, map_scores, label_counts):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{score}%\n({labels} labels)',
            ha='center', va='bottom', fontsize=10, weight='bold', color='white')

ax2.axhline(y=85, color='#b91c1c', linestyle='--', linewidth=2, alpha=0.7,
           label='Target: 85% mAP')

ax2.set_ylabel('mAP@0.5 (%)', fontsize=13, weight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(chapters, fontsize=11)
ax2.set_title('mAP Evolution: Self-Supervised Methods Excel', 
             fontsize=14, weight='bold')
ax2.legend(fontsize=11, loc='lower right')
ax2.set_ylim(65, 90)
ax2.grid(True, alpha=0.3, axis='y')

# ========== Attention Maps (Emergent Property) ==========
ax3 = fig.add_subplot(gs[1, 1])
ax3.axis('off')

ax3.text(0.5, 0.95, 'DINO Emergent Property: Attention Maps', 
        ha='center', va='top', fontsize=14, weight='bold',
        transform=ax3.transAxes, color='white')

# Simulate attention visualization
shelf_img = np.ones((100, 100, 3)) * 0.2
products = [(20, 40, 30, 70), (50, 70, 30, 70), (80, 100, 30, 70)]
for x1, x2, y1, y2 in products:
    shelf_img[y1:y2, x1:x2] = np.random.rand(3)

# Create attention overlay
attention = np.zeros((100, 100))
for x1, x2, y1, y2 in products:
    cx, cy = (x1+x2)//2, (y1+y2)//2
    Y, X = np.ogrid[:100, :100]
    attention += np.exp(-((X-cx)**2 + (Y-cy)**2) / 200)

attention = attention / attention.max()

# Display
from matplotlib.gridspec import GridSpecFromSubplotSpec
inner_gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 1], wspace=0.1)

ax3a = fig.add_subplot(inner_gs[0])
ax3a.imshow(shelf_img)
ax3a.set_title('Input', fontsize=11, weight='bold')
ax3a.axis('off')

ax3b = fig.add_subplot(inner_gs[1])
ax3b.imshow(attention, cmap='hot')
ax3b.set_title('Attention (no labels!)', fontsize=11, weight='bold')
ax3b.axis('off')

# Add insight text
ax3.text(0.5, 0.05, 'Network learned to focus on products\nwithout bounding box labels!',
        ha='center', va='bottom', fontsize=11, weight='bold',
        transform=ax3.transAxes,
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#15803d', alpha=0.9),
        color='white')

# ========== Methods Comparison Table ==========
ax4 = fig.add_subplot(gs[2, :])
ax4.axis('off')

table_data = [
    ['Method', 'mAP @ 1k labels', 'Requires Negatives?', 'Batch Size', 'Training Time', 'Key Advantage'],
    ['SimCLR', '84%', 'Yes (510/anchor)', '256 (4096 ideal)', '3 days', 'Simple, proven'],
    ['MoCo', '84%', 'Yes (65k queue)', '256', '3 days', 'Queue scales negatives'],
    ['DINO', '86% ✅', 'No', '256', '3 days', 'Attention emerges'],
    ['MAE', '87% ✅', 'No', '256', '3 days', 'Enables ViT, simple']
]

col_widths = [0.15, 0.15, 0.18, 0.13, 0.15, 0.24]
row_height = 0.18

for i, row in enumerate(table_data):
    y_pos = 0.88 - i * row_height
    
    for j, (cell, width) in enumerate(zip(row, col_widths)):
        x_pos = sum(col_widths[:j])
        
        if i == 0:
            bbox_props = dict(boxstyle='round,pad=0.4', facecolor='#1e3a8a', 
                            alpha=0.9, edgecolor='white', linewidth=1.5)
            weight = 'bold'
            fontsize = 10
        else:
            if '✅' in str(cell):
                bbox_props = dict(boxstyle='round,pad=0.4', facecolor='#15803d', 
                                alpha=0.4, edgecolor='gray', linewidth=1)
            else:
                bbox_props = dict(boxstyle='round,pad=0.4', facecolor='#1a1a2e', 
                                alpha=0.7, edgecolor='gray', linewidth=1)
            weight = 'normal'
            fontsize = 9
        
        ax4.text(x_pos + width/2, y_pos, cell,
                ha='center', va='center', fontsize=fontsize, weight=weight,
                transform=ax4.transAxes,
                bbox=bbox_props, color='white')

ax4.text(0.5, 0.02, '🎯 Key Result: DINO & MAE eliminate negative sampling and exceed 85% mAP target with <1k labels',
        ha='center', va='center', fontsize=13, weight='bold',
        transform=ax4.transAxes,
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#15803d', alpha=0.9),
        color='white')

plt.tight_layout()
plt.savefig('../img/ch08-progress-check.png', dpi=150, 
           bbox_inches='tight', facecolor='#1a1a2e')
print("Saved: ch08-progress-check.png")
plt.close()
