"""
Generate progress check visualization for Ch.1 ResNets.

Output: ch01-progress-check.png

Shows ProductionCV constraint progress after implementing ResNets.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Dark theme
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(14, 8), facecolor='#1a1a2e')
ax.set_facecolor('#1a1a2e')
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(7, 9.5, 'ProductionCV Progress — After Ch.1 (ResNets)', 
        ha='center', va='top', fontsize=18, fontweight='bold', color='white')

# Constraints with progress bars
constraints = [
    ('Constraint #1\nDetection Accuracy', 'mAP@0.5 ≥ 85%', 78.2, 85, '#10b981'),
    ('Constraint #2\nSegmentation Quality', 'IoU ≥ 70%', 0, 70, '#6b7280'),
    ('Constraint #3\nInference Latency', '<50ms', 85, 50, '#ef4444'),  # Inverted (lower is better)
    ('Constraint #4\nModel Size', '<100 MB', 98, 100, '#f59e0b'),  # Inverted
    ('Constraint #5\nData Efficiency', '<1k labels', 0, 1, '#6b7280'),  # Not started
]

y_start = 8
y_step = 1.5

for i, (name, target, current, goal, color) in enumerate(constraints):
    y = y_start - i * y_step
    
    # Constraint label
    ax.text(0.5, y + 0.3, name, ha='left', va='top', fontsize=11, color='white', fontweight='bold')
    ax.text(0.5, y - 0.1, target, ha='left', va='top', fontsize=9, color='#9ca3af')
    
    # Progress bar background
    bar_x_start = 3.5
    bar_width = 8
    bar_height = 0.4
    
    bg_rect = patches.Rectangle((bar_x_start, y - 0.2), bar_width, bar_height,
                                 linewidth=1, edgecolor='#6b7280', facecolor='#374151')
    ax.add_patch(bg_rect)
    
    # Progress bar fill
    if current > 0:
        # For latency and size (inverted metrics), show progress as (goal/current)
        if 'Latency' in name or 'Size' in name:
            progress_ratio = min(goal / current, 1.0) if current > 0 else 0
            status_text = f'{current:.0f} (need ≤{goal})'
        else:
            progress_ratio = min(current / goal, 1.0) if goal > 0 else 0
            status_text = f'{current:.1f} / {goal}'
        
        fill_width = bar_width * progress_ratio
        fill_rect = patches.Rectangle((bar_x_start, y - 0.2), fill_width, bar_height,
                                       linewidth=0, facecolor=color, alpha=0.8)
        ax.add_patch(fill_rect)
        
        # Status text
        ax.text(bar_x_start + bar_width + 0.3, y, status_text, ha='left', va='center',
                fontsize=10, color=color, fontweight='bold')
        
        # Check mark if completed
        if progress_ratio >= 1.0:
            ax.text(bar_x_start + bar_width + 2.5, y, '✅', ha='center', va='center', fontsize=16)
    else:
        # Not started
        ax.text(bar_x_start + bar_width + 0.3, y, 'Not started', ha='left', va='center',
                fontsize=10, color='#6b7280', style='italic')

# Summary box
summary_y = 1.5
summary_box = patches.FancyBboxPatch((0.5, summary_y - 0.7), 13, 1.4, boxstyle="round,pad=0.2",
                                     edgecolor='#3b82f6', facecolor='#1e3a8a', linewidth=2, alpha=0.5)
ax.add_patch(summary_box)

ax.text(1, summary_y + 0.3, '🎯 Key Achievements:', ha='left', va='top', fontsize=12, color='#3b82f6', fontweight='bold')
ax.text(1, summary_y - 0.1, '  • Gradient flow solved — 100+ layer networks train successfully', 
        ha='left', va='top', fontsize=10, color='white')
ax.text(1, summary_y - 0.4, '  • ResNet-50 baseline: 78.2% mAP (6.8 points from target)', 
        ha='left', va='top', fontsize=10, color='white')

ax.text(8, summary_y + 0.3, '⚠️ Still Blocked:', ha='left', va='top', fontsize=12, color='#f59e0b', fontweight='bold')
ax.text(8, summary_y - 0.1, '  • Model too large (98 MB) and slow (85ms)', 
        ha='left', va='top', fontsize=10, color='white')
ax.text(8, summary_y - 0.4, '  • Next: Ch.2 (Efficient Architectures) → 5× smaller, 3× faster', 
        ha='left', va='top', fontsize=10, color='white')

plt.tight_layout()
plt.savefig('../img/ch01-progress-check.png', dpi=150, facecolor='#1a1a2e', bbox_inches='tight')
print('✅ Generated: ../img/ch01-progress-check.png')
plt.close()
