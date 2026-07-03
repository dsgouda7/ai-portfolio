"""
Generate depthwise separable convolution visualization.

Output: ch02-depthwise-separable.png

Shows how standard 3×3 conv is factorized into depthwise + pointwise.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Dark theme
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(14, 10), facecolor='#1a1a2e')
ax.set_facecolor('#1a1a2e')
ax.set_xlim(0, 14)
ax.set_ylim(0, 12)
ax.axis('off')

# Title
ax.text(7, 11.5, 'Depthwise Separable Convolution Breakdown', 
        ha='center', va='top', fontsize=18, fontweight='bold', color='white')

# --- Standard Conv (top) ---
ax.text(7, 10.5, 'Standard 3×3 Convolution', ha='center', fontsize=14, color='#ef4444', fontweight='bold')

# Input
input_box1 = patches.FancyBboxPatch((0.5, 9), 2, 1, boxstyle="round,pad=0.1",
                                    edgecolor='#3b82f6', facecolor='#1e3a8a', linewidth=2)
ax.add_patch(input_box1)
ax.text(1.5, 9.5, '56×56×128\nInput', ha='center', va='center', fontsize=10, color='white', fontweight='bold')

# Standard conv operation
conv_box1 = patches.FancyBboxPatch((4, 9), 3.5, 1, boxstyle="round,pad=0.1",
                                   edgecolor='#ef4444', facecolor='#b91c1c', linewidth=2)
ax.add_patch(conv_box1)
ax.text(5.75, 9.7, '3×3 Conv', ha='center', va='top', fontsize=11, color='white', fontweight='bold')
ax.text(5.75, 9.3, '256 kernels (3×3×128 each)', ha='center', va='center', fontsize=8, color='white')

# Output
output_box1 = patches.FancyBboxPatch((9, 9), 2, 1, boxstyle="round,pad=0.1",
                                     edgecolor='#3b82f6', facecolor='#1e3a8a', linewidth=2)
ax.add_patch(output_box1)
ax.text(10, 9.5, '56×56×256\nOutput', ha='center', va='center', fontsize=10, color='white', fontweight='bold')

# Arrows
ax.arrow(2.7, 9.5, 1, 0, head_width=0.2, head_length=0.15, fc='white', ec='white', linewidth=2)
ax.arrow(7.7, 9.5, 1, 0, head_width=0.2, head_length=0.15, fc='white', ec='white', linewidth=2)

# Cost annotation
cost_box1 = patches.FancyBboxPatch((11.5, 9.1), 2.2, 0.8, boxstyle="round,pad=0.1",
                                   edgecolor='#ef4444', facecolor='#1a1a2e', linewidth=2, linestyle='--')
ax.add_patch(cost_box1)
ax.text(12.6, 9.5, 'FLOPs: 925M\nParams: 295k', ha='center', va='center', fontsize=9, color='#ef4444', fontweight='bold')

# --- Depthwise Separable (bottom) ---
ax.text(7, 7.5, 'Depthwise Separable (Factorized)', ha='center', fontsize=14, color='#10b981', fontweight='bold')

# Input
input_box2 = patches.FancyBboxPatch((0.5, 3.5), 2, 1, boxstyle="round,pad=0.1",
                                    edgecolor='#3b82f6', facecolor='#1e3a8a', linewidth=2)
ax.add_patch(input_box2)
ax.text(1.5, 4, '56×56×128\nInput', ha='center', va='center', fontsize=10, color='white', fontweight='bold')

# Step 1: Depthwise
depthwise_box = patches.FancyBboxPatch((3.5, 5.5), 3, 1.2, boxstyle="round,pad=0.1",
                                       edgecolor='#f59e0b', facecolor='#b45309', linewidth=2)
ax.add_patch(depthwise_box)
ax.text(5, 6.5, 'Step 1: Depthwise 3×3', ha='center', va='top', fontsize=11, color='white', fontweight='bold')
ax.text(5, 6.0, '128 filters (3×3×1 each)', ha='center', va='center', fontsize=8, color='white')
ax.text(5, 5.7, 'One filter per channel', ha='center', va='center', fontsize=8, color='#fbbf24', style='italic')

# Intermediate
inter_box = patches.FancyBboxPatch((3.5, 3.5), 3, 1, boxstyle="round,pad=0.1",
                                   edgecolor='#6b7280', facecolor='#374151', linewidth=2)
ax.add_patch(inter_box)
ax.text(5, 4, '56×56×128\nIntermediate', ha='center', va='center', fontsize=9, color='white')

# Step 2: Pointwise
pointwise_box = patches.FancyBboxPatch((7.5, 5.5), 3, 1.2, boxstyle="round,pad=0.1",
                                       edgecolor='#10b981', facecolor='#15803d', linewidth=2)
ax.add_patch(pointwise_box)
ax.text(9, 6.5, 'Step 2: Pointwise 1×1', ha='center', va='top', fontsize=11, color='white', fontweight='bold')
ax.text(9, 6.0, '256 filters (1×1×128 each)', ha='center', va='center', fontsize=8, color='white')
ax.text(9, 5.7, 'Mix channels only', ha='center', va='center', fontsize=8, color='#34d399', style='italic')

# Output
output_box2 = patches.FancyBboxPatch((7.5, 3.5), 3, 1, boxstyle="round,pad=0.1",
                                     edgecolor='#3b82f6', facecolor='#1e3a8a', linewidth=2)
ax.add_patch(output_box2)
ax.text(9, 4, '56×56×256\nOutput', ha='center', va='center', fontsize=10, color='white', fontweight='bold')

# Arrows
ax.arrow(2.7, 4, 0.6, 0, head_width=0.2, head_length=0.15, fc='white', ec='white', linewidth=2)
ax.arrow(5, 5.3, 0, -0.8, head_width=0.15, head_length=0.1, fc='white', ec='white', linewidth=2)
ax.arrow(6.7, 4, 0.6, 0, head_width=0.2, head_length=0.15, fc='white', ec='white', linewidth=2)
ax.arrow(9, 5.3, 0, -0.8, head_width=0.15, head_length=0.1, fc='white', ec='white', linewidth=2)

# Cost annotations
cost_dw = patches.FancyBboxPatch((3.5, 2.2), 3, 0.8, boxstyle="round,pad=0.1",
                                 edgecolor='#f59e0b', facecolor='#1a1a2e', linewidth=2, linestyle='--')
ax.add_patch(cost_dw)
ax.text(5, 2.6, 'FLOPs: 3.6M\nParams: 1.2k', ha='center', va='center', fontsize=9, color='#f59e0b', fontweight='bold')

cost_pw = patches.FancyBboxPatch((7.5, 2.2), 3, 0.8, boxstyle="round,pad=0.1",
                                 edgecolor='#10b981', facecolor='#1a1a2e', linewidth=2, linestyle='--')
ax.add_patch(cost_pw)
ax.text(9, 2.6, 'FLOPs: 103M\nParams: 33k', ha='center', va='center', fontsize=9, color='#10b981', fontweight='bold')

# Total cost
total_cost = patches.FancyBboxPatch((11, 3), 2.5, 1.5, boxstyle="round,pad=0.15",
                                    edgecolor='#10b981', facecolor='#1a1a2e', linewidth=3)
ax.add_patch(total_cost)
ax.text(12.25, 4.2, 'Total:', ha='center', va='top', fontsize=11, color='#10b981', fontweight='bold')
ax.text(12.25, 3.7, 'FLOPs: 107M', ha='center', va='center', fontsize=10, color='white')
ax.text(12.25, 3.3, 'Params: 34k', ha='center', va='center', fontsize=10, color='white')

# Speedup arrow
ax.annotate('', xy=(12.25, 8.8), xytext=(12.25, 4.7),
            arrowprops=dict(arrowstyle='<->', color='#10b981', lw=3))
ax.text(13.2, 6.75, '8.7×\nSpeedup!', ha='center', va='center', fontsize=14, 
        color='#10b981', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e', edgecolor='#10b981', linewidth=2))

# Formula at bottom
formula_box = patches.FancyBboxPatch((1, 0.3), 12, 1, boxstyle="round,pad=0.15",
                                     edgecolor='#3b82f6', facecolor='#1e3a8a', linewidth=2, alpha=0.5)
ax.add_patch(formula_box)
formula_text = r'Speedup = $\frac{k^2 \cdot C_{out}}{k^2 + C_{out}}$ = $\frac{9 \times 256}{9 + 256}$ = 8.7×'
ax.text(7, 0.8, formula_text, ha='center', va='center', fontsize=13, 
        color='white', fontweight='bold')

plt.tight_layout()
plt.savefig('../img/ch02-depthwise-separable.png', dpi=150, facecolor='#1a1a2e', bbox_inches='tight')
print('✅ Generated: ../img/ch02-depthwise-separable.png')
plt.close()
