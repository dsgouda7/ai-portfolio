"""
Generate ResNet architecture diagram showing skip connections.

Output: ch01-resnet-architecture.png

Shows the structure of a residual block with F(x) and skip connection (+x).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Dark theme
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(12, 8), facecolor='#1a1a2e')
ax.set_facecolor('#1a1a2e')
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(6, 9.5, 'ResNet Residual Block Architecture', 
        ha='center', va='top', fontsize=18, fontweight='bold', color='white')

# Input
input_box = patches.FancyBboxPatch((5, 8), 2, 0.6, boxstyle="round,pad=0.1",
                                   edgecolor='#3b82f6', facecolor='#1e3a8a', linewidth=2)
ax.add_patch(input_box)
ax.text(6, 8.3, 'Input x\n56×56×64', ha='center', va='center', fontsize=11, color='white', fontweight='bold')

# Main branch (F(x))
# Conv1
conv1_box = patches.FancyBboxPatch((4.5, 6.5), 3, 0.8, boxstyle="round,pad=0.1",
                                   edgecolor='#10b981', facecolor='#15803d', linewidth=2)
ax.add_patch(conv1_box)
ax.text(6, 6.9, 'Conv 3×3, 64\nBatchNorm + ReLU', ha='center', va='center', fontsize=10, color='white')

# Conv2
conv2_box = patches.FancyBboxPatch((4.5, 5), 3, 0.8, boxstyle="round,pad=0.1",
                                   edgecolor='#10b981', facecolor='#15803d', linewidth=2)
ax.add_patch(conv2_box)
ax.text(6, 5.4, 'Conv 3×3, 64\nBatchNorm (no ReLU)', ha='center', va='center', fontsize=10, color='white')

# F(x) label
ax.text(3.5, 5.75, 'F(x)', fontsize=14, color='#10b981', fontweight='bold')

# Arrows in main branch
ax.arrow(6, 7.8, 0, -0.8, head_width=0.2, head_length=0.1, fc='white', ec='white', linewidth=2)
ax.arrow(6, 6.5, 0, -0.5, head_width=0.2, head_length=0.1, fc='white', ec='white', linewidth=2)

# Skip connection (curved arrow on the right)
skip_x = np.linspace(7.5, 7.5, 100)
skip_y = np.linspace(8, 3.5, 100)
curve_offset = 1.5 * np.sin(np.linspace(0, np.pi, 100))
ax.plot(skip_x + curve_offset, skip_y, color='#f59e0b', linewidth=3, linestyle='--', alpha=0.9)
ax.arrow(8.8, 3.6, -0.3, 0, head_width=0.2, head_length=0.15, fc='#f59e0b', ec='#f59e0b', linewidth=2)
ax.text(9.5, 5.75, 'Skip\nConnection\n(identity)', ha='center', va='center', fontsize=11, 
        color='#f59e0b', fontweight='bold')

# Addition operation
add_circle = patches.Circle((6, 3.5), 0.4, edgecolor='white', facecolor='#1a1a2e', linewidth=2)
ax.add_patch(add_circle)
ax.text(6, 3.5, '+', ha='center', va='center', fontsize=20, color='white', fontweight='bold')

# Arrow from F(x) to addition
ax.arrow(6, 4.8, 0, -0.9, head_width=0.2, head_length=0.1, fc='white', ec='white', linewidth=2)

# Final ReLU
relu_box = patches.FancyBboxPatch((4.5, 2), 3, 0.7, boxstyle="round,pad=0.1",
                                  edgecolor='#ef4444', facecolor='#b91c1c', linewidth=2)
ax.add_patch(relu_box)
ax.text(6, 2.35, 'ReLU', ha='center', va='center', fontsize=12, color='white', fontweight='bold')

# Arrow from addition to ReLU
ax.arrow(6, 3.1, 0, -0.3, head_width=0.2, head_length=0.1, fc='white', ec='white', linewidth=2)

# Output
output_box = patches.FancyBboxPatch((5, 0.8), 2, 0.6, boxstyle="round,pad=0.1",
                                    edgecolor='#3b82f6', facecolor='#1e3a8a', linewidth=2)
ax.add_patch(output_box)
ax.text(6, 1.1, 'Output\n56×56×64', ha='center', va='center', fontsize=11, color='white', fontweight='bold')

# Arrow from ReLU to output
ax.arrow(6, 2, 0, -0.5, head_width=0.2, head_length=0.1, fc='white', ec='white', linewidth=2)

# Formula annotation
formula_text = r'$\mathcal{H}(x) = \mathrm{ReLU}(F(x) + x)$'
ax.text(6, 0.2, formula_text, ha='center', va='center', fontsize=14, 
        color='#10b981', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e', edgecolor='#10b981', linewidth=2))

# Key insight box
insight_box = patches.FancyBboxPatch((0.3, 7), 3.5, 1.5, boxstyle="round,pad=0.15",
                                     edgecolor='#f59e0b', facecolor='#1a1a2e', linewidth=2, linestyle='--')
ax.add_patch(insight_box)
ax.text(2.05, 8.2, '💡 Key Insight', ha='center', va='top', fontsize=12, color='#f59e0b', fontweight='bold')
ax.text(2.05, 7.5, 'Skip connection provides\ngradient highway:\n∂ℋ/∂x = ∂F/∂x + 1', 
        ha='center', va='center', fontsize=9, color='white', linespacing=1.5)

plt.tight_layout()
plt.savefig('../img/ch01-resnet-architecture.png', dpi=150, facecolor='#1a1a2e', bbox_inches='tight')
print('✅ Generated: ../img/ch01-resnet-architecture.png')
plt.close()
