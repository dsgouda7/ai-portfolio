"""
Generate efficiency comparison chart: ResNet vs MobileNet vs EfficientNet.

Output: ch02-architecture-comparison.png

Shows accuracy vs params and accuracy vs FLOPs for different architectures.
"""

import matplotlib.pyplot as plt
import numpy as np

# Dark theme
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor='#1a1a2e')
ax1.set_facecolor('#1a1a2e')
ax2.set_facecolor('#1a1a2e')

# Architecture data [name, params (M), GFLOPs, ImageNet top-1 accuracy]
architectures = [
    ('ResNet-18', 11.7, 1.8, 69.8, '#ef4444'),
    ('ResNet-34', 21.8, 3.7, 73.3, '#ef4444'),
    ('ResNet-50', 25.6, 4.1, 76.1, '#ef4444'),
    ('ResNet-101', 44.5, 7.8, 77.4, '#ef4444'),
    ('MobileNetV2 α=0.5', 2.0, 0.097, 65.4, '#3b82f6'),
    ('MobileNetV2 α=0.75', 2.6, 0.209, 69.8, '#3b82f6'),
    ('MobileNetV2 α=1.0', 3.5, 0.300, 72.0, '#3b82f6'),
    ('MobileNetV2 α=1.4', 6.1, 0.585, 74.7, '#3b82f6'),
    ('EfficientNet-B0', 5.3, 0.39, 77.3, '#10b981'),
    ('EfficientNet-B1', 7.8, 0.70, 79.2, '#10b981'),
    ('EfficientNet-B2', 9.2, 1.0, 80.3, '#10b981'),
    ('EfficientNet-B3', 12.0, 1.8, 81.7, '#10b981'),
    ('EfficientNet-B5', 30.0, 9.9, 83.7, '#10b981'),
]

names, params, gflops, accuracies, colors = zip(*architectures)

# Plot 1: Accuracy vs Parameters
marker_sizes = [150 if 'ResNet' in name else 180 if 'Mobile' in name else 200 for name in names]
ax1.scatter(params, accuracies, c=colors, s=marker_sizes, alpha=0.8, edgecolors='white', linewidth=2)

# Annotate key models
annotations = [0, 2, 4, 6, 8, 11]  # ResNet-18, ResNet-50, MobileV2 α=0.5, MobileV2 α=1.0, EfficientNet-B0, EfficientNet-B3
for i in annotations:
    label = names[i].replace('MobileNetV2', 'MobileV2')
    offset_x = 2 if i in [0, 4] else 0
    offset_y = 8 if i == 6 else -8 if i in [4, 8] else 0
    ax1.annotate(label, (params[i], accuracies[i]), 
                 textcoords='offset points', xytext=(offset_x, offset_y), ha='center',
                 fontsize=9, color='white', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', edgecolor=colors[i], linewidth=1.5, alpha=0.8))

ax1.set_xlabel('Parameters (millions)', fontsize=13, fontweight='bold')
ax1.set_ylabel('ImageNet Top-1 Accuracy (%)', fontsize=13, fontweight='bold')
ax1.set_title('Model Efficiency: Accuracy vs Size', fontsize=15, fontweight='bold')
ax1.grid(alpha=0.3, linestyle='--')
ax1.set_xlim(-2, 48)
ax1.set_ylim(63, 85)

# Efficiency frontier line (EfficientNet)
efficient_params = [5.3, 7.8, 9.2, 12.0, 30.0]
efficient_accs = [77.3, 79.2, 80.3, 81.7, 83.7]
ax1.plot(efficient_params, efficient_accs, color='#10b981', linestyle='--', linewidth=2, alpha=0.5, label='Pareto frontier')

# Legend
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ef4444', markersize=10, label='ResNet', linestyle=''),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#3b82f6', markersize=10, label='MobileNetV2', linestyle=''),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#10b981', markersize=10, label='EfficientNet', linestyle=''),
]
ax1.legend(handles=legend_elements, loc='lower right', fontsize=11, framealpha=0.9)

# Plot 2: Accuracy vs GFLOPs
ax2.scatter(gflops, accuracies, c=colors, s=marker_sizes, alpha=0.8, edgecolors='white', linewidth=2)

# Annotate key models
for i in annotations:
    label = names[i].replace('MobileNetV2', 'MobileV2')
    offset_x = 0.3 if i in [0, 4] else 0
    offset_y = 8 if i == 6 else -8 if i in [4, 8] else 0
    ax2.annotate(label, (gflops[i], accuracies[i]), 
                 textcoords='offset points', xytext=(offset_x, offset_y), ha='center',
                 fontsize=9, color='white', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', edgecolor=colors[i], linewidth=1.5, alpha=0.8))

ax2.set_xlabel('FLOPs (GFLOPs)', fontsize=13, fontweight='bold')
ax2.set_ylabel('ImageNet Top-1 Accuracy (%)', fontsize=13, fontweight='bold')
ax2.set_title('Model Efficiency: Accuracy vs Compute', fontsize=15, fontweight='bold')
ax2.grid(alpha=0.3, linestyle='--')
ax2.set_xlim(-0.5, 11)
ax2.set_ylim(63, 85)

# Efficiency frontier line (EfficientNet)
efficient_gflops = [0.39, 0.70, 1.0, 1.8, 9.9]
ax2.plot(efficient_gflops, efficient_accs, color='#10b981', linestyle='--', linewidth=2, alpha=0.5, label='Pareto frontier')

# Legend
ax2.legend(handles=legend_elements, loc='lower right', fontsize=11, framealpha=0.9)

# Key insight box
insight_text = ('💡 Key Insight: MobileNetV2 and EfficientNet achieve ResNet-level accuracy\\n'
                'with 5–8× fewer parameters and 8–10× less compute!')
fig.text(0.5, 0.02, insight_text, ha='center', fontsize=12, color='#10b981', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.8', facecolor='#1a1a2e', edgecolor='#10b981', linewidth=2))

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('../img/ch02-architecture-comparison.png', dpi=150, facecolor='#1a1a2e', bbox_inches='tight')
print('✅ Generated: ../img/ch02-architecture-comparison.png')
plt.close()
