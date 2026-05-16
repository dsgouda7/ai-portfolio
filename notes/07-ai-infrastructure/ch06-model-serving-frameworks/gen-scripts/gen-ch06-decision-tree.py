"""
Generate decision tree diagram for Ch.6 Model Serving Frameworks.

Helps users choose the right framework based on:
- Model type (LLM vs other)
- Throughput requirements
- GPU availability
- Latency requirements
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(16, 12))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#16213e')
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)
ax.axis('off')

# Colors
color_question = '#1d4ed8'    # Blue - Questions
color_decision = '#15803d'    # Green - Final recommendations
color_path = '#888888'        # Gray - Arrows

def draw_box(x, y, w, h, text, color, bold=False):
    """Draw a decision box"""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.15",
                         edgecolor='white', facecolor=color, linewidth=2.5)
    ax.add_patch(box)
    
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center',
            fontsize=11, fontweight=weight, color='white',
            wrap=True)

def draw_arrow(x1, y1, x2, y2, label=''):
    """Draw an arrow with optional label"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=25, 
                           linewidth=2, color='white', alpha=0.7)
    ax.add_patch(arrow)
    
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.2, label, ha='center',
                fontsize=9, color='white',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#0f172a', 
                         edgecolor='white', linewidth=1))

# Title
ax.text(8, 11.5, 'Framework Selection Decision Tree', 
        ha='center', fontsize=22, fontweight='bold', color='white')

# Level 1: Model type
draw_box(8, 10, 3, 0.8, 'What type of model?', color_question, bold=True)

draw_arrow(8, 9.6, 4, 8.5, 'LLM')
draw_arrow(8, 9.6, 12, 8.5, 'Other')

draw_box(4, 8, 2.5, 0.7, 'Decoder-only\n(GPT, Llama)', '#0f172a')
draw_box(12, 8, 2.5, 0.7, 'Vision/Encoder', '#0f172a')

# Right branch (Other models)
draw_arrow(12, 7.65, 12, 6.5)
draw_box(12, 6, 3, 0.8, 'Use ONNX Runtime\nor TensorRT', color_decision, bold=True)
ax.text(12, 5.3, '✅ Better for non-LLM architectures\n✅ Cross-platform support',
        ha='center', fontsize=9, color='white')

# Level 2: Throughput requirement (LLM branch)
draw_arrow(4, 7.65, 4, 6.8)
draw_box(4, 6.3, 3, 0.8, 'Throughput\nrequirement?', color_question, bold=True)

draw_arrow(4, 5.9, 2, 5, '<100 req/s')
draw_arrow(4, 5.9, 6, 5, '>100 req/s')

# Low throughput branch
draw_box(2, 4.5, 2.2, 0.7, 'Small scale', '#0f172a')
draw_arrow(2, 4.15, 2, 3.3)
draw_box(2, 2.8, 3, 0.8, 'Use ONNX Runtime', color_decision, bold=True)
ax.text(2, 2.1, '✅ Memory efficient (INT8)\n✅ Easy setup',
        ha='center', fontsize=9, color='white')

# Level 3: GPU availability (High throughput branch)
draw_box(6, 4.5, 2.2, 0.7, 'Production scale', '#0f172a')
draw_arrow(6, 4.15, 6, 3.8)
draw_box(6, 3.3, 2.8, 0.8, 'GPU availability?', color_question, bold=True)

draw_arrow(6, 2.9, 4.5, 2.2, 'Multi-vendor')
draw_arrow(6, 2.9, 7.5, 2.2, 'NVIDIA only')

# Multi-vendor branch
draw_box(4.5, 1.7, 2.8, 0.7, 'AMD, CPU, ARM', '#0f172a')
draw_arrow(4.5, 1.35, 4.5, 0.8)
draw_box(4.5, 0.5, 3, 0.6, 'Use ONNX Runtime', color_decision, bold=True)

# Level 4: Latency requirement (NVIDIA branch)
draw_box(7.5, 1.7, 2.5, 0.7, 'NVIDIA GPU', '#0f172a')
draw_arrow(7.5, 1.35, 7.5, 0.8)
draw_box(7.5, 0.3, 2.5, 0.6, 'Latency critical?', color_question)

draw_arrow(7.5, -0.05, 9.5, -0.8, 'Yes (<150ms)')
draw_arrow(7.5, -0.05, 5.5, -0.8, 'No (<200ms)')

# Final recommendations
draw_box(5.5, -1.3, 2.8, 0.8, '✅ vLLM', '#15803d', bold=True)
ax.text(5.5, -2, 'Best choice:\n• 10-20× throughput\n• Easy setup\n• PagedAttention',
        ha='center', fontsize=9, color='white',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#0f172a', 
                 edgecolor='#15803d', linewidth=2))

draw_box(9.5, -1.3, 2.8, 0.8, '⚡ TensorRT-LLM', '#b45309', bold=True)
ax.text(9.5, -2, 'Max performance:\n• 20-30× throughput\n• Complex setup\n• Kernel auto-tuning',
        ha='center', fontsize=9, color='white',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#0f172a', 
                 edgecolor='#b45309', linewidth=2))

# Legend
legend_items = [
    ('Question', color_question),
    ('Recommended', color_decision),
    ('Alternative', '#b45309')
]

legend_y = -3.5
for i, (label, color) in enumerate(legend_items):
    x = 5 + i * 2.5
    box = FancyBboxPatch((x - 0.3, legend_y - 0.2), 0.6, 0.4,
                         boxstyle="round,pad=0.05",
                         edgecolor='white', facecolor=color, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + 0.6, legend_y, label, ha='left', va='center',
            fontsize=10, color='white')

plt.tight_layout()
plt.savefig('../img/ch06-decision-tree.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
print("✅ Generated: ../img/ch06-decision-tree.png")
plt.close()
