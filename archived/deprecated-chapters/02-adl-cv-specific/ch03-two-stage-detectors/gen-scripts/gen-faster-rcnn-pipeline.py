"""
Generate Faster R-CNN pipeline animation.

Shows the two-stage detection process:
1. Backbone CNN extracts features
2. RPN proposes regions
3. RoI pooling aligns features
4. Detection head classifies + refines boxes
5. NMS removes duplicates
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np

# Dark theme
plt.style.use('dark_background')

fig, ax = plt.subplots(figsize=(14, 8), facecolor='#1a1a2e')
ax.set_facecolor('#1a1a2e')
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Animation stages
stages = [
    {
        'title': 'Input Image (1024×768×3)',
        'boxes': [{'x': 10, 'y': 60, 'w': 20, 'h': 30, 'color': '#1d4ed8', 'label': 'Image'}],
        'arrows': []
    },
    {
        'title': 'Backbone CNN (ResNet-50)',
        'boxes': [
            {'x': 10, 'y': 60, 'w': 20, 'h': 30, 'color': '#1d4ed8', 'label': 'Image'},
            {'x': 35, 'y': 60, 'w': 20, 'h': 30, 'color': '#15803d', 'label': 'Feature\nMap'}
        ],
        'arrows': [(30, 75, 35, 75)]
    },
    {
        'title': 'RPN Proposes 300 Regions',
        'boxes': [
            {'x': 35, 'y': 60, 'w': 20, 'h': 30, 'color': '#15803d', 'label': 'Feature\nMap'},
            {'x': 60, 'y': 75, 'w': 15, 'h': 10, 'color': '#b45309', 'label': 'RPN'},
            {'x': 60, 'y': 55, 'w': 15, 'h': 10, 'color': '#b45309', 'label': 'Proposals\n×300'}
        ],
        'arrows': [(55, 75, 60, 75), (55, 65, 60, 60)]
    },
    {
        'title': 'RoI Pooling (7×7 per region)',
        'boxes': [
            {'x': 60, 'y': 55, 'w': 15, 'h': 10, 'color': '#b45309', 'label': 'Proposals\n×300'},
            {'x': 80, 'y': 60, 'w': 15, 'h': 20, 'color': '#7c3aed', 'label': 'RoI\nPooled'}
        ],
        'arrows': [(75, 60, 80, 70)]
    },
    {
        'title': 'Detection Head (Classify + Refine)',
        'boxes': [
            {'x': 80, 'y': 60, 'w': 15, 'h': 20, 'color': '#7c3aed', 'label': 'RoI\nPooled'},
            {'x': 10, 'y': 20, 'w': 12, 'h': 15, 'color': '#00d9ff', 'label': 'Coca\nCola\n95%'},
            {'x': 25, 'y': 20, 'w': 12, 'h': 15, 'color': '#00d9ff', 'label': 'Pepsi\n92%'},
            {'x': 40, 'y': 20, 'w': 12, 'h': 15, 'color': '#00d9ff', 'label': 'Sprite\n88%'}
        ],
        'arrows': [(90, 70, 45, 35)]
    }
]

def update(frame):
    ax.clear()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    stage = stages[frame]
    
    # Title
    ax.text(50, 95, stage['title'], ha='center', va='top', 
           fontsize=16, weight='bold', color='white')
    
    # Draw boxes
    for box_data in stage['boxes']:
        rect = patches.FancyBboxPatch(
            (box_data['x'], box_data['y']), 
            box_data['w'], box_data['h'],
            boxstyle="round,pad=0.5", 
            edgecolor=box_data['color'],
            facecolor=box_data['color'],
            alpha=0.3,
            linewidth=2
        )
        ax.add_patch(rect)
        
        # Label
        ax.text(box_data['x'] + box_data['w']/2, 
               box_data['y'] + box_data['h']/2,
               box_data['label'], 
               ha='center', va='center',
               fontsize=10, weight='bold', color='white')
    
    # Draw arrows
    for arrow in stage['arrows']:
        x1, y1, x2, y2 = arrow
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='#00d9ff'))
    
    # Stage counter
    ax.text(50, 5, f'Stage {frame + 1}/5', ha='center', va='bottom',
           fontsize=12, color='#666', style='italic')

# Create animation
anim = FuncAnimation(fig, update, frames=len(stages), interval=2000, repeat=True)

# Save
anim.save('../img/ch03-faster-rcnn-pipeline.gif', writer='pillow', fps=0.5, dpi=150)
print("✓ Saved: ch03-faster-rcnn-pipeline.gif")

plt.close()
