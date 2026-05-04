"""
Generate Non-Maximum Suppression (NMS) process visualization.

Shows how NMS removes duplicate detections by suppressing overlapping boxes
with lower confidence scores.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np

# Dark theme
plt.style.use('dark_background')

fig, ax = plt.subplots(figsize=(12, 8), facecolor='#1a1a2e')
ax.set_facecolor('#1a1a2e')

# Define overlapping detection boxes around a single object (soda can)
# Format: [x, y, width, height, confidence, label]
detections = [
    {'box': [150, 200, 80, 150], 'conf': 0.95, 'label': 'Coca-Cola', 'color': '#15803d'},
    {'box': [155, 205, 78, 145], 'conf': 0.88, 'label': 'Coca-Cola', 'color': '#b45309'},
    {'box': [145, 198, 85, 152], 'conf': 0.82, 'label': 'Coca-Cola', 'color': '#b45309'},
    {'box': [160, 210, 75, 140], 'conf': 0.76, 'label': 'Coca-Cola', 'color': '#b45309'},
    {'box': [148, 203, 82, 148], 'conf': 0.71, 'label': 'Coca-Cola', 'color': '#b45309'},
]

# NMS animation stages
stages = [
    {'title': 'Before NMS: 5 Overlapping Detections', 'keep_idx': list(range(5)), 'suppress_idx': []},
    {'title': 'NMS Step 1: Keep Highest Confidence (0.95)', 'keep_idx': [0], 'suppress_idx': []},
    {'title': 'NMS Step 2: Suppress IoU > 0.5 (boxes 2-5)', 'keep_idx': [0], 'suppress_idx': [1, 2, 3, 4]},
    {'title': 'After NMS: 1 Final Detection', 'keep_idx': [0], 'suppress_idx': [1, 2, 3, 4], 'final': True}
]

def compute_iou(box1, box2):
    """Compute Intersection over Union."""
    x1_1, y1_1, w1, h1 = box1
    x1_2, y1_2, w2, h2 = box2
    x2_1, y2_1 = x1_1 + w1, y1_1 + h1
    x2_2, y2_2 = x1_2 + w2, y1_2 + h2
    
    # Intersection
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def update(frame):
    ax.clear()
    ax.set_xlim(100, 350)
    ax.set_ylim(150, 400)
    ax.invert_yaxis()  # Match image coordinates (y increases downward)
    ax.set_aspect('equal')
    ax.axis('off')
    
    stage = stages[frame]
    
    # Title
    ax.text(225, 160, stage['title'], ha='center', va='top',
           fontsize=14, weight='bold', color='white')
    
    # Draw suppressed boxes (semi-transparent red)
    for idx in stage['suppress_idx']:
        det = detections[idx]
        x, y, w, h = det['box']
        rect = patches.Rectangle((x, y), w, h,
                                linewidth=2, edgecolor='#b91c1c',
                                facecolor='#b91c1c', alpha=0.2,
                                linestyle='--')
        ax.add_patch(rect)
        
        # Label
        ax.text(x + w/2, y + h + 10, f"Suppressed\n(IoU > 0.5)",
               ha='center', va='top', fontsize=9, color='#b91c1c',
               style='italic')
    
    # Draw kept boxes
    for idx in stage['keep_idx']:
        det = detections[idx]
        x, y, w, h = det['box']
        
        # Highlight final detection
        if stage.get('final'):
            linewidth = 4
            edgecolor = '#00d9ff'
        else:
            linewidth = 2
            edgecolor = det['color']
        
        rect = patches.Rectangle((x, y), w, h,
                                linewidth=linewidth, edgecolor=edgecolor,
                                facecolor='none')
        ax.add_patch(rect)
        
        # Confidence label
        ax.text(x + w/2, y - 5, f"{det['label']}: {det['conf']:.2f}",
               ha='center', va='bottom', fontsize=10, weight='bold',
               color='white',
               bbox=dict(boxstyle='round', facecolor='black', 
                        alpha=0.7, edgecolor=edgecolor, linewidth=2))
    
    # Show IoU calculation for stage 2
    if frame == 2:
        # Compute IoU between box 0 and box 1
        iou_01 = compute_iou(detections[0]['box'], detections[1]['box'])
        ax.text(300, 250, f"IoU calculation:\nBox 1 vs Box 2: {iou_01:.2f}\n(> 0.5 threshold → suppress)",
               ha='left', va='top', fontsize=10, color='white',
               bbox=dict(boxstyle='round', facecolor='black', 
                        alpha=0.8, edgecolor='#b91c1c', linewidth=2))
    
    # Stage indicator
    ax.text(225, 385, f'Stage {frame + 1}/4', ha='center', va='bottom',
           fontsize=11, color='#666', style='italic')

# Create animation
anim = FuncAnimation(fig, update, frames=len(stages), interval=2000, repeat=True)

# Save
anim.save('../img/ch03-nms-process.gif', writer='pillow', fps=0.5, dpi=150)
print("✓ Saved: ch03-nms-process.gif")

plt.close()
