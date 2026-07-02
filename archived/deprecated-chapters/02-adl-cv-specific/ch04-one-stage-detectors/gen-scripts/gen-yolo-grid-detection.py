"""
Generate YOLO grid-based detection animation.

Shows how YOLO divides the image into a grid and predicts boxes + classes
at each grid cell in a single forward pass.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np

# Dark theme
plt.style.use('dark_background')

fig, ax = plt.subplots(figsize=(12, 10), facecolor='#1a1a2e')
ax.set_facecolor('#1a1a2e')

# Create synthetic shelf image
img_size = (640, 640)
grid_size = 7

# Define products (colored boxes)
products = [
    {'pos': (150, 200), 'size': (80, 150), 'color': '#dc2626', 'label': 'Coca-Cola'},
    {'pos': (350, 180), 'size': (75, 140), 'color': '#2563eb', 'label': 'Pepsi'},
    {'pos': (520, 210), 'size': (70, 145), 'color': '#16a34a', 'label': 'Sprite'},
]

# Animation stages
stages = [
    {'title': 'Step 1: Input Image (640×640)', 'show_grid': False, 'show_predictions': False},
    {'title': 'Step 2: Divide into 7×7 Grid', 'show_grid': True, 'show_predictions': False},
    {'title': 'Step 3: Each Cell Predicts Boxes + Classes', 'show_grid': True, 'show_predictions': 'cells'},
    {'title': 'Step 4: Filter by Confidence (>0.25)', 'show_grid': False, 'show_predictions': 'filtered'},
    {'title': 'Step 5: Apply NMS → Final Detections', 'show_grid': False, 'show_predictions': 'final'}
]

def update(frame):
    ax.clear()
    ax.set_xlim(0, img_size[0])
    ax.set_ylim(0, img_size[1])
    ax.invert_yaxis()  # Match image coordinates
    ax.set_aspect('equal')
    ax.axis('off')
    
    stage = stages[frame]
    
    # Title
    ax.text(img_size[0]/2, -30, stage['title'], ha='center', va='top',
           fontsize=16, weight='bold', color='white')
    
    # Draw background
    background = patches.Rectangle((0, 0), img_size[0], img_size[1],
                                  facecolor='#1a1a2e', edgecolor='#444', linewidth=2)
    ax.add_patch(background)
    
    # Draw products
    for prod in products:
        x, y = prod['pos']
        w, h = prod['size']
        rect = patches.Rectangle((x, y), w, h,
                                facecolor=prod['color'], 
                                edgecolor='white', linewidth=2, alpha=0.7)
        ax.add_patch(rect)
        
        # Label (only in final stages)
        if stage['show_predictions'] in ['filtered', 'final']:
            ax.text(x + w/2, y - 10, prod['label'],
                   ha='center', va='bottom', fontsize=11, weight='bold',
                   color='white',
                   bbox=dict(boxstyle='round', facecolor='black', 
                            alpha=0.8, edgecolor=prod['color'], linewidth=2))
    
    # Draw grid
    if stage['show_grid']:
        cell_size = img_size[0] / grid_size
        for i in range(grid_size + 1):
            # Vertical lines
            ax.plot([i * cell_size, i * cell_size], [0, img_size[1]],
                   color='#00d9ff', alpha=0.4, linewidth=1)
            # Horizontal lines
            ax.plot([0, img_size[0]], [i * cell_size, i * cell_size],
                   color='#00d9ff', alpha=0.4, linewidth=1)
        
        # Highlight responsible cells
        if stage['show_predictions'] == 'cells':
            for prod in products:
                # Find grid cell containing product center
                cx = prod['pos'][0] + prod['size'][0] / 2
                cy = prod['pos'][1] + prod['size'][1] / 2
                gx = int(cx // cell_size)
                gy = int(cy // cell_size)
                
                # Highlight cell
                highlight = patches.Rectangle(
                    (gx * cell_size, gy * cell_size),
                    cell_size, cell_size,
                    facecolor='lime', alpha=0.2,
                    edgecolor='lime', linewidth=3
                )
                ax.add_patch(highlight)
                
                # Label cell
                ax.text(gx * cell_size + cell_size/2, 
                       gy * cell_size + cell_size/2,
                       f'Cell ({gx},{gy})\npredicts box',
                       ha='center', va='center', fontsize=9,
                       color='lime', weight='bold',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # Draw prediction boxes (slightly offset to show multiple predictions)
    if stage['show_predictions'] == 'cells':
        for prod in products:
            # Show 2 predictions per cell (before filtering)
            for offset in [-5, 5]:
                x = prod['pos'][0] + offset
                y = prod['pos'][1] + offset
                w, h = prod['size']
                pred_box = patches.Rectangle((x, y), w, h,
                                            facecolor='none',
                                            edgecolor='#b45309',
                                            linewidth=2, linestyle='--', alpha=0.6)
                ax.add_patch(pred_box)
    
    # Draw final detection boxes
    if stage['show_predictions'] in ['filtered', 'final']:
        for prod in products:
            x, y = prod['pos']
            w, h = prod['size']
            final_box = patches.Rectangle((x, y), w, h,
                                         facecolor='none',
                                         edgecolor='#00d9ff',
                                         linewidth=3)
            ax.add_patch(final_box)
            
            # Confidence score
            if stage['show_predictions'] == 'final':
                conf = np.random.uniform(0.85, 0.95)
                ax.text(x + w + 5, y + h/2, f'{conf:.2f}',
                       ha='left', va='center', fontsize=11,
                       color='#00d9ff', weight='bold')
    
    # Info box
    if frame == 0:
        info_text = "YOLO divides image into grid\nEach cell predicts boxes + classes\nSingle forward pass → fast!"
    elif frame == 1:
        info_text = f"Grid: {grid_size}×{grid_size} = {grid_size**2} cells\nEach cell: 2 boxes × 25 values\nTotal predictions: {grid_size**2 * 2}"
    elif frame == 2:
        info_text = "Each cell outputs:\n• Objectness (1 value)\n• Box coords (4 values)\n• Class probs (20 values)"
    elif frame == 3:
        info_text = f"Initial predictions: ~{grid_size**2 * 2}\nAfter confidence filter: ~{len(products) * 3}\nRemove low-confidence boxes"
    else:
        info_text = f"Final detections: {len(products)}\nInference time: ~18ms\n10× faster than two-stage!"
    
    ax.text(10, img_size[1] - 10, info_text,
           ha='left', va='bottom', fontsize=11, color='white',
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.8,
                    edgecolor='#00d9ff', linewidth=2))
    
    # Stage indicator
    ax.text(img_size[0]/2, img_size[1] + 30, f'Stage {frame + 1}/5',
           ha='center', va='bottom', fontsize=12, color='#666', style='italic')

# Create animation
anim = FuncAnimation(fig, update, frames=len(stages), interval=2500, repeat=True)

# Save
anim.save('../img/ch04-yolo-grid-detection.gif', writer='pillow', fps=0.4, dpi=150)
print("✓ Saved: ch04-yolo-grid-detection.gif")

plt.close()
