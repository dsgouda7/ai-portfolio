"""
Generate visualization of instance segmentation output showing separate masks per object.

Demonstrates how Mask R-CNN outputs individual instance masks enabling counting,
unlike semantic segmentation which merges all instances of a class.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Set dark theme
plt.style.use('dark_background')

def generate_instance_counting():
    """Visualize instance segmentation output with separate object masks."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Instance Segmentation: Separate Masks Enable Object Counting', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Color scheme for different instances
    colors_instances = ['#ff6b6b', '#4a9eff', '#51cf66', '#ffd43b', '#9d4edd']
    
    # ===== Subplot 1: Original Image (Simulated) =====
    ax = axes[0, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.set_title('Input: Retail Shelf Image', fontsize=12, fontweight='bold', color='white')
    ax.axis('off')
    
    # Draw simulated shelf with products
    ax.add_patch(mpatches.Rectangle((0, 0), 10, 10, facecolor='#1a1a1a', edgecolor='#444444', linewidth=2))
    
    # Shelf edge
    ax.plot([0, 10], [2, 2], color='#666666', linewidth=3)
    ax.text(5, 1, 'Shelf Edge', ha='center', fontsize=9, color='#888888', style='italic')
    
    # Products (boxes/bottles)
    products = [
        {'x': 1, 'y': 3, 'w': 1.5, 'h': 3, 'label': 'Box 1', 'color': '#8b4513'},
        {'x': 3, 'y': 3.5, 'w': 1.2, 'h': 2.5, 'label': 'Box 2', 'color': '#a0522d'},
        {'x': 5, 'y': 2.8, 'w': 1, 'h': 4, 'label': 'Bottle 1', 'color': '#4682b4'},
        {'x': 6.5, 'y': 3, 'w': 1.3, 'h': 2.8, 'label': 'Box 3', 'color': '#cd853f'},
        {'x': 8.2, 'y': 3.2, 'w': 1, 'h': 3.5, 'label': 'Bottle 2', 'color': '#5f9ea0'},
    ]
    
    for prod in products:
        ax.add_patch(mpatches.FancyBboxPatch((prod['x'], prod['y']), prod['w'], prod['h'],
                                              boxstyle="round,pad=0.05", facecolor=prod['color'],
                                              edgecolor='white', linewidth=2))
        ax.text(prod['x'] + prod['w']/2, prod['y'] + prod['h']/2, prod['label'],
                ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    ax.text(5, 8.5, 'Question: How many products on shelf?', ha='center', fontsize=10,
            fontweight='bold', color='#ffd43b', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#2c2c2c', edgecolor='#ffd43b', linewidth=2))
    
    # ===== Subplot 2: Semantic Segmentation (Can't Count) =====
    ax = axes[0, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.set_title('Semantic Segmentation\n(Cannot Count Instances)', fontsize=12, fontweight='bold', color='#ff6b6b')
    ax.axis('off')
    
    # Background
    ax.add_patch(mpatches.Rectangle((0, 0), 10, 10, facecolor='#1a1a1a', edgecolor='#444444', linewidth=2))
    
    # All products as one blob (semantic segmentation output)
    all_product_mask = mpatches.Polygon(
        [(1, 3), (1.5, 6), (3, 6), (3.5, 5.8), (4.2, 6), 
         (5, 6.8), (6, 6.8), (6.5, 5.8), (7.8, 5.8), (8.2, 6.7), 
         (9.2, 6.7), (9.2, 3), (8.2, 3.2), (6.5, 3), (5, 2.8), (3, 3.5), (1.5, 3)],
        facecolor='#4a9eff', edgecolor='white', linewidth=2, alpha=0.7
    )
    ax.add_patch(all_product_mask)
    
    ax.text(5, 5, 'All Pixels:\nClass = "Product"', ha='center', va='center', fontsize=11,
            fontweight='bold', color='white', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#2c2c2c', edgecolor='white', linewidth=2))
    
    ax.text(5, 1.5, '❌ Cannot distinguish instances\n❌ Cannot count: 1 blob or 5 products?',
            ha='center', fontsize=9, fontweight='bold', color='#ff6b6b')
    
    # ===== Subplot 3: Instance Segmentation (Individual Masks) =====
    ax = axes[1, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.set_title('Mask R-CNN: Instance Segmentation\n(Separate Masks)', fontsize=12, fontweight='bold', color='#51cf66')
    ax.axis('off')
    
    # Background
    ax.add_patch(mpatches.Rectangle((0, 0), 10, 10, facecolor='#1a1a1a', edgecolor='#444444', linewidth=2))
    
    # Individual instance masks with different colors
    for i, prod in enumerate(products):
        ax.add_patch(mpatches.FancyBboxPatch((prod['x'], prod['y']), prod['w'], prod['h'],
                                              boxstyle="round,pad=0.05", facecolor=colors_instances[i],
                                              edgecolor='white', linewidth=2.5, alpha=0.8))
        ax.text(prod['x'] + prod['w']/2, prod['y'] + prod['h']/2, f'Instance {i+1}',
                ha='center', va='center', fontsize=8, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#000000', alpha=0.7))
    
    ax.text(5, 1, '✅ Each product has separate mask\n✅ Count = 5 products',
            ha='center', fontsize=9, fontweight='bold', color='#51cf66')
    
    # ===== Subplot 4: Output Data Structure =====
    ax = axes[1, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Mask R-CNN Output\n(Per-Instance Data)', fontsize=12, fontweight='bold', color='#ffd43b')
    
    # Show output structure as code-like boxes
    output_data = [
        {'id': 1, 'class': 'cereal_box', 'score': 0.95, 'bbox': '[1.0, 3.0, 2.5, 6.0]', 'color': colors_instances[0]},
        {'id': 2, 'class': 'cereal_box', 'score': 0.92, 'bbox': '[3.0, 3.5, 4.2, 6.0]', 'color': colors_instances[1]},
        {'id': 3, 'class': 'milk_bottle', 'score': 0.89, 'bbox': '[5.0, 2.8, 6.0, 6.8]', 'color': colors_instances[2]},
        {'id': 4, 'class': 'cereal_box', 'score': 0.88, 'bbox': '[6.5, 3.0, 7.8, 5.8]', 'color': colors_instances[3]},
        {'id': 5, 'class': 'milk_bottle', 'score': 0.91, 'bbox': '[8.2, 3.2, 9.2, 6.7]', 'color': colors_instances[4]},
    ]
    
    y_offset = 8.5
    for i, inst in enumerate(output_data):
        # Box for each instance
        box_y = y_offset - i * 1.5
        ax.add_patch(mpatches.FancyBboxPatch((0.5, box_y), 9, 1.2, boxstyle="round,pad=0.1",
                                              facecolor='#2c2c2c', edgecolor=inst['color'], linewidth=2.5))
        
        # Instance info text
        info_text = f"Instance {inst['id']}: class='{inst['class']}', score={inst['score']:.2f}\nbbox={inst['bbox']}, mask=28×28"
        ax.text(1, box_y + 0.6, info_text, ha='left', va='center', fontsize=7,
                fontweight='bold', color='white', family='monospace')
        
        # Color indicator
        ax.add_patch(mpatches.Circle((0.8, box_y + 0.6), 0.15, facecolor=inst['color'], 
                                     edgecolor='white', linewidth=1.5))
    
    # Count summary
    ax.text(5, 0.5, f'Total Detections: {len(output_data)} instances\nCereal Boxes: 3 | Milk Bottles: 2',
            ha='center', fontsize=10, fontweight='bold', color='#ffd43b',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#2c2c2c', edgecolor='#ffd43b', linewidth=2))
    
    # Overall legend
    fig.text(0.5, 0.01, 'Instance Segmentation = Detection (bbox + class) + Segmentation (per-object mask) → Enables counting & tracking',
             ha='center', fontsize=11, style='italic', color='#aaaaaa')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    output_path = Path(__file__).parent.parent / 'img' / 'ch06-instance-counting.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1e1e1e')
    print(f"✅ Generated: {output_path}")
    plt.close()

if __name__ == '__main__':
    generate_instance_counting()
