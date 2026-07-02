"""
Generate instance segmentation results visualization.

Shows the difference between:
1. Semantic segmentation (all products labeled "product")
2. Instance segmentation (each product gets unique mask + ID)

Demonstrates why instance segmentation is needed for inventory counting.

Output: ch06-semantic-vs-instance.gif
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

# Dark theme
plt.style.use('dark_background')

def create_semantic_vs_instance_comparison():
    """Create comparison showing semantic vs instance segmentation."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('#1a1a2e')
    
    # Create synthetic shelf with overlapping products
    img_size = 256
    image = np.ones((img_size, img_size, 3)) * 0.9
    
    # Add 3 overlapping products
    products = [
        (50, 80, 120, 200, [0.9, 0.4, 0.2]),   # Cereal box 1
        (70, 100, 140, 220, [0.9, 0.5, 0.3]),  # Cereal box 2
        (140, 90, 210, 210, [0.8, 0.9, 1.0]),  # Milk carton
    ]
    
    for x1, y1, x2, y2, color in products:
        image[y1:y2, x1:x2] = color
    
    # 1. Original image
    axes[0].imshow(image)
    axes[0].set_title('Input Image (Overlapping Products)', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Draw bounding boxes on original
    for i, (x1, y1, x2, y2, _) in enumerate(products):
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=3,
                                edgecolor='lime', facecolor='none')
        axes[0].add_patch(rect)
        axes[0].text(x1+5, y1+15, f'Product {i+1}', fontsize=10, 
                    color='lime', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    # 2. Semantic segmentation (all labeled "product")
    semantic_mask = np.zeros((img_size, img_size))
    for x1, y1, x2, y2, _ in products:
        semantic_mask[y1:y2, x1:x2] = 1  # All same label
    
    # Overlay semantic mask
    axes[1].imshow(image)
    semantic_overlay = np.zeros((img_size, img_size, 4))
    semantic_overlay[:, :, 1] = semantic_mask  # Green channel
    semantic_overlay[:, :, 3] = semantic_mask * 0.5  # Alpha
    axes[1].imshow(semantic_overlay)
    
    axes[1].set_title('❌ Semantic Segmentation\n(All pixels = \"product\")', 
                     fontsize=14, fontweight='bold', color='#f39c12')
    axes[1].axis('off')
    
    axes[1].text(img_size/2, img_size + 20, 
                '⚠️  Cannot count instances!\nAll products merged into one blob',
                ha='center', fontsize=11, color='#e74c3c', style='italic',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e', 
                         edgecolor='#e74c3c', linewidth=2))
    
    # 3. Instance segmentation (unique mask per product)
    axes[2].imshow(image)
    
    # Different color per instance
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    
    for i, (x1, y1, x2, y2, _) in enumerate(products):
        # Create instance mask overlay
        instance_overlay = np.zeros((img_size, img_size, 4))
        instance_overlay[y1:y2, x1:x2, :3] = colors[i]
        instance_overlay[y1:y2, x1:x2, 3] = 0.4
        axes[2].imshow(instance_overlay)
        
        # Draw bounding box
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=3,
                                edgecolor='white', facecolor='none', linestyle='--')
        axes[2].add_patch(rect)
        
        # Instance label
        axes[2].text(x1 + (x2-x1)/2, y1 + (y2-y1)/2, f'ID={i+1}', 
                    ha='center', va='center', fontsize=14, 
                    color='white', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8))
    
    axes[2].set_title('✅ Instance Segmentation\n(Each product = unique mask)', 
                     fontsize=14, fontweight='bold', color='#2ecc71')
    axes[2].axis('off')
    
    axes[2].text(img_size/2, img_size + 20, 
                f'✅  Can count: {len(products)} products detected\nSKU-level inventory tracking enabled',
                ha='center', fontsize=11, color='#2ecc71', style='italic',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e', 
                         edgecolor='#2ecc71', linewidth=2))
    
    # Overall title
    fig.suptitle('Semantic vs Instance Segmentation — Why Mask R-CNN Matters', 
                fontsize=16, fontweight='bold', color='#ecf0f1', y=0.98)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    return fig

def create_animation():
    """Generate animated GIF showing semantic vs instance comparison."""
    fig = create_semantic_vs_instance_comparison()
    
    def update(frame):
        # Static comparison
        return fig,
    
    anim = FuncAnimation(fig, update, frames=60, interval=100, blit=False)
    
    output_path = '../img/ch06-semantic-vs-instance.gif'
    anim.save(output_path, writer='pillow', fps=10, dpi=150)
    print(f"✅ Saved semantic vs instance comparison to {output_path}")
    
    plt.close()

if __name__ == '__main__':
    create_animation()
    print("Semantic vs instance comparison generation complete!")
