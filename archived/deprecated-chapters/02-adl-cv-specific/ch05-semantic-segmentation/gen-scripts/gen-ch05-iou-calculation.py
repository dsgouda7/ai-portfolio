"""
Generate IoU (Intersection over Union) calculation visualization.

Shows step-by-step IoU computation for semantic segmentation:
1. Ground truth mask
2. Predicted mask  
3. Intersection (overlap)
4. Union (combined area)
5. IoU = Intersection / Union

Output: ch05-iou-calculation.gif
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

# Dark theme
plt.style.use('dark_background')

def create_iou_visualization():
    """Create animated IoU calculation demonstration."""
    
    # Create sample masks (64x64 for visibility)
    size = 64
    
    # Ground truth: square product region
    gt_mask = np.zeros((size, size))
    gt_mask[20:50, 25:55] = 1  # Product class
    
    # Prediction: slightly offset square
    pred_mask = np.zeros((size, size))
    pred_mask[22:52, 23:53] = 1  # Predicted product
    
    # Calculate intersection and union
    intersection = np.logical_and(gt_mask, pred_mask).astype(int)
    union = np.logical_or(gt_mask, pred_mask).astype(int)
    
    iou_value = intersection.sum() / union.sum()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.patch.set_facecolor('#1a1a2e')
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    titles = [
        'Ground Truth Mask',
        'Predicted Mask',
        'Intersection (AND)',
        'Union (OR)',
        'IoU Calculation',
        'IoU Visualization'
    ]
    
    cmaps = ['Greens', 'Blues', 'Reds', 'Purples', 'gray', 'gray']
    
    for ax, title, cmap in zip(axes, titles, cmaps):
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.axis('off')
    
    # Ground truth
    axes[0].imshow(gt_mask, cmap='Greens', vmin=0, vmax=1, interpolation='nearest')
    axes[0].text(32, -3, 'True Positives + False Negatives', ha='center', fontsize=10, color='#2ecc71')
    
    # Prediction
    axes[1].imshow(pred_mask, cmap='Blues', vmin=0, vmax=1, interpolation='nearest')
    axes[1].text(32, -3, 'True Positives + False Positives', ha='center', fontsize=10, color='#3498db')
    
    # Intersection
    axes[2].imshow(intersection, cmap='Reds', vmin=0, vmax=1, interpolation='nearest')
    axes[2].text(32, -3, f'TP = {intersection.sum()} pixels', ha='center', fontsize=11, color='#e74c3c', fontweight='bold')
    
    # Union
    axes[3].imshow(union, cmap='Purples', vmin=0, vmax=1, interpolation='nearest')
    axes[3].text(32, -3, f'TP + FP + FN = {union.sum()} pixels', ha='center', fontsize=11, color='#9b59b6', fontweight='bold')
    
    # IoU calculation (text-based)
    axes[4].text(0.5, 0.7, 'IoU Formula:', ha='center', va='center', fontsize=16, 
                fontweight='bold', transform=axes[4].transAxes, color='#ecf0f1')
    axes[4].text(0.5, 0.5, 'IoU = |Intersection| / |Union|', ha='center', va='center', 
                fontsize=14, transform=axes[4].transAxes, color='#95a5a6')
    axes[4].text(0.5, 0.3, f'IoU = {intersection.sum()} / {union.sum()}', ha='center', va='center', 
                fontsize=14, transform=axes[4].transAxes, color='#3498db')
    axes[4].text(0.5, 0.1, f'IoU = {iou_value:.4f}', ha='center', va='center', 
                fontsize=18, transform=axes[4].transAxes, color='#2ecc71', fontweight='bold')
    
    # IoU visualization (combined overlay)
    overlay = np.zeros((size, size, 3))
    overlay[gt_mask == 1] = [0, 1, 0]  # Green for GT
    overlay[pred_mask == 1] = [0, 0, 1]  # Blue for pred
    overlay[intersection == 1] = [1, 0, 0]  # Red for overlap (TP)
    
    axes[5].imshow(overlay)
    axes[5].text(32, -3, f'Red = Overlap (TP), Green = FN, Blue = FP', ha='center', fontsize=9, color='#ecf0f1')
    
    # Add quality indicator
    quality_text = "Good" if iou_value >= 0.7 else "Fair" if iou_value >= 0.5 else "Poor"
    quality_color = "#2ecc71" if iou_value >= 0.7 else "#f39c12" if iou_value >= 0.5 else "#e74c3c"
    axes[5].text(32, 70, f'Quality: {quality_text} ({iou_value:.2f})', ha='center', 
                fontsize=12, color=quality_color, fontweight='bold')
    
    plt.tight_layout()
    return fig, iou_value

def create_animation():
    """Generate animated GIF showing IoU calculation steps."""
    fig, iou_val = create_iou_visualization()
    
    def update(frame):
        # Static for this version (could animate mask overlay in future)
        return fig,
    
    anim = FuncAnimation(fig, update, frames=60, interval=100, blit=False)
    
    output_path = '../img/ch05-iou-calculation.gif'
    anim.save(output_path, writer='pillow', fps=10, dpi=150)
    print(f"✅ Saved IoU calculation animation to {output_path}")
    print(f"   Sample IoU: {iou_val:.4f}")
    
    plt.close()

if __name__ == '__main__':
    create_animation()
    print("IoU visualization generation complete!")
