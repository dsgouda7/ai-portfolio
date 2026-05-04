"""
Generate RoIAlign vs RoI Pooling comparison animation.

Shows the critical difference:
- RoI Pooling: Rounds coordinates (quantization errors)
- RoIAlign: Bilinear interpolation (sub-pixel precision)

Demonstrates why RoIAlign is essential for pixel-level mask prediction.

Output: ch06-roialign-comparison.gif
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

# Dark theme
plt.style.use('dark_background')

def create_roialign_comparison():
    """Create side-by-side comparison of RoI Pooling vs RoIAlign."""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor('#1a1a2e')
    
    methods = ['RoI Pooling (Faster R-CNN)', 'RoIAlign (Mask R-CNN)']
    colors = ['#e74c3c', '#2ecc71']
    
    for ax, method, color in zip(axes, methods, colors):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(method, fontsize=16, fontweight='bold', color=color, pad=20)
        
        # Feature map grid (8×8)
        for i in range(9):
            ax.plot([i, i], [0, 8], 'w-', alpha=0.2, linewidth=0.5)
            ax.plot([0, 8], [i, i], 'w-', alpha=0.2, linewidth=0.5)
        
        ax.text(4, -0.5, 'Feature Map (8×8 grid)', ha='center', fontsize=11, color='#95a5a6')
        
        # Original RoI (exact coordinates with decimals)
        roi_x1, roi_y1 = 2.3, 3.7
        roi_x2, roi_y2 = 6.8, 7.2
        
        # Draw original RoI (dashed, exact coordinates)
        original_roi = patches.Rectangle((roi_x1, roi_y1), roi_x2 - roi_x1, roi_y2 - roi_y1,
                                        linewidth=3, edgecolor='#3498db', facecolor='none',
                                        linestyle='--', label='Original RoI')
        ax.add_patch(original_roi)
        
        ax.text(roi_x1 + 0.1, roi_y1 - 0.3, f'({roi_x1:.1f}, {roi_y1:.1f})', 
                fontsize=9, color='#3498db', fontweight='bold')
        ax.text(roi_x2 - 0.8, roi_y2 + 0.2, f'({roi_x2:.1f}, {roi_y2:.1f})', 
                fontsize=9, color='#3498db', fontweight='bold')
        
        if method == 'RoI Pooling (Faster R-CNN)':
            # Round to integers (quantization)
            quant_x1, quant_y1 = int(np.round(roi_x1)), int(np.round(roi_y1))
            quant_x2, quant_y2 = int(np.round(roi_x2)), int(np.round(roi_y2))
            
            # Draw quantized RoI (solid)
            quant_roi = patches.Rectangle((quant_x1, quant_y1), quant_x2 - quant_x1, quant_y2 - quant_y1,
                                         linewidth=4, edgecolor=color, facecolor=color, alpha=0.3)
            ax.add_patch(quant_roi)
            
            ax.text(quant_x1 + 0.1, quant_y1 - 0.6, f'Rounded: ({quant_x1}, {quant_y1})', 
                    fontsize=9, color=color, fontweight='bold')
            ax.text(quant_x2 - 1.2, quant_y2 + 0.5, f'({quant_x2}, {quant_y2})', 
                    fontsize=9, color=color, fontweight='bold')
            
            # Show error
            error_x = abs(roi_x1 - quant_x1)
            error_y = abs(roi_y1 - quant_y1)
            
            ax.text(4, 8.8, f'❌ Quantization Error:', ha='center', fontsize=12, 
                    color=color, fontweight='bold')
            ax.text(4, 8.3, f'Δx = {error_x:.1f} pixels, Δy = {error_y:.1f} pixels', 
                    ha='center', fontsize=10, color='#95a5a6')
            ax.text(4, 7.8, 'Misaligns mask predictions!', ha='center', fontsize=10, 
                    color='#e74c3c', style='italic')
            
            # Draw sampling grid (7×7) on quantized RoI
            grid_size = 7
            for i in range(grid_size + 1):
                x = quant_x1 + i * (quant_x2 - quant_x1) / grid_size
                ax.plot([x, x], [quant_y1, quant_y2], 'r-', alpha=0.4, linewidth=1.5)
                y = quant_y1 + i * (quant_y2 - quant_y1) / grid_size
                ax.plot([quant_x1, quant_x2], [y, y], 'r-', alpha=0.4, linewidth=1.5)
            
        else:  # RoIAlign
            # Keep exact coordinates, sample with bilinear interpolation
            exact_roi = patches.Rectangle((roi_x1, roi_y1), roi_x2 - roi_x1, roi_y2 - roi_y1,
                                         linewidth=4, edgecolor=color, facecolor=color, alpha=0.3)
            ax.add_patch(exact_roi)
            
            ax.text(4, 8.8, f'✅ Sub-Pixel Precision:', ha='center', fontsize=12, 
                    color=color, fontweight='bold')
            ax.text(4, 8.3, 'Exact coordinates preserved', ha='center', fontsize=10, color='#95a5a6')
            ax.text(4, 7.8, 'Bilinear interpolation at sampling points', ha='center', 
                    fontsize=10, color='#2ecc71', style='italic')
            
            # Draw sampling grid (7×7) with interpolation points
            grid_size = 7
            for i in range(grid_size + 1):
                x = roi_x1 + i * (roi_x2 - roi_x1) / grid_size
                ax.plot([x, x], [roi_y1, roi_y2], 'g-', alpha=0.4, linewidth=1.5)
                y = roi_y1 + i * (roi_y2 - roi_y1) / grid_size
                ax.plot([roi_x1, roi_x2], [y, y], 'g-', alpha=0.4, linewidth=1.5)
            
            # Show some interpolation sample points
            for i in range(1, grid_size):
                for j in range(1, grid_size):
                    x = roi_x1 + i * (roi_x2 - roi_x1) / grid_size
                    y = roi_y1 + j * (roi_y2 - roi_y1) / grid_size
                    if i % 2 == 0 and j % 2 == 0:  # Show subset for clarity
                        ax.plot(x, y, 'o', color='#f39c12', markersize=6, alpha=0.8)
    
    # Bottom summary
    fig.text(0.5, 0.02, 'RoIAlign eliminates quantization errors → accurate pixel-level masks', 
             ha='center', fontsize=13, color='#ecf0f1', fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_animation():
    """Generate animated GIF comparing RoI Pooling and RoIAlign."""
    fig = create_roialign_comparison()
    
    def update(frame):
        # Static comparison (could animate interpolation in future)
        return fig,
    
    anim = FuncAnimation(fig, update, frames=60, interval=100, blit=False)
    
    output_path = '../img/ch06-roialign-comparison.gif'
    anim.save(output_path, writer='pillow', fps=10, dpi=150)
    print(f"✅ Saved RoIAlign comparison animation to {output_path}")
    
    plt.close()

if __name__ == '__main__':
    create_animation()
    print("RoIAlign comparison generation complete!")
