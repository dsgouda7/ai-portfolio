"""
Generate visualization of Mask R-CNN mask prediction process.

Shows the mask head architecture: RoIAlign features → Conv layers → 
Transposed conv → Per-class masks → Binary mask output.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Set dark theme
plt.style.use('dark_background')

def generate_mask_prediction():
    """Visualize Mask R-CNN mask prediction head workflow."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle('Mask R-CNN: Mask Prediction Head (RoI → Binary Mask)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Color scheme
    color_roialign = '#4a9eff'
    color_conv = '#ff6b6b'
    color_deconv = '#51cf66'
    color_output = '#ffd43b'
    
    # ===== Step 1: RoIAlign Feature Extraction =====
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Step 1: RoIAlign\nFeature Extraction', fontsize=11, fontweight='bold', color=color_roialign)
    
    # Draw detected object box
    ax.add_patch(mpatches.Rectangle((1, 2), 8, 6, facecolor='none', edgecolor='white', linewidth=3, linestyle='--'))
    ax.text(5, 8.5, 'Detected Object\n(bbox from detection head)', ha='center', fontsize=9, color='white')
    
    # Draw feature map extraction
    ax.add_patch(mpatches.FancyBboxPatch((3, 3.5), 4, 3, boxstyle="round,pad=0.2",
                                          facecolor=color_roialign, edgecolor='white', linewidth=2.5, alpha=0.8))
    ax.text(5, 5, 'RoIAlign\n14×14×256\nfeature tensor', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')
    
    ax.text(5, 1, 'Bilinear interpolation\n(sub-pixel precision)', ha='center', fontsize=8, 
            style='italic', color='#aaaaaa')
    
    # ===== Step 2: Conv Layers (Feature Refinement) =====
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Step 2: Conv Layers\nFeature Refinement', fontsize=11, fontweight='bold', color=color_conv)
    
    conv_layers = [
        {'y': 6.5, 'label': 'Conv1 3×3\n256 filters'},
        {'y': 4.8, 'label': 'Conv2 3×3\n256 filters'},
        {'y': 3.1, 'label': 'Conv3 3×3\n256 filters'},
        {'y': 1.4, 'label': 'Conv4 3×3\n256 filters'},
    ]
    
    for i, layer in enumerate(conv_layers):
        ax.add_patch(mpatches.FancyBboxPatch((2, layer['y']), 6, 1.2, boxstyle="round,pad=0.1",
                                              facecolor=color_conv, edgecolor='white', linewidth=2, alpha=0.8 - i*0.1))
        ax.text(5, layer['y'] + 0.6, layer['label'], ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')
        if i < len(conv_layers) - 1:
            ax.arrow(5, layer['y'], 0, -0.5, head_width=0.4, head_length=0.15, fc='white', ec='white', linewidth=2)
    
    ax.text(5, 0.3, 'Output: 14×14×256', ha='center', fontsize=9, fontweight='bold', color='white')
    
    # ===== Step 3: Transposed Conv (Upsampling) =====
    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Step 3: Deconvolution\n2× Upsampling', fontsize=11, fontweight='bold', color=color_deconv)
    
    # Small input
    ax.add_patch(mpatches.FancyBboxPatch((3.5, 6), 3, 2, boxstyle="round,pad=0.1",
                                          facecolor='#555555', edgecolor='white', linewidth=2))
    ax.text(5, 7, '14×14×256', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # Upsampling arrow
    ax.arrow(5, 5.8, 0, -1.5, head_width=0.5, head_length=0.3, fc=color_deconv, ec=color_deconv, linewidth=3)
    ax.text(6.5, 5, 'Transposed\nConv 2×2\nstride=2', ha='left', fontsize=8, 
            fontweight='bold', color=color_deconv)
    
    # Large output
    ax.add_patch(mpatches.FancyBboxPatch((2, 1), 6, 3, boxstyle="round,pad=0.1",
                                          facecolor=color_deconv, edgecolor='white', linewidth=2.5))
    ax.text(5, 2.5, '28×28×256\n(2× resolution)', ha='center', va='center', 
            fontsize=10, fontweight='bold', color='white')
    
    # ===== Step 4: Per-Class Mask Prediction =====
    ax = axes[3]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Step 4: 1×1 Conv\nPer-Class Masks', fontsize=11, fontweight='bold', color=color_output)
    
    # Input
    ax.add_patch(mpatches.FancyBboxPatch((3, 7), 4, 1.5, boxstyle="round,pad=0.1",
                                          facecolor='#555555', edgecolor='white', linewidth=2))
    ax.text(5, 7.75, '28×28×256', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    # 1x1 Conv
    ax.arrow(5, 6.8, 0, -0.8, head_width=0.4, head_length=0.2, fc='white', ec='white', linewidth=2)
    ax.text(6.2, 6.3, '1×1 Conv\nC filters', ha='left', fontsize=8, color='white')
    
    # Multiple class masks
    class_names = ['BG', 'Product', 'Shelf', 'Label']
    colors_classes = ['#2c2c2c', '#4a9eff', '#ff6b6b', '#51cf66']
    
    for i, (cls_name, cls_color) in enumerate(zip(class_names, colors_classes)):
        x_offset = 1 + i * 2
        ax.add_patch(mpatches.FancyBboxPatch((x_offset, 3), 1.5, 2, boxstyle="round,pad=0.05",
                                              facecolor=cls_color, edgecolor='white', linewidth=1.5, alpha=0.7))
        ax.text(x_offset + 0.75, 4, f'{cls_name}\n28×28', ha='center', va='center',
                fontsize=7, fontweight='bold', color='white')
    
    ax.text(5, 1.5, 'C masks (one per class)', ha='center', fontsize=9, fontweight='bold', color=color_output)
    ax.text(5, 0.8, 'Only use mask for predicted class k', ha='center', fontsize=8, 
            style='italic', color='#aaaaaa')
    
    # ===== Step 5: Binary Mask Output =====
    ax = axes[4]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Step 5: Sigmoid\nBinary Mask', fontsize=11, fontweight='bold', color=color_output)
    
    # Draw 28x28 grid representation
    grid_size = 28
    cell_size = 6 / grid_size
    start_x, start_y = 2, 2
    
    # Create a simple mask pattern (product shape)
    for i in range(grid_size):
        for j in range(grid_size):
            # Create an ellipse mask pattern
            dist = ((i - grid_size/2)**2 / (grid_size/3)**2 + 
                   (j - grid_size/2)**2 / (grid_size/4)**2)
            if dist < 1:
                intensity = max(0, 1 - dist)  # Gradient from center
                color = plt.cm.YlOrRd(intensity)
                ax.add_patch(mpatches.Rectangle((start_x + j*cell_size, start_y + i*cell_size),
                                                 cell_size, cell_size, facecolor=color, 
                                                 edgecolor='none'))
    
    # Border
    ax.add_patch(mpatches.Rectangle((start_x, start_y), 6, 6, facecolor='none',
                                     edgecolor='white', linewidth=2.5))
    
    ax.text(5, 8.5, 'Binary Mask\n28×28×1', ha='center', fontsize=10, fontweight='bold', color='white')
    ax.text(5, 1, 'Sigmoid → [0, 1]\nThreshold at 0.5', ha='center', fontsize=8, 
            style='italic', color='#aaaaaa')
    
    # Add colorbar legend
    cbar_ax = fig.add_axes([0.82, 0.15, 0.015, 0.15])
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='YlOrRd'), cax=cbar_ax)
    cbar.set_label('Mask Probability', fontsize=8, color='white')
    cbar.ax.tick_params(labelsize=7, colors='white')
    
    # Overall annotation
    fig.text(0.5, 0.02, 'Loss: Binary cross-entropy per pixel (averaged over 28×28 mask)',
             ha='center', fontsize=10, style='italic', color='#aaaaaa')
    
    plt.tight_layout(rect=[0, 0.04, 0.98, 0.96])
    
    output_path = Path(__file__).parent.parent / 'img' / 'ch06-mask-prediction.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1e1e1e')
    print(f"✅ Generated: {output_path}")
    plt.close()

if __name__ == '__main__':
    generate_mask_prediction()
