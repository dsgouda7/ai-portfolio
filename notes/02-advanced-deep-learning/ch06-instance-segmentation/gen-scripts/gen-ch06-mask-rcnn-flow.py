"""
Generate Mask R-CNN architecture and information flow animation.

Visualizes:
1. Backbone (ResNet-50 + FPN)
2. RPN (Region Proposal Network)
3. RoIAlign (bilinear interpolation, no quantization)
4. Detection head (classification + box regression)
5. Mask head (28×28 binary mask prediction)

Output: ch06-mask-rcnn-flow.gif
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

# Dark theme
plt.style.use('dark_background')

def create_mask_rcnn_diagram():
    """Create Mask R-CNN end-to-end architecture diagram."""
    
    fig, ax = plt.subplots(figsize=(18, 12))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Colors
    backbone_color = '#3498db'
    rpn_color = '#9b59b6'
    roi_color = '#f39c12'
    detection_color = '#e74c3c'
    mask_color = '#2ecc71'
    
    # Title
    ax.text(10, 13.5, 'Mask R-CNN Architecture — Instance Segmentation Pipeline', 
            fontsize=18, ha='center', fontweight='bold', color='#ecf0f1')
    
    # 1. Input Image
    img_rect = patches.Rectangle((1, 9), 2, 3, linewidth=3, 
                                edgecolor='white', facecolor='#2c3e50', alpha=0.8)
    ax.add_patch(img_rect)
    ax.text(2, 10.5, 'Input\nImage\n512×512', ha='center', va='center', 
            fontsize=11, fontweight='bold', color='white')
    
    # 2. Backbone (ResNet-50 + FPN)
    backbone_x, backbone_y = 4, 9
    backbone_heights = [3, 2.5, 2, 1.5]
    for i, h in enumerate(backbone_heights):
        rect = patches.Rectangle((backbone_x + i*0.6, backbone_y + (3-h)/2), 
                                0.5, h, linewidth=2, 
                                edgecolor='white', facecolor=backbone_color, alpha=0.7)
        ax.add_patch(rect)
    
    ax.text(backbone_x + 1.2, 7.8, 'ResNet-50 + FPN', ha='center', fontsize=11, 
            fontweight='bold', color=backbone_color)
    ax.text(backbone_x + 1.2, 7.3, '(Feature Extraction)', ha='center', fontsize=9, 
            color='#95a5a6', style='italic')
    
    # Arrow from input to backbone
    ax.annotate('', xy=(backbone_x, 10.5), xytext=(3, 10.5),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='white'))
    
    # 3. RPN (Region Proposal Network)
    rpn_x, rpn_y = 7, 9.5
    rpn_rect = patches.Rectangle((rpn_x, rpn_y), 1.5, 2, linewidth=3, 
                                edgecolor='white', facecolor=rpn_color, alpha=0.8)
    ax.add_patch(rpn_rect)
    ax.text(rpn_x + 0.75, rpn_y + 1, 'RPN\n~2000\nProposals', ha='center', va='center', 
            fontsize=10, fontweight='bold', color='white')
    
    # Arrow backbone to RPN
    ax.annotate('', xy=(rpn_x, 10.5), xytext=(backbone_x + 2.4, 10.5),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='white'))
    
    # 4. RoIAlign (highlighted in gold)
    roialign_x, roialign_y = 9.5, 9.5
    roialign_rect = patches.FancyBboxPatch((roialign_x, roialign_y), 1.8, 2, 
                                          boxstyle="round,pad=0.1", linewidth=4, 
                                          edgecolor=roi_color, facecolor='#1a1a2e', alpha=1.0)
    ax.add_patch(roialign_rect)
    ax.text(roialign_x + 0.9, roialign_y + 1.3, 'RoIAlign', ha='center', fontsize=12, 
            fontweight='bold', color=roi_color)
    ax.text(roialign_x + 0.9, roialign_y + 0.7, '7×7 features\n(bilinear)', ha='center', 
            fontsize=9, color='#95a5a6')
    
    # Arrow RPN to RoIAlign
    ax.annotate('', xy=(roialign_x, 10.5), xytext=(rpn_x + 1.5, 10.5),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='white'))
    
    # 5. Split into two heads
    split_x = roialign_x + 1.8
    split_y = 10.5
    
    # Detection Head (top branch)
    det_x, det_y = 12.5, 11
    det_rect = patches.Rectangle((det_x, det_y), 2, 1.5, linewidth=3, 
                                edgecolor='white', facecolor=detection_color, alpha=0.8)
    ax.add_patch(det_rect)
    ax.text(det_x + 1, det_y + 0.75, 'Detection\nHead', ha='center', va='center', 
            fontsize=11, fontweight='bold', color='white')
    
    # Arrow to detection head
    ax.annotate('', xy=(det_x, det_y + 0.75), xytext=(split_x, split_y),
               arrowprops=dict(arrowstyle='->', lw=2.5, color=detection_color,
                             connectionstyle="arc3,rad=0.2"))
    
    # Mask Head (bottom branch)
    mask_x, mask_y = 12.5, 8.5
    mask_rect = patches.Rectangle((mask_x, mask_y), 2, 1.5, linewidth=3, 
                                 edgecolor='white', facecolor=mask_color, alpha=0.8)
    ax.add_patch(mask_rect)
    ax.text(mask_x + 1, mask_y + 0.75, 'Mask\nHead', ha='center', va='center', 
            fontsize=11, fontweight='bold', color='white')
    
    # Arrow to mask head
    ax.annotate('', xy=(mask_x, mask_y + 0.75), xytext=(split_x, split_y),
               arrowprops=dict(arrowstyle='->', lw=2.5, color=mask_color,
                             connectionstyle="arc3,rad=-0.2"))
    
    # 6. Outputs
    # Detection output
    det_out_x, det_out_y = 15.5, 11
    det_out_rect = patches.Rectangle((det_out_x, det_out_y), 2.5, 1.5, linewidth=2, 
                                    edgecolor=detection_color, facecolor='#2c3e50', alpha=0.8)
    ax.add_patch(det_out_rect)
    ax.text(det_out_x + 1.25, det_out_y + 0.75, 'Class + BBox\n[x1,y1,x2,y2]', 
            ha='center', va='center', fontsize=10, color='white')
    
    ax.annotate('', xy=(det_out_x, det_out_y + 0.75), xytext=(det_x + 2, det_y + 0.75),
               arrowprops=dict(arrowstyle='->', lw=2.5, color=detection_color))
    
    # Mask output
    mask_out_x, mask_out_y = 15.5, 8.5
    mask_out_rect = patches.Rectangle((mask_out_x, mask_out_y), 2.5, 1.5, linewidth=2, 
                                     edgecolor=mask_color, facecolor='#2c3e50', alpha=0.8)
    ax.add_patch(mask_out_rect)
    ax.text(mask_out_x + 1.25, mask_out_y + 0.75, '28×28\nBinary Mask', 
            ha='center', va='center', fontsize=10, color='white')
    
    ax.annotate('', xy=(mask_out_x, mask_out_y + 0.75), xytext=(mask_x + 2, mask_y + 0.75),
               arrowprops=dict(arrowstyle='->', lw=2.5, color=mask_color))
    
    # Bottom: RoIAlign detail callout
    callout_y = 6
    ax.text(10, callout_y + 1.5, 'RoIAlign Innovation:', fontsize=13, ha='center', 
            fontweight='bold', color=roi_color)
    
    # Old way (RoI Pooling)
    ax.text(5, callout_y, '❌ RoI Pooling (Faster R-CNN)', fontsize=11, ha='center', 
            color='#e74c3c', fontweight='bold')
    ax.text(5, callout_y - 0.5, 'Rounds coordinates → quantization errors', 
            fontsize=9, ha='center', color='#95a5a6')
    ax.text(5, callout_y - 1, '[15.7, 23.3] → [16, 23]', fontsize=9, ha='center', 
            color='#7f8c8d', family='monospace')
    
    # New way (RoIAlign)
    ax.text(15, callout_y, '✅ RoIAlign (Mask R-CNN)', fontsize=11, ha='center', 
            color='#2ecc71', fontweight='bold')
    ax.text(15, callout_y - 0.5, 'Bilinear interpolation → sub-pixel precision', 
            fontsize=9, ha='center', color='#95a5a6')
    ax.text(15, callout_y - 1, '[15.7, 23.3] preserved exactly', fontsize=9, ha='center', 
            color='#7f8c8d', family='monospace')
    
    # Loss components
    loss_y = 3
    ax.text(10, loss_y + 1, 'Multi-Task Loss:', fontsize=13, ha='center', 
            fontweight='bold', color='#ecf0f1')
    
    loss_components = [
        ('L_cls', 'Classification', detection_color),
        ('L_bbox', 'Box Regression', detection_color),
        ('L_mask', 'Mask Prediction', mask_color)
    ]
    
    for i, (symbol, name, color) in enumerate(loss_components):
        x = 4 + i * 5
        ax.text(x, loss_y, f'${symbol}$', fontsize=16, ha='center', 
                fontweight='bold', color=color)
        ax.text(x, loss_y - 0.5, name, fontsize=10, ha='center', color='#95a5a6')
    
    ax.text(10, loss_y - 1.5, '$\\mathcal{L}_{total} = \\mathcal{L}_{cls} + \\mathcal{L}_{bbox} + \\mathcal{L}_{mask}$', 
            fontsize=14, ha='center', color='white', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#2c3e50', edgecolor='#ecf0f1', linewidth=2))
    
    plt.tight_layout()
    return fig

def create_animation():
    """Generate animated GIF showing Mask R-CNN flow."""
    fig = create_mask_rcnn_diagram()
    
    def update(frame):
        # Static diagram (could animate flow in future)
        return fig,
    
    anim = FuncAnimation(fig, update, frames=60, interval=100, blit=False)
    
    output_path = '../img/ch06-mask-rcnn-flow.gif'
    anim.save(output_path, writer='pillow', fps=10, dpi=150)
    print(f"✅ Saved Mask R-CNN architecture animation to {output_path}")
    
    plt.close()

if __name__ == '__main__':
    create_animation()
    print("Mask R-CNN diagram generation complete!")
