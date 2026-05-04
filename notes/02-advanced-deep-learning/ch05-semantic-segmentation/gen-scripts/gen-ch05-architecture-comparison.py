"""
Generate semantic segmentation architecture comparison animation.

Shows the progression from FCN to U-Net to DeepLab, highlighting:
- FCN: Coarse upsampling with limited skip connections
- U-Net: Symmetric encoder-decoder with skip connections at every level
- DeepLab: Atrous convolutions maintaining spatial resolution

Output: ch05-segmentation-architectures.gif
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')

# Dark theme (matches ML authoring guide)
plt.style.use('dark_background')

def create_architecture_comparison():
    """Create animated comparison of FCN, U-Net, and DeepLab architectures."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('#1a1a2e')
    
    architectures = ['FCN-8s', 'U-Net', 'DeepLabV3+']
    colors = {'encoder': '#3498db', 'decoder': '#e74c3c', 'skip': '#2ecc71', 'aspp': '#f39c12'}
    
    for idx, (ax, arch_name) in enumerate(zip(axes, architectures)):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(arch_name, fontsize=16, fontweight='bold', pad=20)
        
        if arch_name == 'FCN-8s':
            # Encoder (downsampling)
            encoder_heights = [8, 6, 4, 2, 1]
            for i, h in enumerate(encoder_heights):
                x = 1 + i * 1.5
                rect = patches.Rectangle((x, 1), 1, h, linewidth=2, 
                                        edgecolor='white', facecolor=colors['encoder'], alpha=0.7)
                ax.add_patch(rect)
                ax.text(x + 0.5, 0.5, f'{int(512/(2**i))}', ha='center', fontsize=9, color='white')
            
            # Decoder (upsampling with 2 skip connections)
            decoder_heights = [2, 4, 6, 8]
            for i, h in enumerate(decoder_heights):
                x = 8.5 - i * 1.5
                rect = patches.Rectangle((x, 1), 1, h, linewidth=2, 
                                        edgecolor='white', facecolor=colors['decoder'], alpha=0.7)
                ax.add_patch(rect)
            
            # Skip connections from conv3 and conv4 only
            ax.arrow(3.5, 4, 2.5, 0, head_width=0.3, head_length=0.2, fc=colors['skip'], ec=colors['skip'], linewidth=2)
            ax.arrow(5, 6, 1, 0, head_width=0.3, head_length=0.2, fc=colors['skip'], ec=colors['skip'], linewidth=2)
            
            ax.text(5, 9.5, 'Limited skip connections', ha='center', fontsize=11, color='#f39c12')
            ax.text(5, 9, '(Conv3 + Conv4 only)', ha='center', fontsize=9, color='#95a5a6')
            
        elif arch_name == 'U-Net':
            # Encoder
            encoder_heights = [8, 6, 4, 2, 1]
            for i, h in enumerate(encoder_heights):
                x = 1 + i * 1.5
                rect = patches.Rectangle((x, 5 - h/2), 1, h, linewidth=2, 
                                        edgecolor='white', facecolor=colors['encoder'], alpha=0.7)
                ax.add_patch(rect)
                ax.text(x + 0.5, 0.5, f'{int(512/(2**i))}', ha='center', fontsize=9, color='white')
            
            # Decoder (symmetric)
            decoder_heights = [1, 2, 4, 6, 8]
            for i, h in enumerate(decoder_heights):
                x = 8.5 - i * 1.5
                rect = patches.Rectangle((x, 5 - h/2), 1, h, linewidth=2, 
                                        edgecolor='white', facecolor=colors['decoder'], alpha=0.7)
                ax.add_patch(rect)
            
            # Skip connections at EVERY level (U-shaped)
            skip_levels = [(1.5, 9), (3, 8), (4.5, 6), (6, 4)]
            for (x, y) in skip_levels:
                ax.arrow(x, y, 2 + (9-y)*0.3, 0, head_width=0.25, head_length=0.15, 
                        fc=colors['skip'], ec=colors['skip'], linewidth=2.5, linestyle='--')
            
            ax.text(5, 9.5, 'Skip at every resolution', ha='center', fontsize=11, color='#2ecc71', fontweight='bold')
            ax.text(5, 9, '(Concatenate, not add)', ha='center', fontsize=9, color='#95a5a6')
            
        else:  # DeepLabV3+
            # Encoder with atrous convolutions (maintains resolution)
            encoder_heights = [8, 7, 6, 5, 4]
            for i, h in enumerate(encoder_heights):
                x = 1 + i * 1.5
                rect = patches.Rectangle((x, 1), 1, h, linewidth=2, 
                                        edgecolor='white', facecolor=colors['encoder'], alpha=0.7)
                ax.add_patch(rect)
                if i >= 3:
                    ax.text(x + 0.5, h + 1.3, f'r={6*2**i}', ha='center', fontsize=8, color='#f39c12')
            
            # ASPP module (parallel branches)
            aspp_x = 8
            aspp_heights = [2, 2.5, 3, 3.5, 2]
            aspp_labels = ['1×1', 'r=6', 'r=12', 'r=18', 'pool']
            for i, (h, label) in enumerate(zip(aspp_heights, aspp_labels)):
                y = 1 + i * 1.2
                rect = patches.Rectangle((aspp_x, y), 0.8, 0.8, linewidth=2, 
                                        edgecolor='white', facecolor=colors['aspp'], alpha=0.8)
                ax.add_patch(rect)
                ax.text(aspp_x + 0.4, y + 0.4, label, ha='center', va='center', fontsize=8, fontweight='bold')
            
            # Decoder (lightweight)
            decoder_x = 9.2
            rect = patches.Rectangle((decoder_x, 1), 0.6, 7, linewidth=2, 
                                    edgecolor='white', facecolor=colors['decoder'], alpha=0.7)
            ax.add_patch(rect)
            
            ax.text(8.5, 9.5, 'Atrous Spatial Pyramid Pooling', ha='center', fontsize=11, color='#f39c12', fontweight='bold')
            ax.text(8.5, 9, '(Multi-scale context)', ha='center', fontsize=9, color='#95a5a6')
    
    plt.tight_layout()
    return fig

def create_animation():
    """Generate animated GIF showing architecture evolution."""
    fig = create_architecture_comparison()
    
    def update(frame):
        # Animation shows sequential highlighting (static for this version)
        return fig,
    
    anim = FuncAnimation(fig, update, frames=60, interval=100, blit=False)
    
    # Save as GIF
    output_path = '../img/ch05-segmentation-architectures.gif'
    anim.save(output_path, writer='pillow', fps=10, dpi=150)
    print(f"✅ Saved animation to {output_path}")
    
    plt.close()

if __name__ == '__main__':
    create_animation()
    print("Animation generation complete!")
