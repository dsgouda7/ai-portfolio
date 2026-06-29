"""
Generate U-Net skip connection flow animation.

Visualizes how skip connections preserve spatial information:
- Encoder: Downsampling path (extract features, lose resolution)
- Skip connections: Copy features from encoder to decoder
- Decoder: Upsampling path (recover resolution with fine details)

Shows symmetric U-shaped architecture with highlighted feature flow.

Output: ch05-unet-skip-connections.gif
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

# Dark theme
plt.style.use('dark_background')

def create_unet_diagram():
    """Create U-Net architecture diagram with skip connections."""
    
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Colors
    encoder_color = '#3498db'
    decoder_color = '#e74c3c'
    skip_color = '#2ecc71'
    bottleneck_color = '#f39c12'
    
    # Encoder blocks (left side, descending)
    encoder_specs = [
        (2, 8, 1.5, 6, '512×512\n64ch'),
        (4, 7, 1.5, 5, '256×256\n128ch'),
        (6, 6, 1.5, 4, '128×128\n256ch'),
        (8, 5, 1.5, 3, '64×64\n512ch'),
    ]
    
    encoder_patches = []
    for x, y, w, h, label in encoder_specs:
        rect = patches.Rectangle((x, y), w, h, linewidth=3, 
                                edgecolor='white', facecolor=encoder_color, alpha=0.7)
        ax.add_patch(rect)
        encoder_patches.append((x, y, w, h))
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', 
                fontsize=11, fontweight='bold', color='white')
    
    # Bottleneck (center)
    bottleneck_x, bottleneck_y = 10, 4
    bottleneck_w, bottleneck_h = 1.5, 2
    rect = patches.Rectangle((bottleneck_x, bottleneck_y), bottleneck_w, bottleneck_h, 
                            linewidth=3, edgecolor='white', facecolor=bottleneck_color, alpha=0.8)
    ax.add_patch(rect)
    ax.text(bottleneck_x + bottleneck_w/2, bottleneck_y + bottleneck_h/2, 
            '32×32\n1024ch', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    
    # Decoder blocks (right side, ascending)
    decoder_specs = [
        (12, 5, 1.5, 3, '64×64\n512ch'),
        (14, 6, 1.5, 4, '128×128\n256ch'),
        (16, 7, 1.5, 5, '256×256\n128ch'),
        (18, 8, 1.5, 6, '512×512\n64ch'),
    ]
    
    decoder_patches = []
    for x, y, w, h, label in decoder_specs:
        rect = patches.Rectangle((x, y), w, h, linewidth=3, 
                                edgecolor='white', facecolor=decoder_color, alpha=0.7)
        ax.add_patch(rect)
        decoder_patches.append((x, y, w, h))
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', 
                fontsize=11, fontweight='bold', color='white')
    
    # Downsampling arrows (max pooling)
    for i in range(len(encoder_specs) - 1):
        x1 = encoder_specs[i][0] + encoder_specs[i][2]
        y1 = encoder_specs[i][1]
        x2 = encoder_specs[i+1][0]
        y2 = encoder_specs[i+1][1] + encoder_specs[i+1][3]/2
        
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1 + encoder_specs[i][3]/2),
                   arrowprops=dict(arrowstyle='->', lw=2.5, color=encoder_color))
        ax.text((x1 + x2)/2, (y1 + y2)/2 - 0.3, 'MaxPool', fontsize=9, 
                ha='center', color='#95a5a6', style='italic')
    
    # Final encoder to bottleneck
    x1 = encoder_specs[-1][0] + encoder_specs[-1][2]
    y1 = encoder_specs[-1][1] + encoder_specs[-1][3]/2
    x2 = bottleneck_x
    y2 = bottleneck_y + bottleneck_h/2
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
               arrowprops=dict(arrowstyle='->', lw=2.5, color=encoder_color))
    
    # Upsampling arrows (transposed conv)
    x1 = bottleneck_x + bottleneck_w
    y1 = bottleneck_y + bottleneck_h/2
    x2 = decoder_specs[0][0]
    y2 = decoder_specs[0][1] + decoder_specs[0][3]/2
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
               arrowprops=dict(arrowstyle='->', lw=2.5, color=decoder_color))
    
    for i in range(len(decoder_specs) - 1):
        x1 = decoder_specs[i][0] + decoder_specs[i][2]
        y1 = decoder_specs[i][1] + decoder_specs[i][3]/2
        x2 = decoder_specs[i+1][0]
        y2 = decoder_specs[i+1][1] + decoder_specs[i+1][3]/2
        
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2.5, color=decoder_color))
        ax.text((x1 + x2)/2, (y1 + y2)/2 + 0.3, 'UpConv', fontsize=9, 
                ha='center', color='#95a5a6', style='italic')
    
    # Skip connections (horizontal, curved)
    for i in range(len(encoder_specs)):
        # From encoder to corresponding decoder
        enc_x = encoder_specs[i][0] + encoder_specs[i][2]
        enc_y = encoder_specs[i][1] + encoder_specs[i][3]/2
        dec_x = decoder_specs[len(decoder_specs) - 1 - i][0]
        dec_y = decoder_specs[len(decoder_specs) - 1 - i][1] + decoder_specs[len(decoder_specs) - 1 - i][3]/2
        
        # Curved arrow
        mid_y = enc_y + 0.5 if i % 2 == 0 else enc_y - 0.5
        ax.annotate('', xy=(dec_x, dec_y), xytext=(enc_x, enc_y),
                   arrowprops=dict(arrowstyle='->', lw=3, color=skip_color, 
                                 connectionstyle=f"arc3,rad=0.3", linestyle='--'))
        
        # Label
        label_x = (enc_x + dec_x) / 2
        label_y = (enc_y + dec_y) / 2 + (1 if i % 2 == 0 else -1)
        ax.text(label_x, label_y, 'Skip\nConnection', fontsize=9, ha='center', 
                color=skip_color, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', 
                facecolor='#1a1a2e', edgecolor=skip_color, linewidth=2))
    
    # Title and annotations
    ax.text(10, 11.5, 'U-Net Architecture: Symmetric Encoder-Decoder with Skip Connections', 
            fontsize=18, ha='center', fontweight='bold', color='#ecf0f1')
    
    ax.text(5, 0.5, '← Encoder (Contracting Path)', fontsize=12, ha='center', color=encoder_color, fontweight='bold')
    ax.text(15, 0.5, 'Decoder (Expanding Path) →', fontsize=12, ha='center', color=decoder_color, fontweight='bold')
    ax.text(10, 2.5, 'Bottleneck', fontsize=12, ha='center', color=bottleneck_color, fontweight='bold')
    
    # Legend
    legend_y = 1.5
    ax.text(1, legend_y, '■', fontsize=20, color=encoder_color)
    ax.text(1.5, legend_y, 'Downsampling (feature extraction)', fontsize=10, va='center')
    
    ax.text(7, legend_y, '■', fontsize=20, color=decoder_color)
    ax.text(7.5, legend_y, 'Upsampling (resolution recovery)', fontsize=10, va='center')
    
    ax.text(13, legend_y, '■', fontsize=20, color=skip_color)
    ax.text(13.5, legend_y, 'Skip connections (fine detail preservation)', fontsize=10, va='center')
    
    plt.tight_layout()
    return fig

def create_animation():
    """Generate animated GIF showing U-Net architecture."""
    fig = create_unet_diagram()
    
    def update(frame):
        # Could animate information flow in future version
        return fig,
    
    anim = FuncAnimation(fig, update, frames=60, interval=100, blit=False)
    
    output_path = '../img/ch05-unet-skip-connections.gif'
    anim.save(output_path, writer='pillow', fps=10, dpi=150)
    print(f"✅ Saved U-Net skip connections animation to {output_path}")
    
    plt.close()

if __name__ == '__main__':
    create_animation()
    print("U-Net diagram generation complete!")
