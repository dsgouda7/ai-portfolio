"""
Generate animated GIF showing gradient flow through skip connections.

Output: ch01-resnet-gradient-flow.gif

Animation: 
- Frame 1-5: Forward pass (data flows down)
- Frame 6-10: Backward pass (gradients flow up through skip + main path)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import os

# Dark theme
plt.style.use('dark_background')

def create_frame(frame_num, total_frames):
    """Create a single frame of the animation."""
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Determine phase
    if frame_num <= total_frames // 2:
        phase = 'forward'
        progress = frame_num / (total_frames // 2)
    else:
        phase = 'backward'
        progress = (frame_num - total_frames // 2) / (total_frames // 2)
    
    # Title
    title_text = 'Forward Pass: Data Flow →' if phase == 'forward' else 'Backward Pass: Gradient Flow ←'
    ax.text(5, 9.5, title_text, ha='center', va='top', fontsize=16, fontweight='bold', color='white')
    
    # Input
    input_color = '#3b82f6' if phase == 'forward' else '#6b7280'
    input_box = patches.FancyBboxPatch((4, 8), 2, 0.5, boxstyle="round,pad=0.1",
                                       edgecolor=input_color, facecolor='#1e3a8a', linewidth=2)
    ax.add_patch(input_box)
    ax.text(5, 8.25, 'Input x', ha='center', va='center', fontsize=11, color='white', fontweight='bold')
    
    # Residual block layers
    layers = [
        (4, 6.5, 'Conv1 + BN + ReLU'),
        (4, 5, 'Conv2 + BN'),
    ]
    
    for i, (x, y, label) in enumerate(layers):
        layer_active = (phase == 'forward' and progress > (i + 1) / 3) or \
                       (phase == 'backward' and progress > (2 - i) / 3)
        layer_color = '#10b981' if layer_active else '#374151'
        edge_color = '#10b981' if layer_active else '#6b7280'
        
        layer_box = patches.FancyBboxPatch((x, y), 2, 0.7, boxstyle="round,pad=0.1",
                                           edgecolor=edge_color, facecolor=layer_color, linewidth=2, alpha=0.7)
        ax.add_patch(layer_box)
        ax.text(x + 1, y + 0.35, label, ha='center', va='center', fontsize=9, color='white')
    
    # Skip connection (curved path)
    skip_active = (phase == 'forward' and progress > 0.5) or (phase == 'backward')
    skip_color = '#f59e0b' if skip_active else '#6b7280'
    skip_x = np.linspace(6.5, 6.5, 100)
    skip_y = np.linspace(8, 3.8, 100)
    curve_offset = 1.2 * np.sin(np.linspace(0, np.pi, 100))
    ax.plot(skip_x + curve_offset, skip_y, color=skip_color, linewidth=3, linestyle='--', alpha=0.9)
    ax.text(8.5, 5.8, 'Skip\n(+x)', ha='center', va='center', fontsize=10, 
            color=skip_color, fontweight='bold')
    
    # Addition node
    add_active = (phase == 'forward' and progress > 0.8) or (phase == 'backward' and progress > 0.3)
    add_color = '#f59e0b' if add_active else '#6b7280'
    add_circle = patches.Circle((5, 3.8), 0.35, edgecolor=add_color, facecolor='#1a1a2e', linewidth=2)
    ax.add_patch(add_circle)
    ax.text(5, 3.8, '+', ha='center', va='center', fontsize=18, color=add_color, fontweight='bold')
    
    # ReLU
    relu_active = (phase == 'forward' and progress > 0.9) or (phase == 'backward' and progress > 0.2)
    relu_color = '#ef4444' if relu_active else '#374151'
    relu_edge = '#ef4444' if relu_active else '#6b7280'
    relu_box = patches.FancyBboxPatch((4, 2.5), 2, 0.6, boxstyle="round,pad=0.1",
                                      edgecolor=relu_edge, facecolor=relu_color, linewidth=2, alpha=0.7)
    ax.add_patch(relu_box)
    ax.text(5, 2.8, 'ReLU', ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    
    # Output
    output_active = phase == 'backward' and progress > 0.1
    output_color = '#ef4444' if output_active else '#6b7280'
    output_box = patches.FancyBboxPatch((4, 1), 2, 0.5, boxstyle="round,pad=0.1",
                                        edgecolor=output_color, facecolor='#b91c1c', linewidth=2, alpha=0.7)
    ax.add_patch(output_box)
    ax.text(5, 1.25, 'Output ℋ(x)', ha='center', va='center', fontsize=11, color='white', fontweight='bold')
    
    # Arrows
    if phase == 'forward':
        # Forward arrows (downward)
        if progress > 0.1:
            ax.arrow(5, 7.8, 0, -0.8, head_width=0.15, head_length=0.1, fc='white', ec='white', linewidth=2, alpha=min(progress*2, 1))
        if progress > 0.4:
            ax.arrow(5, 6.3, 0, -0.6, head_width=0.15, head_length=0.1, fc='white', ec='white', linewidth=2, alpha=min((progress-0.3)*2, 1))
        if progress > 0.7:
            ax.arrow(5, 4.8, 0, -0.8, head_width=0.15, head_length=0.1, fc='white', ec='white', linewidth=2, alpha=min((progress-0.6)*2, 1))
        if progress > 0.9:
            ax.arrow(5, 3.4, 0, -0.7, head_width=0.15, head_length=0.1, fc='white', ec='white', linewidth=2, alpha=min((progress-0.8)*2, 1))
        if progress > 0.95:
            ax.arrow(5, 2.4, 0, -0.7, head_width=0.15, head_length=0.1, fc='white', ec='white', linewidth=2)
    else:
        # Backward arrows (upward) - gradients
        if progress > 0.1:
            ax.arrow(5, 1.6, 0, 0.7, head_width=0.15, head_length=0.1, fc='#ef4444', ec='#ef4444', linewidth=2, alpha=min(progress*2, 1))
        if progress > 0.3:
            ax.arrow(5, 3.3, 0, 0.8, head_width=0.15, head_length=0.1, fc='#f59e0b', ec='#f59e0b', linewidth=2, alpha=min((progress-0.2)*2, 1))
        if progress > 0.5:
            ax.arrow(5, 5.2, 0, 0.6, head_width=0.15, head_length=0.1, fc='#10b981', ec='#10b981', linewidth=2, alpha=min((progress-0.4)*2, 1))
        if progress > 0.7:
            ax.arrow(5, 6.7, 0, 0.8, head_width=0.15, head_length=0.1, fc='#10b981', ec='#10b981', linewidth=2, alpha=min((progress-0.6)*2, 1))
        # Skip path gradient
        if progress > 0.3:
            ax.arrow(7.5, 3.9, 0, 0.5, head_width=0.15, head_length=0.1, fc='#f59e0b', ec='#f59e0b', linewidth=2, alpha=min((progress-0.2)*2, 1))
    
    # Legend
    if phase == 'backward':
        legend_y = 0.2
        ax.text(1, legend_y, '💡 Gradient flows through BOTH paths:', fontsize=10, color='#f59e0b', fontweight='bold')
        ax.text(1, legend_y - 0.3, '   • Main path (green): ∂F/∂x', fontsize=9, color='#10b981')
        ax.text(1, legend_y - 0.6, '   • Skip path (orange): +1 (preserved!)', fontsize=9, color='#f59e0b')
    
    plt.tight_layout()
    
    # Save frame
    frame_path = f'../img/temp_frame_{frame_num:03d}.png'
    plt.savefig(frame_path, dpi=100, facecolor='#1a1a2e', bbox_inches='tight')
    plt.close()
    
    return frame_path

# Generate frames
print('Generating animation frames...')
total_frames = 20
frame_paths = []

for i in range(1, total_frames + 1):
    frame_path = create_frame(i, total_frames)
    frame_paths.append(frame_path)
    print(f'  Frame {i}/{total_frames}')

# Create GIF
print('Creating GIF...')
frames = [Image.open(fp) for fp in frame_paths]

# Save as GIF (loop forever, 200ms per frame)
frames[0].save(
    '../img/ch01-resnet-gradient-flow.gif',
    save_all=True,
    append_images=frames[1:],
    duration=200,
    loop=0
)

# Clean up temporary frames
for fp in frame_paths:
    os.remove(fp)

print('✅ Generated: ../img/ch01-resnet-gradient-flow.gif')
