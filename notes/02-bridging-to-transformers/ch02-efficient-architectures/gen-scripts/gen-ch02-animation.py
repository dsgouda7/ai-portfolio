"""
Generate animated GIF showing depthwise separable convolution operation.

Output: ch02-depthwise-separable-animation.gif

Animation shows:
- Frame 1-5: Standard 3×3 conv (all filters at once)
- Frame 6-10: Depthwise step (one filter per channel)
- Frame 11-15: Pointwise step (1×1 mixing channels)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import os

plt.style.use('dark_background')

def create_frame(frame_num, total_frames):
    """Create a single animation frame."""
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Determine animation phase
    phase_duration = total_frames // 3
    if frame_num < phase_duration:
        phase = 'standard'
        progress = frame_num / phase_duration
    elif frame_num < 2 * phase_duration:
        phase = 'depthwise'
        progress = (frame_num - phase_duration) / phase_duration
    else:
        phase = 'pointwise'
        progress = (frame_num - 2 * phase_duration) / phase_duration
    
    # Title
    if phase == 'standard':
        title = 'Standard 3×3 Convolution (Expensive)'
        color = '#ef4444'
    elif phase == 'depthwise':
        title = 'Step 1: Depthwise 3×3 (Spatial Filtering)'
        color = '#f59e0b'
    else:
        title = 'Step 2: Pointwise 1×1 (Channel Mixing)'
        color = '#10b981'
    
    ax.text(6, 9.5, title, ha='center', va='top', fontsize=16, fontweight='bold', color=color)
    
    # Input feature map
    input_active = progress > 0.1
    input_color = color if input_active else '#6b7280'
    input_box = patches.FancyBboxPatch((1, 6), 2, 1.5, boxstyle="round,pad=0.1",
                                       edgecolor=input_color, facecolor='#1e3a8a', linewidth=2, alpha=0.8)
    ax.add_patch(input_box)
    ax.text(2, 6.75, '56×56\n×128', ha='center', va='center', fontsize=11, color='white', fontweight='bold')
    
    if phase == 'standard':
        # Standard convolution (all channels at once)
        conv_active = progress > 0.3
        conv_color = '#ef4444' if conv_active else '#6b7280'
        
        # Big conv block
        conv_box = patches.FancyBboxPatch((4.5, 5.5), 3, 2.5, boxstyle="round,pad=0.1",
                                          edgecolor=conv_color, facecolor='#b91c1c', linewidth=3, alpha=0.7)
        ax.add_patch(conv_box)
        ax.text(6, 7.5, '3×3 Conv', ha='center', va='top', fontsize=13, color='white', fontweight='bold')
        ax.text(6, 6.8, '256 kernels', ha='center', va='center', fontsize=10, color='white')
        ax.text(6, 6.3, '(3×3×128 each)', ha='center', va='center', fontsize=9, color='#fca5a5')
        
        # Arrow
        if progress > 0.2:
            ax.arrow(3.2, 6.75, 1, 0, head_width=0.3, head_length=0.2, fc='white', ec='white', linewidth=2)
        
        # Cost annotation (animated)
        if progress > 0.5:
            cost_alpha = min((progress - 0.5) * 2, 1)
            cost_box = patches.FancyBboxPatch((4.5, 4.5), 3, 0.8, boxstyle="round,pad=0.1",
                                              edgecolor='#ef4444', facecolor='#1a1a2e', linewidth=2, 
                                              linestyle='--', alpha=cost_alpha)
            ax.add_patch(cost_box)
            ax.text(6, 4.9, 'FLOPs: 925M', ha='center', va='center', fontsize=10, 
                    color='#ef4444', fontweight='bold', alpha=cost_alpha)
        
    elif phase == 'depthwise':
        # Depthwise convolution (per-channel filters)
        dw_active = progress > 0.3
        dw_color = '#f59e0b' if dw_active else '#6b7280'
        
        # Depthwise block
        dw_box = patches.FancyBboxPatch((4.5, 5.5), 3, 2.5, boxstyle="round,pad=0.1",
                                        edgecolor=dw_color, facecolor='#b45309', linewidth=3, alpha=0.7)
        ax.add_patch(dw_box)
        ax.text(6, 7.5, 'Depthwise 3×3', ha='center', va='top', fontsize=13, color='white', fontweight='bold')
        ax.text(6, 6.8, '128 filters', ha='center', va='center', fontsize=10, color='white')
        ax.text(6, 6.3, '(3×3×1 each)', ha='center', va='center', fontsize=9, color='#fbbf24')
        ax.text(6, 5.9, 'One per channel!', ha='center', va='center', fontsize=9, color='#fbbf24', style='italic')
        
        # Arrow
        if progress > 0.2:
            ax.arrow(3.2, 6.75, 1, 0, head_width=0.3, head_length=0.2, fc='white', ec='white', linewidth=2)
        
        # Intermediate output
        if progress > 0.7:
            inter_alpha = min((progress - 0.7) * 3, 1)
            inter_box = patches.FancyBboxPatch((4.5, 3), 3, 1, boxstyle="round,pad=0.1",
                                               edgecolor='#6b7280', facecolor='#374151', linewidth=2, alpha=inter_alpha)
            ax.add_patch(inter_box)
            ax.text(6, 3.5, '56×56×128\n(unchanged)', ha='center', va='center', fontsize=9, 
                    color='white', alpha=inter_alpha)
        
        # Cost annotation
        if progress > 0.5:
            cost_alpha = min((progress - 0.5) * 2, 1)
            cost_box = patches.FancyBboxPatch((4.5, 4.3), 3, 0.6, boxstyle="round,pad=0.1",
                                              edgecolor='#f59e0b', facecolor='#1a1a2e', linewidth=2, 
                                              linestyle='--', alpha=cost_alpha)
            ax.add_patch(cost_box)
            ax.text(6, 4.6, 'FLOPs: 3.6M', ha='center', va='center', fontsize=10, 
                    color='#f59e0b', fontweight='bold', alpha=cost_alpha)
        
    else:  # pointwise
        # Pointwise convolution (1×1 channel mixing)
        pw_active = progress > 0.3
        pw_color = '#10b981' if pw_active else '#6b7280'
        
        # Show depthwise output first
        dw_out_box = patches.FancyBboxPatch((1, 3), 2, 1, boxstyle="round,pad=0.1",
                                            edgecolor='#6b7280', facecolor='#374151', linewidth=2)
        ax.add_patch(dw_out_box)
        ax.text(2, 3.5, '56×56\n×128', ha='center', va='center', fontsize=10, color='white')
        
        # Pointwise block
        pw_box = patches.FancyBboxPatch((4.5, 2.5), 3, 2, boxstyle="round,pad=0.1",
                                        edgecolor=pw_color, facecolor='#15803d', linewidth=3, alpha=0.7)
        ax.add_patch(pw_box)
        ax.text(6, 4.0, 'Pointwise 1×1', ha='center', va='top', fontsize=13, color='white', fontweight='bold')
        ax.text(6, 3.4, '256 filters', ha='center', va='center', fontsize=10, color='white')
        ax.text(6, 3.0, '(1×1×128 each)', ha='center', va='center', fontsize=9, color='#34d399')
        ax.text(6, 2.7, 'Mix channels!', ha='center', va='center', fontsize=9, color='#34d399', style='italic')
        
        # Arrow
        if progress > 0.2:
            ax.arrow(3.2, 3.5, 1, 0, head_width=0.3, head_length=0.2, fc='white', ec='white', linewidth=2)
        
        # Cost annotation
        if progress > 0.5:
            cost_alpha = min((progress - 0.5) * 2, 1)
            cost_box = patches.FancyBboxPatch((4.5, 1.5), 3, 0.6, boxstyle="round,pad=0.1",
                                              edgecolor='#10b981', facecolor='#1a1a2e', linewidth=2, 
                                              linestyle='--', alpha=cost_alpha)
            ax.add_patch(cost_box)
            ax.text(6, 1.8, 'FLOPs: 103M', ha='center', va='center', fontsize=10, 
                    color='#10b981', fontweight='bold', alpha=cost_alpha)
    
    # Output feature map
    if phase == 'standard':
        output_active = progress > 0.7
        out_y = 6
    elif phase == 'depthwise':
        output_active = False
        out_y = 6
    else:
        output_active = progress > 0.7
        out_y = 2.75
    
    if output_active or phase == 'standard':
        out_color = color if output_active else '#6b7280'
        output_box = patches.FancyBboxPatch((9, out_y), 2, 1.5, boxstyle="round,pad=0.1",
                                            edgecolor=out_color, facecolor='#1e3a8a', linewidth=2, alpha=0.8)
        ax.add_patch(output_box)
        ax.text(10, out_y + 0.75, '56×56\n×256', ha='center', va='center', fontsize=11, 
                color='white', fontweight='bold')
        
        if output_active:
            ax.arrow(7.7, out_y + 0.75, 1, 0, head_width=0.3, head_length=0.2, 
                     fc='white', ec='white', linewidth=2)
    
    plt.tight_layout()
    frame_path = f'../img/temp_frame_{frame_num:03d}.png'
    plt.savefig(frame_path, dpi=100, facecolor='#1a1a2e', bbox_inches='tight')
    plt.close()
    return frame_path

# Generate frames
print('Generating animation frames...')
total_frames = 30
frame_paths = []

for i in range(total_frames):
    frame_path = create_frame(i, total_frames)
    frame_paths.append(frame_path)
    print(f'  Frame {i+1}/{total_frames}')

# Create GIF
print('Creating GIF...')
frames = [Image.open(fp) for fp in frame_paths]

# Add pause at phase transitions
pause_indices = [9, 19, 29]
durations = [150 if i not in pause_indices else 500 for i in range(total_frames)]

frames[0].save(
    '../img/ch02-depthwise-separable-animation.gif',
    save_all=True,
    append_images=frames[1:],
    duration=durations,
    loop=0
)

# Clean up
for fp in frame_paths:
    os.remove(fp)

print('✅ Generated: ../img/ch02-depthwise-separable-animation.gif')
