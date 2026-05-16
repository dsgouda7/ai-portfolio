#!/usr/bin/env python3
"""
Generate pattern needle movement animation for Chapter 11 — Advanced Agentic Patterns

Creates an animated speedometer showing error rate reduction as patterns are applied.
Shows progression from 8% error rate (baseline) to 0.7% (target met) across 5 frames.

Output: ../img/ch11-pattern-needle-movement.gif
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
from PIL import Image


def ease_in_out(t):
    """Ease-in-out interpolation function"""
    return t * t * (3.0 - 2.0 * t)


def draw_speedometer_frame(error_rate, pattern_label, is_target_met=False):
    """Draw a single frame of the speedometer"""
    
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    # Speedometer parameters
    center_x, center_y = 5.0, 3.0
    radius = 2.5
    
    # Define gauge zones (error rate ranges and colors)
    zones = [
        (8.0, 5.0, '#e74c3c'),    # Red zone: 8-5%
        (5.0, 3.0, '#f39c12'),    # Yellow zone: 5-3%
        (3.0, 1.5, '#a3e6a3'),    # Light green zone: 3-1.5%
        (1.5, 0.5, '#27ae60'),    # Green zone: 1.5-0.5%
        (0.5, 0.0, '#0d7a3d'),    # Deep green zone: <0.5%
    ]
    
    # Draw gauge arc zones
    start_angle = 180  # Start at left (180°)
    end_angle = 0      # End at right (0°)
    total_range = 8.0  # 0-8% error rate
    
    for i, (zone_max, zone_min, color) in enumerate(zones):
        # Calculate arc angles for this zone
        zone_start_angle = start_angle - (zone_max / total_range) * 180
        zone_end_angle = start_angle - (zone_min / total_range) * 180
        
        # Draw arc
        arc = patches.Wedge((center_x, center_y), radius, 
                           zone_start_angle, zone_end_angle,
                           width=0.4, facecolor=color, edgecolor='white',
                           linewidth=2)
        ax.add_patch(arc)
    
    # Draw zone labels
    zone_labels = ['8%', '5%', '3%', '1.5%', '0.5%', '0%']
    for i, label in enumerate(zone_labels):
        angle_deg = start_angle - (float(label.rstrip('%')) / total_range) * 180
        angle_rad = np.radians(angle_deg)
        label_x = center_x + (radius + 0.6) * np.cos(angle_rad)
        label_y = center_y + (radius + 0.6) * np.sin(angle_rad)
        ax.text(label_x, label_y, label, fontsize=9, color='white',
                ha='center', va='center', fontweight='bold')
    
    # Draw target line at 1% (dashed white)
    target_rate = 1.0
    target_angle_deg = start_angle - (target_rate / total_range) * 180
    target_angle_rad = np.radians(target_angle_deg)
    target_x1 = center_x + (radius - 0.5) * np.cos(target_angle_rad)
    target_y1 = center_y + (radius - 0.5) * np.sin(target_angle_rad)
    target_x2 = center_x + (radius + 0.3) * np.cos(target_angle_rad)
    target_y2 = center_y + (radius + 0.3) * np.sin(target_angle_rad)
    ax.plot([target_x1, target_x2], [target_y1, target_y2], 
            'w--', linewidth=2, alpha=0.8)
    ax.text(target_x2 + 0.3, target_y2, 'Target', fontsize=8, color='white',
            ha='left', va='center', style='italic')
    
    # Draw needle
    needle_angle_deg = start_angle - (error_rate / total_range) * 180
    needle_angle_rad = np.radians(needle_angle_deg)
    needle_length = radius - 0.3
    needle_x = center_x + needle_length * np.cos(needle_angle_rad)
    needle_y = center_y + needle_length * np.sin(needle_angle_rad)
    
    # Needle shadow for depth
    shadow_offset = 0.05
    ax.plot([center_x + shadow_offset, needle_x + shadow_offset], 
            [center_y - shadow_offset, needle_y - shadow_offset],
            'k-', linewidth=4, alpha=0.3)
    
    # Needle main line
    ax.plot([center_x, needle_x], [center_y, needle_y],
            'w-', linewidth=3, solid_capstyle='round')
    
    # Needle center dot
    center_dot = patches.Circle((center_x, center_y), 0.15, 
                           facecolor='white', edgecolor='#333', linewidth=2)
    ax.add_patch(center_dot)
    
    # Large number display
    number_color = '#4a9eff' if is_target_met else 'white'
    ax.text(center_x, center_y - 1.2, f"{error_rate:.1f}%",
            fontsize=32, fontweight='bold', color=number_color,
            ha='center', va='center')
    
    ax.text(center_x, center_y - 1.6, "Error Rate",
            fontsize=12, color='#aaa', ha='center', va='center')
    
    # Pattern label
    if pattern_label:
        label_box = patches.FancyBboxPatch((center_x - 2.5, 5.5), 5.0, 0.8,
                                          boxstyle="round,pad=0.1",
                                          linewidth=2, edgecolor='#4a9eff',
                                          facecolor='#2d2d44', alpha=0.9)
        ax.add_patch(label_box)
        
        ax.text(center_x, 5.9, pattern_label,
                fontsize=14, fontweight='bold', color='white',
                ha='center', va='center')
    
    # Target met checkmark
    if is_target_met:
        ax.text(center_x + 3.2, center_y - 1.2, "✅",
                fontsize=24, ha='center', va='center')
    
    # Title
    ax.text(center_x, 6.8, "Pattern Evolution: Error Rate Reduction",
            fontsize=14, fontweight='bold', color='white',
            ha='center', va='center')
    
    # Set axis limits and remove axes
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    return fig


def create_needle_movement_animation():
    """Generate animated speedometer GIF"""
    
    # Define stages with error rates
    stages = [
        (8.0, "PizzaBot v1.0: 8% error rate", False),
        (4.0, "+ Reflection: 4% error rate", False),
        (2.0, "+ Debate: 2% error rate", False),
        (1.0, "+ Hierarchical: 1% error rate", False),
        (0.7, "PizzaBot v2.0: 0.7% error rate", True),
    ]
    
    frames = []
    temp_dir = Path(__file__).parent / "temp_frames"
    temp_dir.mkdir(exist_ok=True)
    
    # Generate frames with smooth interpolation
    frame_idx = 0
    for i in range(len(stages)):
        current_rate, current_label, is_target = stages[i]
        
        # Hold on each stage for a bit
        hold_frames = 8 if i < len(stages) - 1 else 12  # Hold longer on final frame
        
        for _ in range(hold_frames):
            fig = draw_speedometer_frame(current_rate, current_label, is_target)
            
            # Save frame
            frame_path = temp_dir / f"frame_{frame_idx:04d}.png"
            plt.savefig(frame_path, dpi=100, bbox_inches='tight',
                       facecolor='#1a1a2e', edgecolor='none')
            plt.close(fig)
            
            frames.append(Image.open(frame_path))
            frame_idx += 1
        
        # Interpolate to next stage (if not last)
        if i < len(stages) - 1:
            next_rate, next_label, next_target = stages[i + 1]
            transition_frames = 10  # Smooth transition
            
            for t in range(transition_frames):
                # Ease-in-out interpolation
                alpha = ease_in_out(t / transition_frames)
                interpolated_rate = current_rate + alpha * (next_rate - current_rate)
                
                fig = draw_speedometer_frame(interpolated_rate, current_label, False)
                
                frame_path = temp_dir / f"frame_{frame_idx:04d}.png"
                plt.savefig(frame_path, dpi=100, bbox_inches='tight',
                           facecolor='#1a1a2e', edgecolor='none')
                plt.close(fig)
                
                frames.append(Image.open(frame_path))
                frame_idx += 1
    
    # Save as GIF
    output_path = Path(__file__).parent.parent / "img" / "ch11-pattern-needle-movement.gif"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=50,  # 50ms per frame (~20 FPS)
        loop=0  # Infinite loop
    )
    
    # Cleanup temp frames
    import shutil
    shutil.rmtree(temp_dir)
    
    print(f"Generated {output_path}")
    print(f"  Total frames: {len(frames)}")
    print(f"  Duration: ~{len(frames) * 0.05:.1f}s")


if __name__ == "__main__":
    create_needle_movement_animation()
