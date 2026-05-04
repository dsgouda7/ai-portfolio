#!/usr/bin/env python3
"""
Generate secrets lifecycle animation for Ch.8 Security & Secrets Management.

Animation: Create → Store → Access → Rotate → Revoke
Shows the full lifecycle of a production secret with security properties at each stage.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from PIL import Image
import os

# Output configuration
OUTPUT_DIR = "../img"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Animation configuration
STAGES = [
    {
        "name": "Create",
        "description": "Generate secret in secure store",
        "properties": ["Random generation", "Strong entropy", "Unique per service"],
        "color": "#4CAF50"
    },
    {
        "name": "Store",
        "description": "Encrypted at rest in Key Vault",
        "properties": ["AES-256 encryption", "RBAC access control", "Audit logging"],
        "color": "#2196F3"
    },
    {
        "name": "Access",
        "description": "Injected into container at runtime",
        "properties": ["Managed Identity", "No credentials in code", "Memory-only"],
        "color": "#FF9800"
    },
    {
        "name": "Rotate",
        "description": "Update without downtime",
        "properties": ["Dual-password window", "Rolling update", "Version history"],
        "color": "#9C27B0"
    },
    {
        "name": "Revoke",
        "description": "Remove access immediately",
        "properties": ["Instant deactivation", "Audit trail", "No image rebuild"],
        "color": "#F44336"
    }
]

def create_frame(stage_idx, total_stages):
    """Create a single frame showing the current stage in the lifecycle."""
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Title
    ax.text(50, 95, "Secrets Lifecycle", 
            ha='center', va='top', fontsize=24, fontweight='bold')
    
    # Draw all stages in a horizontal flow
    stage_width = 16
    stage_height = 12
    spacing = 3
    total_width = total_stages * stage_width + (total_stages - 1) * spacing
    start_x = (100 - total_width) / 2
    
    for i, stage in enumerate(STAGES):
        x = start_x + i * (stage_width + spacing)
        y = 55
        
        # Determine if this stage is active
        is_active = i <= stage_idx
        alpha = 1.0 if is_active else 0.3
        
        # Draw stage box
        color = stage["color"] if is_active else "#CCCCCC"
        box = FancyBboxPatch(
            (x, y), stage_width, stage_height,
            boxstyle="round,pad=0.5",
            facecolor=color,
            edgecolor='black' if is_active else '#999999',
            linewidth=2 if i == stage_idx else 1,
            alpha=alpha
        )
        ax.add_patch(box)
        
        # Stage name
        ax.text(x + stage_width/2, y + stage_height - 2, stage["name"],
                ha='center', va='center', fontsize=14, fontweight='bold',
                color='white', alpha=alpha)
        
        # Draw arrow to next stage
        if i < total_stages - 1:
            arrow_x = x + stage_width
            arrow = FancyArrowPatch(
                (arrow_x + 0.5, y + stage_height/2),
                (arrow_x + spacing - 0.5, y + stage_height/2),
                arrowstyle='->', mutation_scale=30,
                color='black' if is_active else '#CCCCCC',
                linewidth=2 if is_active else 1,
                alpha=alpha
            )
            ax.add_patch(arrow)
    
    # Current stage details box
    if stage_idx < len(STAGES):
        current_stage = STAGES[stage_idx]
        
        # Details box
        details_box = FancyBboxPatch(
            (10, 10), 80, 35,
            boxstyle="round,pad=1",
            facecolor='#F5F5F5',
            edgecolor=current_stage["color"],
            linewidth=3
        )
        ax.add_patch(details_box)
        
        # Stage description
        ax.text(50, 40, f"Stage {stage_idx + 1}/{total_stages}: {current_stage['name']}",
                ha='center', va='top', fontsize=18, fontweight='bold',
                color=current_stage["color"])
        
        ax.text(50, 36, current_stage["description"],
                ha='center', va='top', fontsize=14, color='#333333')
        
        # Security properties
        ax.text(50, 30, "Security Properties:",
                ha='center', va='top', fontsize=12, fontweight='bold',
                color='#666666')
        
        for i, prop in enumerate(current_stage["properties"]):
            y_pos = 26 - i * 4
            # Checkmark
            ax.text(15, y_pos, "✓", ha='left', va='center',
                    fontsize=14, color=current_stage["color"], fontweight='bold')
            # Property text
            ax.text(20, y_pos, prop, ha='left', va='center',
                    fontsize=11, color='#333333')
    
    plt.tight_layout()
    return fig

def generate_animation():
    """Generate PNG and GIF for the secrets lifecycle."""
    print("Generating secrets lifecycle animation...")
    
    frames = []
    
    # Generate frame for each stage
    for stage_idx in range(len(STAGES)):
        fig = create_frame(stage_idx, len(STAGES))
        
        # Save as temporary file
        temp_file = f"/tmp/ch08_lifecycle_frame_{stage_idx}.png"
        fig.savefig(temp_file, dpi=150, bbox_inches='tight', facecolor='white')
        frames.append(Image.open(temp_file))
        plt.close(fig)
        
        print(f"  Generated frame {stage_idx + 1}/{len(STAGES)}")
    
    # Add pause frames at the end
    for _ in range(3):
        frames.append(frames[-1].copy())
    
    # Save static PNG (final frame)
    output_png = os.path.join(OUTPUT_DIR, "ch08-secrets-lifecycle.png")
    frames[-1].save(output_png)
    print(f"✓ Saved: {output_png}")
    
    # Save animated GIF
    output_gif = os.path.join(OUTPUT_DIR, "ch08-secrets-lifecycle.gif")
    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=1500,  # 1.5 seconds per frame
        loop=0
    )
    print(f"✓ Saved: {output_gif}")
    
    # Clean up temp files
    for stage_idx in range(len(STAGES)):
        temp_file = f"/tmp/ch08_lifecycle_frame_{stage_idx}.png"
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == "__main__":
    generate_animation()
    print("\n✓ Secrets lifecycle animation complete!")
    print("  Files: ch08-secrets-lifecycle.png, ch08-secrets-lifecycle.gif")
