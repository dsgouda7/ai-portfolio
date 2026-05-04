#!/usr/bin/env python3
"""
Generate feature expansion visualization showing 8 → 44 features.
Replaces ASCII box diagram.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent
IMG_DIR = SCRIPT_DIR.parent / "img"
IMG_DIR.mkdir(exist_ok=True)

def create_feature_expansion():
    """Create diagram showing polynomial feature expansion."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Polynomial Feature Expansion (degree=2)',
           ha='center', va='top', fontsize=18, fontweight='bold')
    
    # === TOP: Raw features (8) ===
    raw_y = 7.5
    raw_features = ['MedInc', 'HsAge', 'Rooms', 'Bedrm', 'Pop', 'Occup', 'Lat', 'Long']
    
    # Draw boxes for raw features
    box_width = 1.0
    start_x = 0.5
    
    for i, feat in enumerate(raw_features):
        x = start_x + i * box_width
        rect = mpatches.FancyBboxPatch((x, raw_y-0.3), box_width*0.9, 0.6,
                                       boxstyle="round,pad=0.05",
                                       edgecolor='#1e3a8a', facecolor='#93c5fd',
                                       linewidth=2)
        ax.add_patch(rect)
        ax.text(x + box_width*0.45, raw_y, feat,
               ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Label
    ax.text(start_x - 0.3, raw_y, 'd=8', ha='right', va='center',
           fontsize=12, fontweight='bold', color='#1e3a8a')
    
    # === MIDDLE: Transformation ===
    mid_y = 5.5
    
    # Big arrow down
    ax.annotate('', xy=(5, mid_y + 0.5), xytext=(5, raw_y - 0.5),
               arrowprops=dict(arrowstyle='->', lw=4, color='darkgreen'))
    
    # Process box
    process_box = mpatches.FancyBboxPatch((3, mid_y), 4, 0.8,
                                         boxstyle="round,pad=0.1",
                                         edgecolor='darkgreen', facecolor='lightgreen',
                                         linewidth=2, alpha=0.8)
    ax.add_patch(process_box)
    ax.text(5, mid_y + 0.4, 'PolynomialFeatures(degree=2)',
           ha='center', va='center', fontsize=12, fontweight='bold')
    
    # === BOTTOM: Expanded features (44) ===
    expanded_y = 2.5
    
    # Big arrow down
    ax.annotate('', xy=(5, expanded_y + 1.2), xytext=(5, mid_y - 0.2),
               arrowprops=dict(arrowstyle='->', lw=4, color='darkgreen'))
    
    # Label
    ax.text(9.7, expanded_y + 0.6, 'D=44', ha='right', va='center',
           fontsize=14, fontweight='bold', color='#b91c1c')
    
    # Three groups of features
    groups = [
        ('Original 8', raw_features, 0.5, '#93c5fd'),
        ('Squared 8', [f'{f}²' for f in raw_features[:8]], 0.4, '#fca5a5'),
        ('Interactions 28', ['MedInc×HsAge', 'MedInc×Rooms', '...', 'Lat×Long'], 2.2, '#c4b5fd')
    ]
    
    y_offset = expanded_y + 0.6
    
    for group_name, features, box_height, color in groups:
        # Group box
        group_box = mpatches.FancyBboxPatch((0.3, y_offset - box_height - 0.1),
                                           9.4, box_height,
                                           boxstyle="round,pad=0.05",
                                           edgecolor=color.replace('fd', '44'),
                                           facecolor=color,
                                           linewidth=2, alpha=0.6)
        ax.add_patch(group_box)
        
        # Group label
        ax.text(0.5, y_offset - box_height/2 - 0.1, group_name + ':',
               ha='left', va='center', fontsize=11, fontweight='bold')
        
        # Feature names
        feature_text = ', '.join(features[:4])
        if len(features) > 4:
            feature_text += ', ...'
        ax.text(2.5, y_offset - box_height/2 - 0.1, feature_text,
               ha='left', va='center', fontsize=9)
        
        y_offset -= (box_height + 0.15)
    
    # Bottom summary
    ax.text(5, 0.3, '8 original + 8 squared + 28 interactions = 44 polynomial features',
           ha='center', va='center', fontsize=12, style='italic',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    
    # Save
    output_path = IMG_DIR / "ch04-feature-expansion-diagram.png"
    print(f"Saving feature expansion diagram to {output_path}...")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    
    plt.close()

if __name__ == "__main__":
    print("Generating Ch.4 Feature Expansion Diagram...")
    create_feature_expansion()
    print("Done!")
