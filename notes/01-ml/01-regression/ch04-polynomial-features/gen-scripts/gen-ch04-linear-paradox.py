#!/usr/bin/env python3
"""
Generate "Linear Paradox" visualization showing how linear addition
of polynomial terms creates non-linear curves.
Replaces ASCII box diagram.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent
IMG_DIR = SCRIPT_DIR.parent / "img"
IMG_DIR.mkdir(exist_ok=True)

def create_linear_paradox():
    """Create diagram showing linear math → non-linear result."""
    
    fig = plt.figure(figsize=(12, 8))
    
    # Create two subplots: top for math, bottom for result
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 1], hspace=0.3)
    ax_math = fig.add_subplot(gs[0])
    ax_result = fig.add_subplot(gs[1])
    
    # === TOP PANEL: Math operation ===
    ax_math.set_xlim(0, 10)
    ax_math.set_ylim(0, 10)
    ax_math.axis('off')
    
    # Title
    ax_math.text(5, 9.5, 'MATH OPERATION (Linear)', 
                ha='center', va='top', fontsize=16, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Equation
    ax_math.text(5, 7.5, r'$\hat{y} = w_1 \cdot x + w_2 \cdot x^2 + w_3 \cdot x^3$',
                ha='center', va='center', fontsize=18, fontweight='bold')
    
    # Three terms with boxes
    terms = [
        (2, 5.5, r'$w_1 \cdot x$', '#3b82f6'),
        (5, 5.5, r'$w_2 \cdot x^2$', '#ef4444'),
        (8, 5.5, r'$w_3 \cdot x^3$', '#8b5cf6')
    ]
    
    for x, y, text, color in terms:
        # Box around each term
        box = FancyBboxPatch((x-0.7, y-0.4), 1.4, 0.8,
                            boxstyle="round,pad=0.1",
                            edgecolor=color, facecolor=color,
                            alpha=0.3, linewidth=2)
        ax_math.add_patch(box)
        ax_math.text(x, y, text, ha='center', va='center',
                    fontsize=14, fontweight='bold', color=color)
    
    # Arrows converging to center
    arrow_props = dict(arrowstyle='->', lw=2.5, color='black')
    ax_math.annotate('', xy=(5, 3.5), xytext=(2.5, 5),
                    arrowprops=arrow_props)
    ax_math.annotate('', xy=(5, 3.5), xytext=(5, 5),
                    arrowprops=arrow_props)
    ax_math.annotate('', xy=(5, 3.5), xytext=(7.5, 5),
                    arrowprops=arrow_props)
    
    # "JUST ADDITION" box
    ax_math.text(5, 3, 'JUST ADDITION\n(Linear algebra)',
                ha='center', va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Big arrow down
    arrow_down = FancyArrowPatch((5, 1.8), (5, 0.5),
                                arrowstyle='->', mutation_scale=30,
                                lw=4, color='darkgreen')
    ax_math.add_patch(arrow_down)
    
    # === BOTTOM PANEL: Result curve ===
    ax_result.text(0.5, 0.95, 'RESULTING SHAPE (Non-linear)',
                  transform=ax_result.transAxes,
                  ha='center', va='top', fontsize=16, fontweight='bold',
                  bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Generate curve
    x = np.linspace(-2, 2, 200)
    y = 0.5*x + 0.3*x**2 + 0.1*x**3  # Combined polynomial
    
    ax_result.plot(x, y, 'b-', linewidth=3, label='Combined result')
    ax_result.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
    ax_result.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
    
    # Annotate the curve
    ax_result.annotate('A CURVE!',
                      xy=(1.5, ax_result.get_ylim()[1]*0.7),
                      fontsize=20, fontweight='bold', color='darkblue',
                      bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax_result.set_xlabel('x', fontsize=14, fontweight='bold')
    ax_result.set_ylabel('y', fontsize=14, fontweight='bold')
    ax_result.grid(True, alpha=0.3)
    ax_result.set_title('The math is linear, the result is curved!',
                       fontsize=12, style='italic')
    
    plt.suptitle('The "Linear" Paradox', fontsize=18, fontweight='bold', y=0.98)
    
    # Save
    output_path = IMG_DIR / "ch04-linear-paradox.png"
    print(f"Saving linear paradox diagram to {output_path}...")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    
    plt.close()

if __name__ == "__main__":
    print("Generating Ch.4 Linear Paradox Diagram...")
    create_linear_paradox()
    print("Done!")
