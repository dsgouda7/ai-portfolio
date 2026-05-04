#!/usr/bin/env python3
"""
Generate animation showing how weighted polynomial terms combine.
Shows w1·x, w2·x², w3·x³ and their sum in a single panel.

Addresses requirement: "map w1x, w2x^2, w3x^3 and the final sum"
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent
IMG_DIR = SCRIPT_DIR.parent / "img"
IMG_DIR.mkdir(exist_ok=True)

def create_polynomial_combination_animation():
    """Create animation showing how polynomial terms combine."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # x values for plotting
    x = np.linspace(-2, 2, 200)
    
    # Initialize lines
    line_x, = ax.plot([], [], 'b-', linewidth=2, label=r'$w_1 \cdot x$', alpha=0.7)
    line_x2, = ax.plot([], [], 'r-', linewidth=2, label=r'$w_2 \cdot x^2$', alpha=0.7)
    line_x3, = ax.plot([], [], 'g-', linewidth=2, label=r'$w_3 \cdot x^3$', alpha=0.7)
    line_sum, = ax.plot([], [], 'k-', linewidth=4, label=r'$\hat{y}$ (sum)', zorder=3)
    
    # Setup axes
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-3, 3)
    ax.set_xlabel('x', fontsize=14, fontweight='bold')
    ax.set_ylabel('y', fontsize=14, fontweight='bold')
    ax.set_title('How Polynomial Terms Combine: The Volume Knob Effect',
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.legend(loc='upper left', fontsize=12)
    
    # Text annotations
    weight_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                         fontsize=13, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    status_text = ax.text(0.98, 0.98, '', transform=ax.transAxes,
                         fontsize=11, verticalalignment='top',
                         horizontalalignment='right',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Animation frames with different weight combinations
    # Start simple, then add complexity
    frames_data = []
    
    # Phase 1: Just linear (10 frames)
    for i in range(10):
        w1 = 0.5
        w2 = 0.0
        w3 = 0.0
        frames_data.append((w1, w2, w3, "Phase 1: Linear only"))
    
    # Phase 2: Add quadratic (20 frames)
    for i in range(20):
        w1 = 0.5
        w2 = 0.3 * (i / 20)
        w3 = 0.0
        frames_data.append((w1, w2, w3, "Phase 2: Adding curvature"))
    
    # Phase 3: Add cubic (20 frames)
    for i in range(20):
        w1 = 0.5
        w2 = 0.3
        w3 = 0.15 * (i / 20)
        frames_data.append((w1, w2, w3, "Phase 3: Adding S-curve"))
    
    # Phase 4: Adjust all (30 frames) - demonstrate volume knob adjustments
    for i in range(30):
        t = i / 30
        w1 = 0.5 - 0.3 * t  # Decrease linear
        w2 = 0.3 + 0.2 * t  # Increase quadratic
        w3 = 0.15 - 0.05 * t  # Slightly decrease cubic
        frames_data.append((w1, w2, w3, "Phase 4: Fine-tuning all knobs"))
    
    # Phase 5: Hold final (10 frames)
    for i in range(10):
        w1 = 0.2
        w2 = 0.5
        w3 = 0.1
        frames_data.append((w1, w2, w3, "Final: Optimal combination"))
    
    def init():
        line_x.set_data([], [])
        line_x2.set_data([], [])
        line_x3.set_data([], [])
        line_sum.set_data([], [])
        weight_text.set_text('')
        status_text.set_text('')
        return line_x, line_x2, line_x3, line_sum, weight_text, status_text
    
    def animate(frame):
        w1, w2, w3, phase = frames_data[frame]
        
        # Calculate each term
        y_x = w1 * x
        y_x2 = w2 * x**2
        y_x3 = w3 * x**3
        y_sum = y_x + y_x2 + y_x3
        
        # Update lines
        line_x.set_data(x, y_x)
        line_x2.set_data(x, y_x2)
        line_x3.set_data(x, y_x3)
        line_sum.set_data(x, y_sum)
        
        # Update text
        weight_text.set_text(
            f'Weights:\n'
            f'  w₁ = {w1:+.2f} {"🔊" if abs(w1) > 0.3 else "🔇" if abs(w1) < 0.1 else "🎚️"}\n'
            f'  w₂ = {w2:+.2f} {"🔊" if abs(w2) > 0.3 else "🔇" if abs(w2) < 0.1 else "🎚️"}\n'
            f'  w₃ = {w3:+.2f} {"🔊" if abs(w3) > 0.3 else "🔇" if abs(w3) < 0.1 else "🎚️"}'
        )
        
        status_text.set_text(phase)
        
        return line_x, line_x2, line_x3, line_sum, weight_text, status_text
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=len(frames_data),
                                  interval=100, blit=True, repeat=True)
    
    # Save as GIF
    output_path = IMG_DIR / "ch04-polynomial-combination.gif"
    print(f"Saving polynomial combination animation to {output_path}...")
    anim.save(output_path, writer='pillow', fps=10, dpi=100)
    print(f"✅ Saved: {output_path}")
    
    plt.close()

if __name__ == "__main__":
    print("Generating Ch.4 Polynomial Combination Animation...")
    create_polynomial_combination_animation()
    print("Done!")
