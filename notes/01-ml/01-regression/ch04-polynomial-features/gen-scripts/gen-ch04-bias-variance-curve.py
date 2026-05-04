#!/usr/bin/env python3
"""
Generate bias-variance U-curve visualization.
Replaces ASCII chart.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent
IMG_DIR = SCRIPT_DIR.parent / "img"
IMG_DIR.mkdir(exist_ok=True)

def create_bias_variance_curve():
    """Create U-shaped bias-variance curve."""
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Data points (degree vs MAE)
    degrees = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    
    # Train MAE (always decreasing)
    train_mae = np.array([70, 55, 42, 30, 18, 10, 6, 4])
    
    # Test MAE (U-shaped)
    test_mae = np.array([70, 55, 48, 50, 55, 62, 70, 80])
    
    # Plot both curves
    ax.plot(degrees, train_mae, 'o-', linewidth=3, markersize=10,
           color='#3b82f6', label='Train MAE (always ↓)', zorder=3)
    
    ax.plot(degrees, test_mae, 's-', linewidth=3, markersize=10,
           color='#ef4444', label='Test MAE (U-shaped)', zorder=3)
    
    # Target line
    ax.axhline(y=40, color='green', linestyle='--', linewidth=2,
              label='$40k Target', zorder=2)
    ax.text(8.2, 40, 'TARGET', va='center', fontsize=11,
           fontweight='bold', color='green')
    
    # Highlight sweet spot (degree 2)
    ax.scatter([2], [48], s=400, facecolors='none',
              edgecolors='gold', linewidths=4, zorder=4)
    ax.annotate('Sweet Spot\n(degree=2)',
               xy=(2, 48), xytext=(3, 58),
               arrowprops=dict(arrowstyle='->', color='gold', lw=3),
               fontsize=13, fontweight='bold', color='gold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # Region labels
    ax.text(1, 5, 'Underfitting', ha='center', fontsize=12,
           fontweight='bold', color='#b91c1c',
           bbox=dict(boxstyle='round', facecolor='#fee2e2', alpha=0.7))
    
    ax.text(2.5, 5, 'Sweet\nspot', ha='center', fontsize=12,
           fontweight='bold', color='#15803d',
           bbox=dict(boxstyle='round', facecolor='#dcfce7', alpha=0.7))
    
    ax.text(6, 5, 'Overfitting', ha='center', fontsize=12,
           fontweight='bold', color='#b91c1c',
           bbox=dict(boxstyle='round', facecolor='#fee2e2', alpha=0.7))
    
    # Annotate points
    for deg, train, test in [(1, 70, 70), (4, 30, 50), (8, 4, 80)]:
        ax.annotate(f'${train}k', xy=(deg, train),
                   xytext=(deg, train-5), ha='center',
                   fontsize=9, color='#1e40af')
        ax.annotate(f'${test}k', xy=(deg, test),
                   xytext=(deg, test+5), ha='center',
                   fontsize=9, color='#991b1b')
    
    ax.set_xlabel('Polynomial Degree', fontsize=14, fontweight='bold')
    ax.set_ylabel('MAE ($k)', fontsize=14, fontweight='bold')
    ax.set_title('Bias-Variance Trade-off: Finding the Sweet Spot',
                fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 8.5)
    ax.set_ylim(0, 85)
    ax.set_xticks(degrees)
    
    plt.tight_layout()
    
    # Save
    output_path = IMG_DIR / "ch04-bias-variance-curve.png"
    print(f"Saving bias-variance curve to {output_path}...")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    
    plt.close()

if __name__ == "__main__":
    print("Generating Ch.4 Bias-Variance Curve...")
    create_bias_variance_curve()
    print("Done!")
