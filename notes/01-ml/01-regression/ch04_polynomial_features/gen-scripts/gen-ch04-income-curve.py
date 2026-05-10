#!/usr/bin/env python3
"""
Generate income curve scatter plot showing the plateau effect.
Replaces ASCII art diagram.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent
IMG_DIR = SCRIPT_DIR.parent / "img"
IMG_DIR.mkdir(exist_ok=True)

def create_income_curve():
    """Create scatter plot showing income-value curve with plateau."""
    
    # Generate synthetic data matching California Housing pattern
    np.random.seed(42)
    
    # Create income values
    x = np.linspace(0.5, 15, 200)
    
    # True relationship: saturating curve (logistic-like)
    y_true = 5.0 / (1 + np.exp(-0.8 * (x - 5))) + 0.2
    
    # Add realistic scatter
    noise = np.random.normal(0, 0.3, len(x))
    y_data = y_true + noise
    y_data = np.clip(y_data, 0.5, 5.5)  # Cap at $500k
    
    # Linear fit for comparison
    z = np.polyfit(x, y_data, 1)
    p = np.poly1d(z)
    y_linear = p(x)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot
    ax.scatter(x, y_data, alpha=0.4, s=20, color='#1e3a8a', label='California Housing data')
    
    # True curve
    ax.plot(x, y_true, 'g-', linewidth=2.5, label='True relationship (saturating)', zorder=3)
    
    # Linear fit (dashed)
    ax.plot(x, y_linear, 'r--', linewidth=2, label='Linear fit (misses curve)', zorder=2)
    
    # Annotate plateau
    ax.annotate('Plateau\n(capped at $500k)',
                xy=(13, 5.0), xytext=(10.5, 5.5),
                arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2),
                fontsize=12, color='darkgreen', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # Annotate underprediction
    ax.annotate('Linear fit\nmisses at both ends',
                xy=(3, 1.5), xytext=(0.8, 2.5),
                arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5),
                fontsize=10, color='darkred')
    
    ax.set_xlabel('MedInc ($10k)', fontsize=14, fontweight='bold')
    ax.set_ylabel('MedHouseVal ($100k)', fontsize=14, fontweight='bold')
    ax.set_title('Income-Value Curve: Why We Need Polynomial Features', 
                 fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6)
    
    plt.tight_layout()
    
    # Save
    output_path = IMG_DIR / "ch04-income-curve.png"
    print(f"Saving income curve to {output_path}...")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    
    plt.close()

if __name__ == "__main__":
    print("Generating Ch.4 Income Curve...")
    create_income_curve()
    print("Done!")
