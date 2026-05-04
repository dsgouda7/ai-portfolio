"""
Generate distribution shift visualization (California vs Portland MedInc)

This script creates the main figure showing how Portland's income distribution
differs from California's training data — the key evidence of drift.

Output: ../img/ch03-distribution-shift.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
import sys
from pathlib import Path

# Add parent directory to path for imports if needed
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent.parent))

def generate_distribution_shift_plot():
    """Generate distribution shift visualization."""
    
    # Load California Housing dataset
    print("Loading California Housing dataset...")
    data = fetch_california_housing()
    df_california = pd.DataFrame(data.data, columns=data.feature_names)
    df_california['MedHouseVal'] = data.target
    
    # Simulate Portland data with 37% income shift
    print("Simulating Portland distribution shift...")
    df_portland = df_california.copy()
    df_portland['MedInc'] = df_portland['MedInc'] * 1.37  # 37% increase
    
    print(f"California MedInc mean: {df_california['MedInc'].mean():.2f}")
    print(f"Portland MedInc mean: {df_portland['MedInc'].mean():.2f}")
    print(f"Shift: {((df_portland['MedInc'].mean() / df_california['MedInc'].mean()) - 1) * 100:.1f}%")
    
    # Create plot
    print("Generating visualization...")
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    # Plot overlaid histograms
    ax.hist(df_california['MedInc'], bins=50, alpha=0.6, 
            label='California (Training)', color='#3b82f6', 
            edgecolor='white', linewidth=0.5)
    ax.hist(df_portland['MedInc'], bins=50, alpha=0.6, 
            label='Portland (Production)', color='#f59e0b', 
            edgecolor='white', linewidth=0.5)
    
    # Mark means with vertical lines
    ca_mean = df_california['MedInc'].mean()
    pdx_mean = df_portland['MedInc'].mean()
    ax.axvline(ca_mean, color='#3b82f6', linestyle='--', linewidth=2.5, 
               label=f'CA Mean: {ca_mean:.2f}')
    ax.axvline(pdx_mean, color='#f59e0b', linestyle='--', linewidth=2.5, 
               label=f'PDX Mean: {pdx_mean:.2f}')
    
    # Styling
    ax.set_xlabel('Median Income (tens of thousands $)', color='white', 
                  fontsize=13, fontweight='bold')
    ax.set_ylabel('Frequency', color='white', fontsize=13, fontweight='bold')
    ax.set_title('Distribution Shift: California → Portland\\nIncome Distribution (Training vs Production)', 
                 color='white', fontsize=15, fontweight='bold', pad=20)
    ax.legend(facecolor='#2d2d44', edgecolor='white', labelcolor='white', 
              fontsize=11, loc='upper right')
    ax.tick_params(colors='white', labelsize=11)
    ax.grid(alpha=0.2, linestyle='--')
    
    plt.tight_layout()
    
    # Save
    output_path = Path(__file__).parent.parent / 'img' / 'ch03-distribution-shift.png'
    plt.savefig(output_path, dpi=150, facecolor='#1a1a2e', bbox_inches='tight')
    print(f"✅ Saved to {output_path}")
    
    plt.close()

if __name__ == '__main__':
    # Set dark background style
    plt.style.use('dark_background')
    plt.rcParams['figure.facecolor'] = '#1a1a2e'
    plt.rcParams['axes.facecolor'] = '#1a1a2e'
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    generate_distribution_shift_plot()
    print("✅ Distribution shift visualization generated successfully")
