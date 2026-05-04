"""
Generate KS test results visualization (p-values for all features)

This script creates a bar chart showing which features have drifted between
California training data and Portland production data.

Output: ../img/ch03-ks-test-results.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from scipy.stats import ks_2samp
import sys
from pathlib import Path

# Add parent directory to path for imports if needed
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent.parent))

def generate_ks_test_results_plot():
    """Generate KS test results bar chart."""
    
    # Load California Housing dataset
    print("Loading California Housing dataset...")
    data = fetch_california_housing()
    df_california = pd.DataFrame(data.data, columns=data.feature_names)
    df_california['MedHouseVal'] = data.target
    
    # Simulate Portland data with 37% income shift
    print("Simulating Portland distribution shift...")
    df_portland = df_california.copy()
    df_portland['MedInc'] = df_portland['MedInc'] * 1.37  # 37% increase
    
    # Run KS tests on all features
    print("Running KS tests...")
    features_to_test = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                       'Population', 'AveOccup', 'Latitude', 'Longitude']
    
    drift_results = {}
    
    for feature in features_to_test:
        statistic, p_value = ks_2samp(df_california[feature], df_portland[feature])
        drift_results[feature] = {
            'ks_statistic': statistic,
            'p_value': p_value,
            'drifted': p_value < 0.05
        }
        status = "DRIFTED" if p_value < 0.05 else "OK"
        print(f"{feature:15s} | p-value: {p_value:.6f} | {status}")
    
    # Create plot
    print("Generating visualization...")
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    # Extract data for plotting
    features = list(drift_results.keys())
    p_values = [drift_results[f]['p_value'] for f in features]
    colors = ['#b91c1c' if p < 0.05 else '#15803d' for p in p_values]
    
    # Create bar chart
    bars = ax.bar(features, p_values, color=colors, edgecolor='white', linewidth=1.5)
    
    # Add threshold line at p=0.05
    ax.axhline(0.05, color='#f59e0b', linestyle='--', linewidth=2, 
               label='p=0.05 threshold', zorder=10)
    
    # Styling
    ax.set_xlabel('Features', color='white', fontsize=13, fontweight='bold')
    ax.set_ylabel('KS Test p-value', color='white', fontsize=13, fontweight='bold')
    ax.set_title('Distribution Shift Detection: KS Test p-values\\n(Red = Drift Detected, Green = No Drift)', 
                 color='white', fontsize=15, fontweight='bold', pad=20)
    ax.tick_params(colors='white', labelsize=10)
    ax.set_ylim(0, max(0.1, max(p_values) * 1.1))  # Focus on low p-values
    plt.xticks(rotation=45, ha='right')
    ax.legend(facecolor='#2d2d44', edgecolor='white', labelcolor='white', fontsize=11)
    ax.grid(axis='y', alpha=0.2, linestyle='--')
    
    plt.tight_layout()
    
    # Save
    output_path = Path(__file__).parent.parent / 'img' / 'ch03-ks-test-results.png'
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
    
    generate_ks_test_results_plot()
    print("✅ KS test results visualization generated successfully")
