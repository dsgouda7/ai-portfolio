"""
Generate temperature parameter comparison visualization.
Shows how temperature τ affects contrastive loss behavior.
"""

import numpy as np
import matplotlib.pyplot as plt

# Dark theme setup
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#1a1a2e'
plt.rcParams['axes.facecolor'] = '#1a1a2e'

# Cosine similarities (positive and negative samples)
similarities = np.linspace(-1, 1, 100)
temperatures = [0.05, 0.07, 0.1, 0.5]
temp_colors = ['#b91c1c', '#b45309', '#15803d', '#3b82f6']

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, (tau, color) in enumerate(zip(temperatures, temp_colors)):
    ax = axes[idx]
    
    # Compute exp(sim / tau)
    scores = np.exp(similarities / tau)
    
    # Plot
    ax.plot(similarities, scores, linewidth=3, color=color)
    ax.axvline(x=0, color='white', linestyle='--', alpha=0.3, linewidth=1)
    ax.axhline(y=1, color='white', linestyle='--', alpha=0.3, linewidth=1)
    
    # Highlight positive (high similarity) and negative (low similarity) regions
    ax.fill_between(similarities, 0, scores, where=(similarities > 0.5), 
                    alpha=0.3, color='#15803d', label='Positive pairs')
    ax.fill_between(similarities, 0, scores, where=(similarities < 0), 
                    alpha=0.3, color='#b91c1c', label='Negative pairs')
    
    # Annotations
    ax.set_title(f'Temperature τ = {tau}', fontsize=14, weight='bold', color=color)
    ax.set_xlabel('Cosine Similarity', fontsize=12)
    ax.set_ylabel('exp(sim / τ)', fontsize=12)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=10, loc='upper left')
    
    # Add interpretation text
    if tau == 0.05:
        interpretation = "Very sharp: huge difference between pos/neg\nFocuses on hard negatives\nRisk: numerical instability"
    elif tau == 0.07:
        interpretation = "SimCLR default: good balance\nSharp discrimination\nStable training"
    elif tau == 0.1:
        interpretation = "Moderate: softer discrimination\nEasier gradients\nFaster convergence"
    else:  # 0.5
        interpretation = "Too soft: weak discrimination\nEasy negatives dominate\nPoor final performance"
    
    ax.text(0.98, 0.95, interpretation, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e', alpha=0.9, edgecolor=color),
           color='white')

plt.suptitle('Temperature Parameter Effect on Contrastive Loss', 
            fontsize=18, weight='bold', y=0.995)
plt.tight_layout()
plt.savefig('../img/ch07-temperature-comparison.png', dpi=150, 
           bbox_inches='tight', facecolor='#1a1a2e')
print("Saved: ch07-temperature-comparison.png")
plt.close()
