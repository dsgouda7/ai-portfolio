"""
Generate visual assets for Mutual Information section in ML Ch.3
Companion script to ml/01_regression/ch03_feature_importance/README.md

Creates:
1. ch03-mi-accumulation.gif - Shows MI building up from joint density
2. ch03-mi-entropy-bars.png - Entropy reduction diagram (H(Y), H(Y|X), I(X;Y))
3. Enhanced versions of existing MI images with better annotations

Run from project root: python scripts/generate_mi_visuals.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import gaussian_kde
from sklearn.feature_selection import mutual_info_regression
import seaborn as sns

# Dark theme matching other chapter visuals
plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#1a1a2e',
    'axes.edgecolor': '#e2e8f0',
    'axes.labelcolor': '#e2e8f0',
    'text.color': '#e2e8f0',
    'xtick.color': '#e2e8f0',
    'ytick.color': '#e2e8f0',
    'grid.color': '#4a5568',
    'grid.alpha': 0.3,
})


def generate_mi_accumulation_animation():
    """
    Animation showing MI calculation as weighted sum over scatter plot.
    Each cell contributes based on log(p(x,y) / (p(x)p(y))).
    """
    print("Generating ch03-mi-accumulation.gif...")
    
    # Create non-linear relationship (sigmoid)
    np.random.seed(42)
    n_samples = 500
    x = np.linspace(-3, 3, n_samples)
    y = 1 / (1 + np.exp(-2*x)) + np.random.normal(0, 0.1, n_samples)
    
    # Compute 2D histogram (discrete proxy for density)
    H, xedges, yedges = np.histogram2d(x, y, bins=20)
    H_norm = H / H.sum()  # Joint probability p(x,y)
    
    # Marginals
    px = H_norm.sum(axis=1, keepdims=True)
    py = H_norm.sum(axis=0, keepdims=True)
    
    # Independence baseline
    p_indep = px @ py
    
    # Log ratio
    with np.errstate(divide='ignore', invalid='ignore'):
        log_ratio = np.log(H_norm / p_indep)
        log_ratio[np.isinf(log_ratio)] = 0
        log_ratio[np.isnan(log_ratio)] = 0
    
    # Create animation
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5), facecolor='#1a1a2e')
    
    def update(frame):
        for ax in [ax1, ax2, ax3]:
            ax.clear()
        
        # Panel 1: Scatter with current cell highlighted
        ax1.scatter(x, y, c='#60a5fa', alpha=0.3, s=20, edgecolors='none')
        ax1.set_xlabel('Feature X', fontsize=11)
        ax1.set_ylabel('Target Y', fontsize=11)
        ax1.set_title('Joint Density p(x,y)', fontsize=12, pad=10)
        ax1.grid(True, alpha=0.2)
        
        # Panel 2: Log ratio heatmap (building up)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        n_cells = log_ratio.size
        reveal_cells = int((frame / 100) * n_cells)
        log_ratio_partial = log_ratio.copy()
        flat_indices = np.argsort(log_ratio.ravel())[::-1]  # Highest first
        mask = np.ones_like(log_ratio.ravel(), dtype=bool)
        mask[flat_indices[:reveal_cells]] = False
        log_ratio_partial.ravel()[mask] = np.nan
        
        im = ax2.imshow(log_ratio_partial.T, origin='lower', extent=extent,
                        cmap='RdBu_r', vmin=-2, vmax=2, aspect='auto')
        ax2.set_xlabel('Feature X', fontsize=11)
        ax2.set_ylabel('Target Y', fontsize=11)
        ax2.set_title(f'Log Ratio: log(p(x,y) / p(x)p(y)) [{reveal_cells}/{n_cells} cells]',
                      fontsize=12, pad=10)
        
        # Panel 3: Cumulative MI
        mi_values = (H_norm.ravel()[flat_indices[:reveal_cells]] * 
                     log_ratio.ravel()[flat_indices[:reveal_cells]])
        mi_cumsum = np.cumsum(mi_values)
        
        ax3.plot(range(len(mi_cumsum)), mi_cumsum, color='#34d399', linewidth=2)
        ax3.axhline(y=log_ratio[log_ratio > 0].sum(), color='#fbbf24', 
                    linestyle='--', label='Final MI', linewidth=1.5)
        ax3.set_xlabel('Cells Accumulated', fontsize=11)
        ax3.set_ylabel('Mutual Information (nats)', fontsize=11)
        ax3.set_title('MI Accumulation', fontsize=12, pad=10)
        ax3.legend(loc='lower right')
        ax3.grid(True, alpha=0.2)
        ax3.set_xlim(0, n_cells)
        ax3.set_ylim(0, 1.2)
        
        plt.tight_layout()
    
    anim = FuncAnimation(fig, update, frames=100, interval=50, repeat=True)
    anim.save('notes/01-ml/01_regression/ch03_feature_importance/img/ch03-mi-accumulation.gif',
              writer='pillow', fps=20, dpi=100)
    plt.close()
    print("✓ Saved ch03-mi-accumulation.gif")


def generate_entropy_bars():
    """
    Static diagram showing H(Y), H(Y|X), and I(X;Y) as bar comparison.
    """
    print("Generating ch03-mi-entropy-bars.png...")
    
    # Example entropy values (binary outcome, moderate MI)
    H_Y = 1.0      # Maximum entropy for binary (fair coin)
    H_Y_given_X = 0.3  # Conditional entropy after knowing X
    I_XY = H_Y - H_Y_given_X  # Mutual information
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1a1a2e')
    
    bar_width = 0.5
    positions = [0, 1.2, 2.4]
    
    # Bar 1: H(Y) - full uncertainty
    ax.bar(positions[0], H_Y, width=bar_width, color='#ef4444', 
           edgecolor='#e2e8f0', linewidth=2, label='H(Y) - Initial Uncertainty')
    ax.text(positions[0], H_Y + 0.05, f'{H_Y:.2f} bits', 
            ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Bar 2: H(Y|X) - residual uncertainty
    ax.bar(positions[1], H_Y_given_X, width=bar_width, color='#6b7280',
           edgecolor='#e2e8f0', linewidth=2, label='H(Y|X) - Residual Uncertainty')
    ax.text(positions[1], H_Y_given_X + 0.05, f'{H_Y_given_X:.2f} bits',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Bar 3: I(X;Y) - information gained (split bar to show it's the difference)
    ax.bar(positions[2], I_XY, width=bar_width, color='#34d399',
           edgecolor='#e2e8f0', linewidth=2, label='I(X;Y) - Information Gained')
    ax.bar(positions[2], H_Y_given_X, width=bar_width, bottom=I_XY, 
           color='#6b7280', edgecolor='#e2e8f0', linewidth=2, alpha=0.3)
    ax.text(positions[2], I_XY/2, f'{I_XY:.2f} bits\n(Information)', 
            ha='center', va='center', fontsize=10, fontweight='bold', color='#1a1a2e')
    ax.text(positions[2], I_XY + H_Y_given_X/2, 'Still\nuncertain',
            ha='center', va='center', fontsize=9, alpha=0.7)
    
    # Annotations
    ax.annotate('', xy=(positions[0], H_Y), xytext=(positions[1], H_Y),
                arrowprops=dict(arrowstyle='->', color='#fbbf24', lw=2))
    ax.text((positions[0] + positions[1])/2, H_Y + 0.1, 'Knowing X reduces uncertainty',
            ha='center', fontsize=10, color='#fbbf24')
    
    ax.set_ylabel('Entropy (bits)', fontsize=12, fontweight='bold')
    ax.set_title('Mutual Information as Entropy Reduction\nI(X;Y) = H(Y) - H(Y|X)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(positions)
    ax.set_xticklabels(['Before knowing X\n(H(Y))', 
                        'After knowing X\n(H(Y|X))',
                        'Decomposition\n(same total)'], fontsize=10)
    ax.set_ylim(0, 1.3)
    ax.grid(axis='y', alpha=0.2)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig('notes/01-ml/01_regression/ch03_feature_importance/img/ch03-mi-entropy-bars.png',
                dpi=150, facecolor='#1a1a2e', edgecolor='none')
    plt.close()
    print("✓ Saved ch03-mi-entropy-bars.png")


def generate_comparison_visualization():
    """
    Side-by-side comparison: Linear case (Pearson and MI agree) vs
    Non-linear case (MI high, Pearson low)
    """
    print("Generating ch03-mi-pearson-comparison.png...")
    
    np.random.seed(42)
    n = 300
    
    # Case 1: Linear
    x1 = np.linspace(-3, 3, n)
    y1 = 0.8 * x1 + np.random.normal(0, 0.5, n)
    
    # Case 2: U-shaped
    x2 = np.linspace(-3, 3, n)
    y2 = x2**2 + np.random.normal(0, 0.5, n)
    
    # Compute metrics
    from scipy.stats import pearsonr
    r1, _ = pearsonr(x1, y1)
    mi1 = mutual_info_regression(x1.reshape(-1, 1), y1, random_state=42)[0]
    r2, _ = pearsonr(x2, y2)
    mi2 = mutual_info_regression(x2.reshape(-1, 1), y2, random_state=42)[0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='#1a1a2e')
    
    # Linear case
    ax1.scatter(x1, y1, c='#60a5fa', alpha=0.5, s=30, edgecolors='none')
    ax1.plot(x1, 0.8*x1, color='#fbbf24', linewidth=2, linestyle='--', label='Linear fit')
    ax1.set_xlabel('Feature X', fontsize=11)
    ax1.set_ylabel('Target Y', fontsize=11)
    ax1.set_title(f'Linear Relationship\nPearson ρ = {r1:.3f}  |  MI = {mi1:.3f} nats\n✓ Both agree: strong signal',
                  fontsize=12, pad=10, color='#34d399')
    ax1.grid(True, alpha=0.2)
    ax1.legend()
    
    # U-shaped case
    ax2.scatter(x2, y2, c='#ec4899', alpha=0.5, s=30, edgecolors='none')
    ax2.plot(x2, x2**2, color='#fbbf24', linewidth=2, linestyle='--', label='True relationship')
    ax2.set_xlabel('Feature X', fontsize=11)
    ax2.set_ylabel('Target Y', fontsize=11)
    ax2.set_title(f'U-Shaped Relationship\nPearson ρ = {r2:.3f}  |  MI = {mi2:.3f} nats\n⚠️  Pearson misses it, MI catches it',
                  fontsize=12, pad=10, color='#fbbf24')
    ax2.grid(True, alpha=0.2)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('notes/01-ml/01_regression/ch03_feature_importance/img/ch03-mi-pearson-comparison.png',
                dpi=150, facecolor='#1a1a2e', edgecolor='none')
    plt.close()
    print("✓ Saved ch03-mi-pearson-comparison.png")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Generating Mutual Information Visual Assets")
    print("="*60 + "\n")
    
    generate_entropy_bars()
    generate_mi_accumulation_animation()
    generate_comparison_visualization()
    
    print("\n" + "="*60)
    print("✓ All MI visuals generated successfully!")
    print("="*60 + "\n")
    
    print("Generated files:")
    print("  • ch03-mi-entropy-bars.png - Entropy reduction diagram")
    print("  • ch03-mi-accumulation.gif - MI building from joint density")
    print("  • ch03-mi-pearson-comparison.png - Side-by-side linear vs U-shaped")
    print("\nExisting visuals to verify:")
    print("  • ch03-mi-joint-density.png")
    print("  • ch03-mi-case1-ushape.png")
    print("  • ch03-mi-case2-threshold.png")
    print("  • ch03-broken-ruler-parabola.png")
    print("  • ch03-mi-in-action.gif")
    print("  • ch03-pearson-mi-venn.gif")
    print("\n" + "="*60)
    print("Note: Run generate_feature_candidacy_flow.py separately")
    print("to create the decision flow animation (ch03-feature-candidacy-flow.gif)")
    print("="*60)
