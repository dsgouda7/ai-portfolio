"""
Generate visual aids for Ch.7 Probability & Statistics
- Venn diagram for set notation
- PMF vs PDF comparison
- Discrete to continuous mapping
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import FancyBboxPatch, Circle
from scipy import special  # For binomial coefficient

# Set style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 10


def binomial_pmf(k, n, p):
    """Manual binomial PMF calculation"""
    coef = special.comb(n, k, exact=True)
    return coef * (p ** k) * ((1 - p) ** (n - k))


def gaussian_pdf(x, mu, sigma):
    """Manual Gaussian PDF calculation"""
    return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def gaussian_cdf(x, mu, sigma):
    """Manual Gaussian CDF using erf"""
    return 0.5 * (1 + special.erf((x - mu) / (sigma * np.sqrt(2))))


def create_venn_diagram():
    """Venn diagram showing sample space Ω, events A and B, intersection, union"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Sample space Ω (rectangle)
    omega_box = FancyBboxPatch((0.5, 0.5), 9, 7, 
                               boxstyle="round,pad=0.1", 
                               linewidth=3, 
                               edgecolor='black', 
                               facecolor='lightgray', 
                               alpha=0.3)
    ax.add_patch(omega_box)
    ax.text(9.2, 7.3, r'$\Omega$ (Sample Space)', fontsize=14, weight='bold')
    ax.text(1, 1, 'All possible\nfree-kick outcomes', fontsize=10, style='italic', color='#555')
    
    # Event A (left circle)
    circle_a = Circle((3.5, 4), 1.8, color='#FF6B6B', alpha=0.4, linewidth=2, edgecolor='#C92A2A')
    ax.add_patch(circle_a)
    ax.text(2.2, 4, r'Event $A$', fontsize=13, weight='bold', color='#C92A2A')
    ax.text(2.0, 3.3, '"Ball goes\nin goal"', fontsize=9, color='#C92A2A')
    
    # Event B (right circle)
    circle_b = Circle((6.5, 4), 1.8, color='#4ECDC4', alpha=0.4, linewidth=2, edgecolor='#0B7285')
    ax.add_patch(circle_b)
    ax.text(6.8, 4, r'Event $B$', fontsize=13, weight='bold', color='#0B7285')
    ax.text(6.5, 3.3, '"Windy\nday"', fontsize=9, color='#0B7285')
    
    # Intersection A ∩ B
    ax.text(5.0, 4.3, r'$A \cap B$', fontsize=12, weight='bold', color='#495057', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', linewidth=1.5))
    ax.text(4.5, 3.7, 'Goal in\nwindy', fontsize=8, color='#495057')
    
    # Legend for set notation
    legend_y = 6.5
    ax.text(0.8, legend_y+0.5, 'Set Notation:', fontsize=11, weight='bold')
    ax.text(0.8, legend_y, r'$A \cup B$ = union (goal OR windy)', fontsize=9)
    ax.text(0.8, legend_y-0.4, r'$A \cap B$ = intersection (goal AND windy)', fontsize=9)
    ax.text(0.8, legend_y-0.8, r'$A^c$ = complement (NOT goal)', fontsize=9)
    ax.text(0.8, legend_y-1.2, r'$\mathbb{P}(A)$ = probability of event $A$', fontsize=9)
    
    # Example point (single outcome)
    ax.plot(7.5, 6, 'o', markersize=8, color='#F59F00', markeredgecolor='black', markeredgewidth=1.5)
    ax.text(7.7, 6, 'Single outcome ω', fontsize=9, style='italic')
    ax.text(7.7, 5.6, '(one specific kick)', fontsize=8, color='#555')
    
    # Complement of A (text annotation)
    ax.text(8, 2, r"$A^c$ (miss)", fontsize=10, color='#868E96', weight='bold')
    
    plt.title('Probability Events as Subsets of Sample Space', fontsize=15, weight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('img/ch07-venn-diagram.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Created ch07-venn-diagram.png")


def create_pmf_pdf_comparison():
    """Side-by-side comparison of discrete PMF and continuous PDF"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # LEFT: Discrete PMF (Binomial: n=10 kicks, p=0.25 success rate)
    n, p = 10, 0.25
    k_values = np.arange(0, n+1)
    pmf_values = np.array([binomial_pmf(k, n, p) for k in k_values])
    
    ax1.bar(k_values, pmf_values, width=0.7, color='#4C6EF5', edgecolor='#364FC7', linewidth=1.5, alpha=0.8)
    ax1.set_xlabel('Number of goals (k)', fontsize=12, weight='bold')
    ax1.set_ylabel('Probability P(X = k)', fontsize=12, weight='bold')
    ax1.set_title('DISCRETE: Probability Mass Function (PMF)', fontsize=13, weight='bold')
    ax1.set_xticks(k_values)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, max(pmf_values) * 1.15)
    
    # Annotations
    max_idx = np.argmax(pmf_values)
    ax1.annotate(f'Most likely: {max_idx} goals\nP(X={max_idx}) = {pmf_values[max_idx]:.3f}',
                xy=(max_idx, pmf_values[max_idx]), 
                xytext=(max_idx+2, pmf_values[max_idx]+0.05),
                fontsize=9, color='#1E3A8A',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#DBEAFE', edgecolor='#3B82F6', linewidth=1.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', color='#3B82F6', lw=2))
    
    ax1.text(1, 0.22, 'Discrete outcomes:\n{0, 1, 2, ..., 10}', fontsize=9, 
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF9DB', edgecolor='#F59F00', linewidth=1))
    ax1.text(1, 0.19, r'$\sum_{k=0}^{10} P(X=k) = 1$', fontsize=10, style='italic')
    
    # RIGHT: Continuous PDF (Gaussian: launch angle with μ=37.7°, σ=2.5°)
    mu, sigma = 37.7, 2.5
    x_values = np.linspace(mu - 4*sigma, mu + 4*sigma, 500)
    pdf_values = gaussian_pdf(x_values, mu, sigma)
    
    ax2.plot(x_values, pdf_values, color='#F06292', linewidth=3, label='PDF curve')
    ax2.fill_between(x_values, pdf_values, alpha=0.3, color='#FCE4EC')
    ax2.set_xlabel('Launch angle θ (degrees)', fontsize=12, weight='bold')
    ax2.set_ylabel('Probability density p(θ)', fontsize=12, weight='bold')
    ax2.set_title('CONTINUOUS: Probability Density Function (PDF)', fontsize=13, weight='bold')
    ax2.grid(axis='both', alpha=0.3, linestyle='--')
    
    # Shade region for P(36 < θ < 39)
    mask = (x_values >= 36) & (x_values <= 39)
    ax2.fill_between(x_values[mask], pdf_values[mask], alpha=0.6, color='#AD1457', label='P(36° < θ < 39°)')
    
    prob_shaded = gaussian_cdf(39, mu, sigma) - gaussian_cdf(36, mu, sigma)
    ax2.annotate(f'Area = {prob_shaded:.3f}\n(probability)',
                xy=(37.5, 0.08), xytext=(33, 0.12),
                fontsize=9, color='#6A1B9A',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#F3E5F5', edgecolor='#9C27B0', linewidth=1.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.3', color='#9C27B0', lw=2))
    
    ax2.text(31, 0.14, 'Continuous domain:\nθ ∈ ℝ (any real angle)', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF9DB', edgecolor='#F59F00', linewidth=1))
    ax2.text(31, 0.125, r'$\int_{-\infty}^{\infty} p(\theta) d\theta = 1$', fontsize=10, style='italic')
    ax2.text(31, 0.11, r'P(θ = 37.7°) = 0 (single point)', fontsize=8, color='#C62828')
    
    ax2.axvline(mu, color='#D32F2F', linestyle='--', linewidth=2, alpha=0.7, label=f'Mean μ = {mu}°')
    ax2.legend(loc='upper right', fontsize=9)
    
    plt.suptitle('Discrete PMF vs Continuous PDF', fontsize=15, weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('img/ch07-pmf-pdf-comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Created ch07-pmf-pdf-comparison.png")


def create_discrete_continuous_mapping():
    """Visual showing how discrete {0,1} outcomes map to probability [0,1]"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # TOP: Discrete Bernoulli - coin flips mapping to probabilities
    ax1.set_xlim(-0.5, 10.5)
    ax1.set_ylim(-0.5, 1.5)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('Discrete Outcomes → Probability Values', fontsize=14, weight='bold', pad=20)
    
    # Draw sample space (discrete outcomes)
    outcomes = [0, 1]
    for i, outcome in enumerate(outcomes):
        x_pos = 1 + i*2.5
        # Outcome box
        box = FancyBboxPatch((x_pos-0.4, 0.8), 0.8, 0.5, 
                            boxstyle="round,pad=0.05",
                            linewidth=2.5,
                            edgecolor='#2C3E50',
                            facecolor='#ECF0F1')
        ax1.add_patch(box)
        ax1.text(x_pos, 1.05, f'{outcome}', fontsize=20, weight='bold', ha='center', va='center')
        
        # Label
        label = 'Miss' if outcome == 0 else 'Goal'
        ax1.text(x_pos, 0.5, label, fontsize=11, ha='center', style='italic', color='#34495E')
    
    ax1.text(2.25, 1.35, 'Free-kick outcomes\n(discrete set)', fontsize=10, ha='center', 
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#D5DBDB', edgecolor='black'))
    
    # Arrow mapping to probability
    ax1.annotate('', xy=(6, 0.8), xytext=(4.5, 0.8),
                arrowprops=dict(arrowstyle='->', lw=3, color='#E74C3C'))
    ax1.text(5.25, 0.95, 'maps to', fontsize=11, ha='center', weight='bold', color='#C0392B')
    
    # Probability number line [0, 1]
    ax1.plot([6, 10], [0.8, 0.8], 'k-', linewidth=3)
    ax1.plot([6, 6], [0.75, 0.85], 'k-', linewidth=3)  # Left tick
    ax1.plot([10, 10], [0.75, 0.85], 'k-', linewidth=3)  # Right tick
    
    ax1.text(6, 0.55, '0', fontsize=12, ha='center', weight='bold')
    ax1.text(10, 0.55, '1', fontsize=12, ha='center', weight='bold')
    ax1.text(8, 0.55, '0.5', fontsize=12, ha='center', weight='bold')
    ax1.plot([8, 8], [0.75, 0.85], 'k-', linewidth=2)  # Middle tick
    
    # Probability values with p=0.25
    p = 0.25
    # P(X=0) = 0.75
    prob_0_x = 6 + (1-p) * 4
    ax1.plot(prob_0_x, 0.8, 'o', markersize=15, color='#3498DB', markeredgecolor='#2C3E50', markeredgewidth=2)
    ax1.text(prob_0_x, 0.4, f'P(X=0)\n= {1-p:.2f}', fontsize=10, ha='center', weight='bold', color='#2980B9')
    ax1.plot([prob_0_x, prob_0_x], [0.5, 0.75], 'k--', linewidth=1, alpha=0.5)
    
    # P(X=1) = 0.25
    prob_1_x = 6 + p * 4
    ax1.plot(prob_1_x, 0.8, 'o', markersize=15, color='#27AE60', markeredgecolor='#2C3E50', markeredgewidth=2)
    ax1.text(prob_1_x, 0.4, f'P(X=1)\n= {p:.2f}', fontsize=10, ha='center', weight='bold', color='#229954')
    ax1.plot([prob_1_x, prob_1_x], [0.5, 0.75], 'k--', linewidth=1, alpha=0.5)
    
    ax1.text(8, 0.2, r'Probability number line: $\mathbb{P}(\text{event}) \in [0, 1]$', 
             fontsize=11, ha='center', style='italic')
    ax1.text(8, 0.05, r'Constraint: $P(X=0) + P(X=1) = 1$', fontsize=10, ha='center', 
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FADBD8', edgecolor='#E74C3C', linewidth=1.5))
    
    # BOTTOM: Continuous case - launch angle
    ax2.set_xlim(-0.5, 10.5)
    ax2.set_ylim(-0.5, 1.5)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('Continuous Domain → Probability Density', fontsize=14, weight='bold', pad=20)
    
    # Continuous domain (angle range)
    domain_box = FancyBboxPatch((0.5, 0.7), 4, 0.6,
                               boxstyle="round,pad=0.05",
                               linewidth=2.5,
                               edgecolor='#8E44AD',
                               facecolor='#F4ECF7')
    ax2.add_patch(domain_box)
    ax2.text(2.5, 1.0, r'θ ∈ [20°, 50°]', fontsize=14, ha='center', weight='bold')
    ax2.text(2.5, 0.5, 'Launch angle\n(continuous)', fontsize=10, ha='center', style='italic', color='#6C3483')
    
    # Arrow
    ax2.annotate('', xy=(6, 0.8), xytext=(5, 0.8),
                arrowprops=dict(arrowstyle='->', lw=3, color='#E74C3C'))
    ax2.text(5.5, 0.95, 'density', fontsize=11, ha='center', weight='bold', color='#C0392B')
    
    # Mini PDF curve
    x_pdf = np.linspace(6, 10, 100)
    y_pdf_base = 0.5
    y_pdf = y_pdf_base + 0.3 * np.exp(-((x_pdf - 8)**2) / 0.5)
    ax2.plot(x_pdf, y_pdf, color='#E67E22', linewidth=3)
    ax2.fill_between(x_pdf, y_pdf_base, y_pdf, alpha=0.3, color='#F39C12')
    
    ax2.plot([6, 10], [y_pdf_base, y_pdf_base], 'k-', linewidth=2)
    ax2.text(8, 0.35, r'$p(\theta)$ - probability density', fontsize=11, ha='center', style='italic')
    
    # Highlight area under curve
    mask = (x_pdf >= 7.5) & (x_pdf <= 8.5)
    ax2.fill_between(x_pdf[mask], y_pdf_base, y_pdf[mask], alpha=0.6, color='#D35400')
    ax2.text(8, 0.85, 'Area = probability\nof angle range', fontsize=9, ha='center', weight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FEF5E7', edgecolor='#D68910', linewidth=1.5))
    
    ax2.text(8, 0.15, r'Single point: $P(\theta = 37.7°) = 0$', fontsize=10, ha='center', color='#922B21',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FADBD8', edgecolor='#C0392B', linewidth=1.5))
    ax2.text(8, 0.0, r'Interval: $P(36° < \theta < 39°) = \int_{36}^{39} p(\theta) d\theta$', 
             fontsize=9, ha='center', style='italic')
    
    plt.tight_layout()
    plt.savefig('img/ch07-discrete-continuous-mapping.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Created ch07-discrete-continuous-mapping.png")


if __name__ == "__main__":
    print("Generating Ch.7 Probability Visuals...")
    create_venn_diagram()
    create_pmf_pdf_comparison()
    create_discrete_continuous_mapping()
    print("\n✅ All Ch.7 visuals generated successfully!")
    print("Images saved to: notes/MathUnderTheHood/ch07-probability-statistics/img/")
