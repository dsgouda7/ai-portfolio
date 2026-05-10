"""
Ridge vs Lasso Comparison Animation
Side-by-side showing Ridge smoothly shrinking vs Lasso creating hard zeros.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import matplotlib.animation as animation
from pathlib import Path

# Get absolute path to img directory
script_dir = Path(__file__).parent
img_dir = script_dir.parent / 'img'
img_dir.mkdir(exist_ok=True)
np.random.seed(42)

# 4 features for clarity (too many clutters the comparison)
features = [
    'MedInc',
    'Latitude', 
    'AveBedrms²',
    'Pop × AveOccup'
]

# Initial weights
ols_weights = np.array([0.68, -0.42, -0.18, 0.21])

# Lambda values
lambdas = np.logspace(-3, 1.5, 90)

# Ridge path (smooth shrinkage)
importance_ridge = np.array([1.0, 0.7, 0.2, 0.15])
ridge_paths = []
for lam in lambdas:
    weights = ols_weights / (1 + lam / importance_ridge)
    ridge_paths.append(weights)
ridge_paths = np.array(ridge_paths)

# Lasso path (hard zeros)
zero_thresholds = np.array([10.0, 1.5, 0.08, 0.06])
lasso_paths = []
for lam in lambdas:
    weights = []
    for w_ols, thresh in zip(ols_weights, zero_thresholds):
        if lam >= thresh:
            weights.append(0.0)
        else:
            shrinkage = 1 - (lam / thresh) ** 0.7
            weights.append(w_ols * shrinkage)
    lasso_paths.append(weights)
lasso_paths = np.array(lasso_paths)

# Create figure
fig = plt.figure(figsize=(15, 7))
fig.suptitle('Ridge (L2) vs Lasso (L1): How Weights Shrink', 
             fontsize=15, fontweight='bold', y=0.96)

# Create 2x2 grid: top row = trajectories, bottom row = bar charts
gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], hspace=0.3, wspace=0.3)

ax_ridge_traj = fig.add_subplot(gs[0, 0])
ax_lasso_traj = fig.add_subplot(gs[0, 1])
ax_ridge_bars = fig.add_subplot(gs[1, 0])
ax_lasso_bars = fig.add_subplot(gs[1, 1])

colors = plt.cm.tab10(np.linspace(0, 0.6, len(features)))

# ===== Ridge trajectory =====
ridge_lines = []
for i, (feat, color) in enumerate(zip(features, colors)):
    line, = ax_ridge_traj.plot([], [], '-', linewidth=2.5, color=color, 
                                label=feat, alpha=0.85)
    ridge_lines.append(line)

ax_ridge_traj.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.4)
ax_ridge_traj.set_xlim(np.log10(lambdas[0]), np.log10(lambdas[-1]))
ax_ridge_traj.set_ylim(-0.75, 0.75)
ax_ridge_traj.set_xlabel('log₁₀(λ)', fontsize=11)
ax_ridge_traj.set_ylabel('Weight', fontsize=11)
ax_ridge_traj.set_title('Ridge (L2): Smooth Shrinkage\n(Never Reaches Zero)', 
                         fontsize=12, fontweight='bold', color='#1f77b4')
ax_ridge_traj.legend(loc='upper right', fontsize=9)
ax_ridge_traj.grid(True, alpha=0.3)

# ===== Lasso trajectory =====
lasso_lines = []
for i, (feat, color) in enumerate(zip(features, colors)):
    line, = ax_lasso_traj.plot([], [], '-', linewidth=2.5, color=color, 
                                label=feat, alpha=0.85)
    lasso_lines.append(line)

ax_lasso_traj.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.4)
ax_lasso_traj.set_xlim(np.log10(lambdas[0]), np.log10(lambdas[-1]))
ax_lasso_traj.set_ylim(-0.75, 0.75)
ax_lasso_traj.set_xlabel('log₁₀(λ)', fontsize=11)
ax_lasso_traj.set_ylabel('Weight', fontsize=11)
ax_lasso_traj.set_title('Lasso (L1): Hard Zeros\n(Features Drop Out)', 
                         fontsize=12, fontweight='bold', color='#ff7f0e')
ax_lasso_traj.legend(loc='upper right', fontsize=9)
ax_lasso_traj.grid(True, alpha=0.3)

# ===== Ridge bars =====
ridge_bars = ax_ridge_bars.barh(features, ols_weights, color=colors, 
                                 alpha=0.7, edgecolor='black', linewidth=0.5)
ax_ridge_bars.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.4)
ax_ridge_bars.set_xlim(-0.75, 0.75)
ax_ridge_bars.set_xlabel('Weight Value', fontsize=11)
ax_ridge_bars.set_title('Ridge Weights', fontsize=11)
ax_ridge_bars.grid(True, alpha=0.3, axis='x')

# ===== Lasso bars =====
lasso_bars = ax_lasso_bars.barh(features, ols_weights, color=colors, 
                                 alpha=0.7, edgecolor='black', linewidth=0.5)
ax_lasso_bars.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.4)
ax_lasso_bars.set_xlim(-0.75, 0.75)
ax_lasso_bars.set_xlabel('Weight Value', fontsize=11)
ax_lasso_bars.set_title('Lasso Weights', fontsize=11)
ax_lasso_bars.grid(True, alpha=0.3, axis='x')

# Lasso zero markers
lasso_zero_texts = [ax_lasso_bars.text(0, i, '', ha='center', va='center', 
                                        fontsize=9, fontweight='bold', color='red')
                    for i in range(len(features))]

# Lambda indicator
lambda_text = fig.text(0.5, 0.01, 'λ = 0.001', ha='center', fontsize=11, 
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5))

def init():
    for line in ridge_lines + lasso_lines:
        line.set_data([], [])
    return ridge_lines + lasso_lines + [lambda_text]

def update(frame):
    lam = lambdas[frame]
    log_lam = np.log10(lam)
    
    # Update Ridge trajectory
    for i, line in enumerate(ridge_lines):
        line.set_data(np.log10(lambdas[:frame+1]), ridge_paths[:frame+1, i])
    
    # Update Lasso trajectory
    for i, line in enumerate(lasso_lines):
        line.set_data(np.log10(lambdas[:frame+1]), lasso_paths[:frame+1, i])
    
    # Update Ridge bars (all shrink, none zero)
    ridge_weights = ridge_paths[frame]
    for bar, w, color in zip(ridge_bars, ridge_weights, colors):
        bar.set_width(w)
        bar.set_color(color)
        bar.set_alpha(0.7 if abs(w) > 0.05 else 0.4)
    
    # Update Lasso bars (some zero out)
    lasso_weights = lasso_paths[frame]
    for i, (bar, w, color, zero_text) in enumerate(zip(lasso_bars, lasso_weights, 
                                                        colors, lasso_zero_texts)):
        bar.set_width(w)
        
        if abs(w) < 1e-6:
            bar.set_color('lightgray')
            bar.set_alpha(0.3)
            zero_text.set_text('✗')
            zero_text.set_visible(True)
        else:
            bar.set_color(color)
            bar.set_alpha(0.7)
            zero_text.set_visible(False)
    
    # Count active features
    n_ridge = np.sum(np.abs(ridge_weights) > 0.01)
    n_lasso = np.sum(np.abs(lasso_weights) > 1e-6)
    
    lambda_text.set_text(f'λ = {lam:.3f}  |  Ridge: {n_ridge}/4 active  |  Lasso: {n_lasso}/4 active')
    
    return (ridge_lines + lasso_lines + list(ridge_bars) + list(lasso_bars) + 
            lasso_zero_texts + [lambda_text])

# Create animation
anim = animation.FuncAnimation(fig, update, frames=len(lambdas), 
                               init_func=init, interval=70, blit=False)

# Save
plt.tight_layout(rect=[0, 0.03, 1, 0.94])
output_path = img_dir / 'ch05-ridge-vs-lasso.gif'
anim.save(str(output_path), writer=PillowWriter(fps=14))
print(f"✅ Saved: {output_path}")
plt.close()
