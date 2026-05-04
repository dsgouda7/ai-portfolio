"""
gen_ch07_alpha_sweep.py
Generates: ../img/ch07-alpha-sweep.png
MAE vs regularization strength α for Ridge (degree=2 pipeline) on California Housing.
Shows the sweet spot, Ch.5 baseline, and underfitting zones.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

# ── Data ────────────────────────────────────────────────────────────────
data = fetch_california_housing()
X, y = data.data, data.target

alphas = np.logspace(-3, 3, 60)
maes_mean = []
maes_std  = []

for alpha in alphas:
    pipe = Pipeline([
        ('poly',   PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('model',  Ridge(alpha=alpha))
    ])
    scores = cross_val_score(pipe, X, y, cv=5,
                             scoring='neg_mean_absolute_error', n_jobs=-1)
    maes_mean.append(-scores.mean() * 100_000)
    maes_std.append(scores.std() * 100_000)

maes_mean = np.array(maes_mean)
maes_std  = np.array(maes_std)

best_idx   = int(np.argmin(maes_mean))
best_alpha = alphas[best_idx]
best_mae   = maes_mean[best_idx]

# ── Plot ─────────────────────────────────────────────────────────────────
BG     = '#1a1a2e'
GRID_C = '#1e293b'
TEXT_C = '#e2e8f0'
BLUE   = '#38bdf8'
ORANGE = '#fb923c'
GREEN  = '#4ade80'
RED    = '#f87171'

fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
ax.set_facecolor(BG)

# Confidence band
ax.fill_between(alphas, maes_mean - maes_std, maes_mean + maes_std,
                alpha=0.25, color=BLUE, label='±1 std (5-fold)')

# MAE curve
ax.semilogx(alphas, maes_mean, color=BLUE, linewidth=2.5, label='CV MAE')

# Optimal α
ax.axvline(best_alpha, color=ORANGE, linestyle='--', linewidth=1.8,
           label=f'Optimal α ≈ {best_alpha:.3f} → ${best_mae/1000:.1f}k')
ax.annotate(f'Optimal α≈{best_alpha:.3f}\n→ ${best_mae/1000:.1f}k',
            xy=(best_alpha, best_mae), xytext=(best_alpha * 2.5, best_mae + 2500),
            color=ORANGE, fontsize=9,
            arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.2))

# Ch.5 baseline at α=1.0
ch5_mae = maes_mean[np.argmin(np.abs(alphas - 1.0))]
ax.axvline(1.0, color='#a78bfa', linestyle=':', linewidth=1.5,
           label=f'Ch.5 used α=1.0 → ${ch5_mae/1000:.1f}k')
ax.annotate(f'Ch.5\nα=1.0\n${ch5_mae/1000:.1f}k',
            xy=(1.0, ch5_mae), xytext=(3.0, ch5_mae + 1500),
            color='#a78bfa', fontsize=8.5,
            arrowprops=dict(arrowstyle='->', color='#a78bfa', lw=1))

# $40k constraint line
ax.axhline(40_000, color=GREEN, linestyle='--', linewidth=1.5, alpha=0.7,
           label='$40k SmartVal target')

# Underfit shade left/right
ax.axvspan(1e-3, 0.002,  color=RED, alpha=0.15, label='Underfits (α too small)')
ax.axvspan(200,  1e3,    color=RED, alpha=0.15, label='Underfits (all weights→0)')
ax.text(0.00105, maes_mean.max() * 0.97, 'Under-\nregularized',
        color=RED, fontsize=7.5, va='top')
ax.text(220, maes_mean.max() * 0.97, 'Over-\nregularized',
        color=RED, fontsize=7.5, va='top')

ax.set_xscale('log')
ax.set_xlabel('Regularization strength α', color=TEXT_C, fontsize=12)
ax.set_ylabel('5-fold CV MAE ($)', color=TEXT_C, fontsize=12)
ax.set_title("Ridge α Sweep — Why We Don't Guess the Regularization Strength",
             color=TEXT_C, fontsize=13, fontweight='bold')
ax.tick_params(colors=TEXT_C)
for sp in ax.spines.values():
    sp.set_edgecolor(GRID_C)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v/1000:.0f}k'))
ax.set_ylim(maes_mean.min() - 2000, maes_mean.max() + 5000)
ax.grid(True, which='both', color=GRID_C, linewidth=0.6, linestyle='--')
legend = ax.legend(fontsize=9, framealpha=0.3, facecolor=BG,
                   labelcolor=TEXT_C, loc='upper right')

fig.tight_layout()

out = Path(__file__).parent.parent / 'img' / 'ch07-alpha-sweep.png'
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
print(f'Saved: {out}')
