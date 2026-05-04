"""
gen_logistic_vs_linear_weights.py

Generates a 2-panel PNG explaining the relationship between weights w and their
derivatives ∂L/∂w in Linear Regression vs Logistic Regression.

Output: ../img/logistic_vs_linear_weights.png
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------
OUT_DIR = Path(__file__).parent.parent / 'img'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / 'logistic_vs_linear_weights.png'

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
z = np.linspace(-6, 6, 300)

# Linear regression
y_hat_linear = z                          # identity: ŷ = z
grad_linear = np.ones_like(z)             # ∂ŷ/∂z = 1 everywhere (constant scale)

# Logistic regression
sigma = 1.0 / (1.0 + np.exp(-z))          # σ(z)
sigma_deriv = sigma * (1.0 - sigma)       # σ'(z) = σ(z)(1 − σ(z))

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor('#f9f9f9')

# ── Panel A: Linear Regression ──────────────────────────────────────────────
ax_left.set_facecolor('#f0f4ff')

ax_left.plot(z, y_hat_linear, color='#2563eb', linewidth=2.5, label=r'$\hat{y} = z$  (identity output)')
ax_left.plot(z, grad_linear,  color='#dc2626', linewidth=2.5, linestyle='--',
             label=r'$\partial\hat{y}/\partial z = 1$  (constant)')

ax_left.axhline(0, color='black', linewidth=0.6, linestyle=':')
ax_left.axvline(0, color='black', linewidth=0.6, linestyle=':')

ax_left.set_xlim(-6, 6)
ax_left.set_xlabel(r'Logit  $z = w \cdot x + b$', fontsize=11)
ax_left.set_ylabel('Value', fontsize=11)
ax_left.set_title('Linear Regression\n(Identity output, constant gradient scale)',
                  fontsize=12, fontweight='bold', pad=10)
ax_left.legend(fontsize=10, loc='upper left')

ax_left.annotate(
    r'Gradient $\partial\hat{y}/\partial z = 1$' + '\n(always constant)',
    xy=(0, 1.0), xytext=(1.5, 0.3),
    fontsize=9.5, color='#dc2626',
    arrowprops=dict(arrowstyle='->', color='#dc2626', lw=1.4),
    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#dc2626', alpha=0.85),
)

ax_left.text(
    0.03, 0.97,
    'Gradient update:\n'
    r'$w \leftarrow w - \alpha\,(\hat{y} - y)\cdot x$',
    transform=ax_left.transAxes,
    fontsize=9.5, va='top', ha='left',
    bbox=dict(boxstyle='round,pad=0.4', fc='#dbeafe', ec='#2563eb', alpha=0.9),
)

ax_left.yaxis.set_major_locator(ticker.MultipleLocator(2))
ax_left.grid(True, alpha=0.35)

# ── Panel B: Logistic Regression ────────────────────────────────────────────
ax_right.set_facecolor('#f0fff4')

color_sigma  = '#2563eb'   # blue  – σ(z)
color_deriv  = '#d97706'   # amber – σ'(z)
color_grad   = '#059669'   # green – simplified BCE gradient annotation

# Left y-axis: σ(z)
line_sigma, = ax_right.plot(z, sigma, color=color_sigma, linewidth=2.5,
                             label=r'$\hat{p} = \sigma(z)$')
ax_right.set_xlim(-6, 6)
ax_right.set_ylim(-0.05, 1.15)
ax_right.set_xlabel(r'Logit  $z = w \cdot x + b$', fontsize=11)
ax_right.set_ylabel(r'$\sigma(z)$', fontsize=11, color=color_sigma)
ax_right.tick_params(axis='y', labelcolor=color_sigma)
ax_right.set_title('Logistic Regression\n(Sigmoid output, gradient = ' + r'$\hat{p} - y$)',
                   fontsize=12, fontweight='bold', pad=10)

# Right y-axis: σ'(z)
ax_right2 = ax_right.twinx()
line_deriv, = ax_right2.plot(z, sigma_deriv, color=color_deriv, linewidth=2.5,
                              linestyle='--', label=r"$\sigma'(z) = \sigma(1-\sigma)$")
ax_right2.set_ylim(-0.01, 0.30)
ax_right2.set_ylabel(r"$\sigma'(z)$", fontsize=11, color=color_deriv)
ax_right2.tick_params(axis='y', labelcolor=color_deriv)
ax_right2.yaxis.set_major_locator(ticker.MultipleLocator(0.05))

# Reference lines
ax_right.axhline(0, color='black', linewidth=0.6, linestyle=':')
ax_right.axvline(0, color='black', linewidth=0.6, linestyle=':')
ax_right2.axhline(0.25, color=color_deriv, linewidth=0.8, linestyle=':', alpha=0.6)

# Annotate peak of σ'(z)
peak_idx = np.argmax(sigma_deriv)
ax_right2.annotate(
    r"$\sigma'(0) = 0.25$  (peak)",
    xy=(z[peak_idx], sigma_deriv[peak_idx]),
    xytext=(1.5, 0.20),
    fontsize=9.5, color=color_deriv,
    arrowprops=dict(arrowstyle='->', color=color_deriv, lw=1.4),
    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=color_deriv, alpha=0.85),
)

# Key insight annotation
ax_right.text(
    0.03, 0.97,
    'BCE + sigmoid cancels σ\'(z):\n'
    r'$\partial L/\partial z = \hat{p} - y$' + '\n'
    r'$w \leftarrow w - \alpha\,(\hat{p} - y)\cdot x$',
    transform=ax_right.transAxes,
    fontsize=9.5, va='top', ha='left',
    bbox=dict(boxstyle='round,pad=0.4', fc='#d1fae5', ec='#059669', alpha=0.9),
)

# Combined legend for both y-axes
lines = [line_sigma, line_deriv]
labels = [l.get_label() for l in lines]
ax_right.legend(lines, labels, fontsize=10, loc='center right')

ax_right.grid(True, alpha=0.35)

# ── Shared insight banner ────────────────────────────────────────────────────
fig.text(
    0.5, 0.01,
    'Key insight: despite different output functions, both models share the same gradient update form  '
    r'$w \leftarrow w - \alpha\,(\text{prediction} - \text{truth})\cdot x$',
    ha='center', va='bottom', fontsize=10, style='italic',
    color='#374151',
    bbox=dict(boxstyle='round,pad=0.4', fc='#fefce8', ec='#ca8a04', alpha=0.9),
)

plt.tight_layout(rect=[0, 0.07, 1, 1])

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
fig.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: img/logistic_vs_linear_weights.png")
