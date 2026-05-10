import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

z = np.linspace(-6, 6, 500)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

sigma = sigmoid(z)

# --- Loss values (y=1) ---
L_mse = (1 - sigma) ** 2
L_bce = -np.log(sigma)

# --- Gradients ---
# d(MSE)/dz = -2*(1 - sigma(z)) * sigma(z) * (1 - sigma(z))
grad_mse = -2 * (1 - sigma) * sigma * (1 - sigma)

# d(BCE)/dz = sigma(z) - 1
grad_bce = sigma - 1

# --- Output path ---
out_path = Path(__file__).parent.parent / 'img' / 'mse_vs_bce_panel.png'
out_path.parent.mkdir(parents=True, exist_ok=True)

# --- Figure ---
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor('#1a1a2e')

PANEL_BG = '#16213e'
BLUE_LOSS = '#4fc3f7'
BLUE_GRAD = '#7986cb'
RED_LOSS = '#ef5350'
RED_GRAD = '#ff8a65'
TEXT_COLOR = '#e0e0e0'
ANNOT_COLOR = '#ffd54f'

for ax in (ax_a, ax_b):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444466')
    ax.tick_params(colors=TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)

# ── Panel A: MSE ────────────────────────────────────────────────────────────
ax_a.plot(z, L_mse, color=BLUE_LOSS, linewidth=2.2, label=r'$L_{\mathrm{MSE}} = (1 - \hat{p})^2$')
ax_a.plot(z, grad_mse, color=BLUE_GRAD, linewidth=1.5, linestyle='--',
          label=r'$\frac{dL}{dz} = -2(1-\hat{p})\hat{p}(1-\hat{p})$')

ax_a.set_title('Panel A — MSE Loss Landscape', fontsize=12, pad=10)
ax_a.set_xlabel('z  (logit)', fontsize=11)
ax_a.set_ylabel('Loss / Gradient', fontsize=11)
ax_a.set_xlim(-6, 6)
ax_a.set_ylim(-0.15, 1.15)

# Annotate flat plateau on left (z < -3)
ax_a.annotate(
    'gradient ≈ 0\n(vanishing)',
    xy=(-4.5, L_mse[np.argmin(np.abs(z - (-4.5)))]),
    xytext=(-5.8, 0.6),
    arrowprops=dict(arrowstyle='->', color=ANNOT_COLOR, lw=1.4),
    color=ANNOT_COLOR, fontsize=9,
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', edgecolor=ANNOT_COLOR, alpha=0.85),
)

# Annotate flat plateau on right (z > 3)
ax_a.annotate(
    'gradient ≈ 0\n(vanishing)',
    xy=(4.5, L_mse[np.argmin(np.abs(z - 4.5))]),
    xytext=(2.8, 0.5),
    arrowprops=dict(arrowstyle='->', color=ANNOT_COLOR, lw=1.4),
    color=ANNOT_COLOR, fontsize=9,
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', edgecolor=ANNOT_COLOR, alpha=0.85),
)

# Shade plateau regions
ax_a.axvspan(-6, -3, alpha=0.12, color=BLUE_LOSS, label='flat (non-convex) plateau')
ax_a.axvspan(3, 6, alpha=0.12, color=BLUE_LOSS)

ax_a.legend(fontsize=8.5, loc='upper right',
            facecolor='#1a1a2e', edgecolor='#444466', labelcolor=TEXT_COLOR)
ax_a.grid(alpha=0.15, color='#aaaaaa')

# ── Panel B: BCE ────────────────────────────────────────────────────────────
ax_b.plot(z, L_bce, color=RED_LOSS, linewidth=2.2, label=r'$L_{\mathrm{BCE}} = -\log(\hat{p})$')
ax_b.plot(z, grad_bce, color=RED_GRAD, linewidth=1.5, linestyle='--',
          label=r'$\frac{dL}{dz} = \hat{p} - y = \hat{p} - 1$')

ax_b.set_title('Panel B — BCE Loss Landscape', fontsize=12, pad=10)
ax_b.set_xlabel('z  (logit)', fontsize=11)
ax_b.set_ylabel('Loss / Gradient', fontsize=11)
ax_b.set_xlim(-6, 6)
ax_b.set_ylim(-1.2, 6.5)

# Annotate steep region (z < -2)
ax_b.annotate(
    'large penalty for\nconfident wrong prediction',
    xy=(-4.0, L_bce[np.argmin(np.abs(z - (-4.0)))]),
    xytext=(-2.0, 4.5),
    arrowprops=dict(arrowstyle='->', color=ANNOT_COLOR, lw=1.4),
    color=ANNOT_COLOR, fontsize=9,
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', edgecolor=ANNOT_COLOR, alpha=0.85),
)

# Shade steep region
ax_b.axvspan(-6, -2, alpha=0.12, color=RED_LOSS, label='steep penalty region')

ax_b.legend(fontsize=8.5, loc='upper right',
            facecolor='#1a1a2e', edgecolor='#444466', labelcolor=TEXT_COLOR)
ax_b.grid(alpha=0.15, color='#aaaaaa')

# ── Save ─────────────────────────────────────────────────────────────────────
plt.tight_layout(pad=2.0)
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close(fig)

print("Saved: img/mse_vs_bce_panel.png")
