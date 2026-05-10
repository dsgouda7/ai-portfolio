"""
gen_ch03_mi_joint_density.py
Three-panel conceptual diagram:
  Left:   Actual joint density p(x,y) — tight diagonal band
  Middle: Independent baseline p(x)·p(y) — spread blob
  Right:  log(p(x,y)/p(x)p(y)) heatmap — where the relationship lives
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
from matplotlib.colors import TwoSlopeNorm

DARK_BG  = "#1a1a2e"
TEXT_CLR = "#e2e8f0"

np.random.seed(42)
n = 500

# Simulate income → price: clear positive relationship
x = np.random.gamma(3, 1.2, n)
x = np.clip(x, 0.5, 10)
noise = np.random.normal(0, 0.5, n)
y = 0.45 * x + 0.4 + noise
y = np.clip(y, 0.3, 5.5)

# Shuffle y to build the independence baseline
rng = np.random.default_rng(7)
y_ind = rng.permutation(y)

# Grid for density estimation
x_grid = np.linspace(x.min(), x.max(), 60)
y_grid = np.linspace(y.min(), y.max(), 60)
XX, YY = np.meshgrid(x_grid, y_grid)
flat   = np.vstack([XX.ravel(), YY.ravel()])

# Actual joint density
kde_joint = gaussian_kde(np.vstack([x, y]))
pxy       = kde_joint(flat).reshape(XX.shape)

# Marginal densities → independent product
kde_x  = gaussian_kde(x)
kde_y  = gaussian_kde(y)
px     = kde_x(x_grid)          # shape (60,)
py     = kde_y(y_grid)          # shape (60,)
pxpy   = np.outer(py, px)       # shape (60,60), same grid as pxy

pxy_safe  = np.clip(pxy,  1e-12, None)
pxpy_safe = np.clip(pxpy, 1e-12, None)
log_ratio = np.log(pxy_safe / pxpy_safe)

# ── Figure ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 4.8), facecolor=DARK_BG)
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.45)

def style_ax(ax, title):
    ax.set_facecolor(DARK_BG)
    ax.set_title(title, color=TEXT_CLR, fontsize=10, fontweight="bold", pad=8)
    ax.tick_params(colors=TEXT_CLR, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333355")
    ax.set_xlabel("Income (x)", color=TEXT_CLR, fontsize=8)
    ax.set_ylabel("House Value (y)", color=TEXT_CLR, fontsize=8)

# ── Panel 1: actual p(x,y) ─────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
style_ax(ax1, "Actual  $p(x,\\,y)$")
ax1.scatter(x, y, s=9, alpha=0.28, color="#4ecdc4", lw=0, zorder=2)
qs = np.percentile(kde_joint(np.vstack([x, y])), [35, 62, 84])
ax1.contour(XX, YY, pxy, levels=qs, colors=["#a29bfe"], linewidths=1.3, alpha=0.85)
ax1.text(0.05, 0.95,
         "Narrow diagonal band\n→ income predicts value",
         transform=ax1.transAxes, color="#a29bfe", fontsize=8,
         va="top", linespacing=1.5)
ax1.text(0.50, 0.08, "sand piles up along\nthe relationship",
         transform=ax1.transAxes, color="#e2e8f0", fontsize=7.5,
         ha="center", style="italic", alpha=0.75)

# ── Panel 2: independent p(x)·p(y) ────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
style_ax(ax2, "Independent  $p(x)\\cdot p(y)$\n(y shuffled — no signal)")
ax2.scatter(x, y_ind, s=9, alpha=0.28, color="#f7b731", lw=0, zorder=2)
# Use the same analytical outer product as the Panel 3 denominator — no separate 2D KDE
qs_ind = np.percentile(pxpy, [35, 62, 84])
ax2.contour(XX, YY, pxpy, levels=qs_ind, colors=["#f7b731"],
            linewidths=1.3, alpha=0.65)
ax2.text(0.05, 0.95,
         "Spread blob — any income\npairs with any value",
         transform=ax2.transAxes, color="#f7b731", fontsize=8,
         va="top", linespacing=1.5)
ax2.text(0.50, 0.08, "sand spreads uniformly\n— no pattern",
         transform=ax2.transAxes, color="#e2e8f0", fontsize=7.5,
         ha="center", style="italic", alpha=0.75)

# ── Panel 3: log ratio heatmap ─────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2])
style_ax(ax3, "$\\log\\,\\dfrac{p(x,y)}{p(x)\\,p(y)}$  — the MI signal")
vmax = min(np.percentile(np.abs(log_ratio), 97), 4.0)
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
im   = ax3.pcolormesh(XX, YY, log_ratio, cmap="RdBu_r", norm=norm, shading="auto")
cbar = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
cbar.ax.tick_params(colors=TEXT_CLR, labelsize=7)
cbar.set_label("log ratio", color=TEXT_CLR, fontsize=7)
ax3.text(0.04, 0.95,
         "Red  = pair more common\n         than chance → signal\nBlue = pair less common\n         → avoided",
         transform=ax3.transAxes, color=TEXT_CLR, fontsize=7.5,
         va="top", linespacing=1.5)

# Annotation arrow pointing to the hot diagonal
ax3.annotate("", xy=(0.62, 0.68), xytext=(0.82, 0.50),
             xycoords="axes fraction",
             arrowprops=dict(arrowstyle="->", color="#e94560", lw=1.5))
ax3.text(0.84, 0.48, "MI\nhere", transform=ax3.transAxes,
         color="#e94560", fontsize=7.5, ha="center", va="top",
         fontweight="bold")

fig.suptitle("Joint Density vs Independence — Where the Relationship Lives",
             color=TEXT_CLR, fontsize=11, fontweight="bold", y=1.02)

out = "../img/ch03-mi-joint-density.png"
plt.savefig(out, dpi=130, facecolor=DARK_BG, bbox_inches="tight")
plt.close()
print(f"Saved {out}")
