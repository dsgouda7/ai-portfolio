"""gen_ch06_metrics_journey.py
Generates img/ch06-metrics-journey.png
Three-subplot convergence chart: MAE, RMSE, R² across Ch.1 → Ch.6.
Hard-coded journey data — no sklearn calls required.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── Journey data ──────────────────────────────────────────────────────────
chapters = [
    'Ch.1\n(1 feat)',
    'Ch.2\n(8 feat)',
    'Ch.3\n(VIF)',
    'Ch.4\n(poly)',
    'Ch.5\n(Ridge)',
    'Ch.6\n(CV)',
]
x = np.arange(len(chapters))

mae_k  = [70, 55, 55, 48, 38, 38]
rmse_k = [88, 71, 71, 63, 52, 52]
r2     = [0.47, 0.61, 0.61, 0.67, 0.68, 0.68]

mae_err  = [0, 0, 0, 0, 0, 2]
rmse_err = [0, 0, 0, 0, 0, 3]
r2_err   = [0, 0, 0, 0, 0, 0.01]

colors = ['#ef4444', '#f97316', '#f97316', '#f97316', '#22c55e', '#22c55e']

# ── Style constants ───────────────────────────────────────────────────────
BG_DARK  = '#1a1a2e'
BG_AXES  = '#16213e'
TEXT_COL = '#e2e8f0'
GRID_COL = '#374151'


def _style_ax(ax):
    ax.set_facecolor(BG_AXES)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    ax.title.set_color(TEXT_COL)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.grid(axis='y', color=GRID_COL, alpha=0.3, linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(chapters, color=TEXT_COL, fontsize=7)


def _plot_metric(ax, values, errs, ylabel, threshold=None, final_label=''):
    # Scatter with colour gradient
    for i in range(len(x) - 1):
        ax.plot(x[i:i+2], values[i:i+2], '-', color=colors[i+1], lw=2)
    ax.scatter(x, values, s=80, color=colors, zorder=5)
    # Error bar for Ch.6
    ax.errorbar(x[-1], values[-1], yerr=errs[-1],
                fmt='none', color=TEXT_COL, capsize=5, lw=1.5)
    if threshold is not None:
        ax.axhline(threshold, color='#fbbf24', lw=1.2, linestyle='--', alpha=0.8)
    # Annotate final point
    ax.annotate(
        final_label,
        xy=(x[-1], values[-1]),
        xytext=(x[-1] - 0.5, values[-1] + (max(values) - min(values)) * 0.12),
        color=TEXT_COL, fontsize=8,
        arrowprops=dict(arrowstyle='->', color=TEXT_COL, lw=0.8),
    )
    ax.set_ylabel(ylabel, color=TEXT_COL)
    _style_ax(ax)


# ── Figure ────────────────────────────────────────────────────────────────
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(BG_DARK)

_plot_metric(ax1, mae_k, mae_err, 'MAE ($k)', threshold=40, final_label='$38k ± $2k')
ax1.set_title('MAE ($k)', color=TEXT_COL)

_plot_metric(ax2, rmse_k, rmse_err, 'RMSE ($k)', threshold=52, final_label='$52k ± $3k')
ax2.set_title('RMSE ($k)', color=TEXT_COL)

_plot_metric(ax3, r2, r2_err, 'R²', final_label='0.68')
ax3.set_title('R²', color=TEXT_COL)

fig.suptitle(
    'SmartVal AI — Metrics Journey Ch.1 → Ch.6',
    color=TEXT_COL, fontsize=14, y=1.02,
)
plt.tight_layout()

out = Path(__file__).parent.parent / 'img' / 'ch06-metrics-journey.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG_DARK)
plt.close()
print(f"Saved: {out}")
