"""gen_ch06_residuals_vs_predicted.py
Generates img/ch06-residuals-vs-predicted.png
Two-panel figure: residuals vs predicted scatter + residual histogram.
Ch.5 Ridge pipeline (poly degree=2, alpha=1.0) on California Housing.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# ── Data & model ─────────────────────────────────────────────────────────
data = fetch_california_housing()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipe = Pipeline([
    ('poly',   PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('model',  Ridge(alpha=1.0)),
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

residuals   = (y_test - y_pred) * 100_000   # dollars
y_pred_d    = y_pred * 100_000              # dollars

mae  = mean_absolute_error(y_test, y_pred) * 100_000
rmse = np.sqrt(mean_squared_error(y_test, y_pred)) * 100_000

# ── Style constants ───────────────────────────────────────────────────────
BG_DARK  = '#1a1a2e'
BG_AXES  = '#16213e'
TEXT_COL = '#e2e8f0'
GRID_COL = '#374151'
DOT_COL  = '#60a5fa'
RED_LINE = '#f87171'
HIST_COL = '#818cf8'

def _style_ax(ax):
    ax.set_facecolor(BG_AXES)
    ax.tick_params(colors=TEXT_COL)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    ax.title.set_color(TEXT_COL)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.grid(color=GRID_COL, alpha=0.3, linewidth=0.6)

# ── Figure ────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor(BG_DARK)

# Panel 1 — Residuals vs Predicted
ax1.scatter(y_pred_d, residuals, alpha=0.15, s=4, color=DOT_COL)
ax1.axhline(0, color=RED_LINE, lw=2, linestyle='--')
ax1.axvline(400_000, color='#fbbf24', lw=1.2, linestyle=':', alpha=0.8)

# Annotation for luxury-segment underestimate
mask_luxury = y_pred_d > 400_000
if mask_luxury.sum() > 0:
    anchor_x = y_pred_d[mask_luxury & (residuals > 0)].mean() if (mask_luxury & (residuals > 0)).sum() > 0 else 430_000
    anchor_y = residuals[mask_luxury & (residuals > 0)].mean() if (mask_luxury & (residuals > 0)).sum() > 0 else 60_000
    ax1.annotate(
        "Systematic underestimate\nabove $400k",
        xy=(anchor_x, anchor_y),
        xytext=(300_000, 150_000),
        color=TEXT_COL, fontsize=8,
        arrowprops=dict(arrowstyle='->', color=TEXT_COL, lw=1.2),
    )

ax1.set_xlabel('Predicted Price ($)', color=TEXT_COL)
ax1.set_ylabel('Residual ($)', color=TEXT_COL)
ax1.set_title('Residuals vs Predicted — Ridge poly d=2', color=TEXT_COL)
_style_ax(ax1)

# Panel 2 — Residual Histogram
ax2.hist(residuals, bins=60, color=HIST_COL, edgecolor=BG_DARK, alpha=0.85)
ax2.axvline(0, color=RED_LINE, lw=2, linestyle='--')
ax2.text(
    0.97, 0.95,
    f"mean = ${residuals.mean():+,.0f}\nstd  = ${residuals.std():,.0f}",
    transform=ax2.transAxes,
    ha='right', va='top', color=TEXT_COL, fontsize=8,
    bbox=dict(facecolor=BG_AXES, edgecolor=GRID_COL, alpha=0.7),
)
ax2.set_xlabel('Residual ($)', color=TEXT_COL)
ax2.set_ylabel('Count', color=TEXT_COL)
ax2.set_title('Residual Distribution', color=TEXT_COL)
_style_ax(ax2)

fig.suptitle(
    f'Ridge poly d=2 | MAE=${mae:,.0f}  RMSE=${rmse:,.0f}',
    color=TEXT_COL, fontsize=11, y=1.01,
)
plt.tight_layout()

out = Path(__file__).parent.parent / 'img' / 'ch06-residuals-vs-predicted.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG_DARK)
plt.close()
print(f"Saved: {out}")
