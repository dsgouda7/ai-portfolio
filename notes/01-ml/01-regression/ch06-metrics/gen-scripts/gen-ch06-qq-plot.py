"""gen_ch06_qq_plot.py
Generates img/ch06-qq-plot.png
Q-Q plot of residuals checking normality.
Ch.5 Ridge pipeline (poly degree=2, alpha=1.0) on California Housing.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge
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

residuals = (y_test - y_pred) * 100_000  # dollars

# ── Q-Q probability plot ──────────────────────────────────────────────────
(osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist='norm')
ref_x = np.array([osm.min(), osm.max()])
ref_y = slope * ref_x + intercept

# ── Style constants ───────────────────────────────────────────────────────
BG_DARK  = '#1a1a2e'
BG_AXES  = '#16213e'
TEXT_COL = '#e2e8f0'
GRID_COL = '#374151'
DOT_COL  = '#60a5fa'
RED_LINE = '#f87171'

# ── Figure ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))
fig.patch.set_facecolor(BG_DARK)
ax.set_facecolor(BG_AXES)

ax.scatter(osm, osr, color=DOT_COL, alpha=0.4, s=6)
ax.plot(ref_x, ref_y, color=RED_LINE, lw=2)

# Tail annotations
ax.annotate(
    "Negative tail thicker than normal\n(model over-predicts cheap homes)",
    xy=(osm[10], osr[10]),
    xytext=(-3.5, -200_000),
    color=TEXT_COL, fontsize=8,
    arrowprops=dict(arrowstyle='->', color=TEXT_COL, lw=1.0),
)
ax.annotate(
    "Positive tail much thicker\n(model under-predicts luxury homes)",
    xy=(osm[-10], osr[-10]),
    xytext=(1.0, 350_000),
    color=TEXT_COL, fontsize=8,
    arrowprops=dict(arrowstyle='->', color=TEXT_COL, lw=1.0),
)

ax.set_xlabel('Theoretical Quantiles', color=TEXT_COL)
ax.set_ylabel('Residual Quantiles ($)', color=TEXT_COL)
ax.set_title('Q-Q Plot of Residuals — Normal Reference', color=TEXT_COL)
ax.tick_params(colors=TEXT_COL)
for spine in ax.spines.values():
    spine.set_edgecolor(GRID_COL)
ax.grid(color=GRID_COL, alpha=0.3, linewidth=0.6)

# Caption
ax.text(
    0.5, -0.10,
    f"R²={r**2:.3f}  (1.0 = perfect normality)",
    transform=ax.transAxes,
    ha='center', va='top', color=TEXT_COL, fontsize=9,
)

plt.tight_layout()

out = Path(__file__).parent.parent / 'img' / 'ch06-qq-plot.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG_DARK)
plt.close()
print(f"Saved: {out}")
