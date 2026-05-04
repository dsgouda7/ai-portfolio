"""gen_ch06_learning_curve.py
Generates img/ch06-learning-curve.png
Train vs validation MAE as a function of training set size.
Ch.5 Ridge pipeline (poly degree=2, alpha=1.0) on California Housing.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge
from sklearn.model_selection import learning_curve, train_test_split
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

train_sizes_rel, train_scores, val_scores = learning_curve(
    pipe, X_train, y_train,
    train_sizes=np.linspace(0.05, 1.0, 12),
    scoring='neg_mean_absolute_error',
    cv=5,
    n_jobs=-1,
)

# Convert to dollar MAE (positive)
to_dollars = lambda arr: -arr * 100_000
train_mae = to_dollars(train_scores.mean(axis=1))
val_mae   = to_dollars(val_scores.mean(axis=1))
train_std = train_scores.std(axis=1) * 100_000
val_std   = val_scores.std(axis=1)   * 100_000
n_train   = (train_sizes_rel * len(X_train)).astype(int)

gap = val_mae[-1] - train_mae[-1]

# ── Style constants ───────────────────────────────────────────────────────
BG_DARK  = '#1a1a2e'
BG_AXES  = '#16213e'
TEXT_COL = '#e2e8f0'
GRID_COL = '#374151'

# ── Figure ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor(BG_DARK)
ax.set_facecolor(BG_AXES)

ax.plot(n_train, train_mae, 'o-', color='#4ade80', lw=2, label='Train MAE')
ax.plot(n_train, val_mae,   's-', color='#fb923c', lw=2, label='Val MAE (5-fold CV)')
ax.fill_between(n_train, train_mae - train_std, train_mae + train_std,
                alpha=0.2, color='#4ade80')
ax.fill_between(n_train, val_mae - val_std, val_mae + val_std,
                alpha=0.2, color='#fb923c')
ax.axhline(40_000, color='#fbbf24', lw=1.5, linestyle='--', label='$40k target')

# Gap annotation at rightmost point
mid_gap_y = (train_mae[-1] + val_mae[-1]) / 2
ax.annotate(
    f"${gap:,.0f} gap",
    xy=(n_train[-1], mid_gap_y),
    xytext=(n_train[-1] * 0.75, mid_gap_y + 5_000),
    color=TEXT_COL, fontsize=9,
    arrowprops=dict(arrowstyle='->', color=TEXT_COL, lw=1.0),
)

ax.set_xlabel('Training set size (samples)', color=TEXT_COL)
ax.set_ylabel('MAE ($)', color=TEXT_COL)
ax.set_title('Learning Curve — Ridge poly d=2 | 5-fold CV', color=TEXT_COL)
ax.tick_params(colors=TEXT_COL)
for spine in ax.spines.values():
    spine.set_edgecolor(GRID_COL)
ax.grid(color=GRID_COL, alpha=0.3, linewidth=0.6)
ax.legend(facecolor=BG_AXES, labelcolor=TEXT_COL, edgecolor=GRID_COL)

plt.tight_layout()

out = Path(__file__).parent.parent / 'img' / 'ch06-learning-curve.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG_DARK)
plt.close()
print(f"Saved: {out}  |  gap at full data: ${gap:,.0f}")
