"""
gen_ch05_lambda_sweep.py
Generates: ../img/ch05-lambda-sweep.png

U-shaped curve showing Train MAE and CV MAE vs log10(λ) for Ridge regression
on California Housing with degree-2 polynomial features (44 features).
Highlights overfitting zone, sweet spot, and underfitting zone.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "img", "ch05-lambda-sweep.png")

# ── Data ──────────────────────────────────────────────────────────────────────
data = fetch_california_housing()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Sweep ─────────────────────────────────────────────────────────────────────
alphas = np.logspace(-3, 3, 60)
train_maes, cv_maes = [], []

for alpha in alphas:
    pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])
    pipe.fit(X_train, y_train)
    train_maes.append(mean_absolute_error(y_train, pipe.predict(X_train)) * 100_000)
    scores = cross_val_score(
        pipe, X_train, y_train, cv=5,
        scoring="neg_mean_absolute_error", n_jobs=-1
    )
    cv_maes.append(-scores.mean() * 100_000)

log_alphas = np.log10(alphas)
train_maes = np.array(train_maes)
cv_maes = np.array(cv_maes)

best_idx = int(np.argmin(cv_maes))
optimal_log_alpha = log_alphas[best_idx]
best_cv_mae = cv_maes[best_idx]

print(f"Optimal log10(alpha) = {optimal_log_alpha:.2f}  ->  alpha ~= {alphas[best_idx]:.4f}")
print(f"Best CV MAE = ${best_cv_mae:,.0f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#1a1a2e")

# Shaded zones
sw = 0.5  # zone half-width in log units
ax.axvspan(log_alphas[0], optimal_log_alpha - sw,
           color="#b91c1c", alpha=0.08, label="Overfitting zone")
ax.axvspan(optimal_log_alpha - sw, optimal_log_alpha + sw,
           color="#15803d", alpha=0.12, label="Sweet spot")
ax.axvspan(optimal_log_alpha + sw, log_alphas[-1],
           color="#1e40af", alpha=0.08, label="Underfitting zone")

ax.text(optimal_log_alpha - sw - 0.4, 72_000, "Overfitting\nzone",
        color="#f87171", ha="center", fontsize=9, alpha=0.9)
ax.text(optimal_log_alpha, 72_000, "Sweet spot",
        color="#4ade80", ha="center", fontsize=9, alpha=0.9)
ax.text(optimal_log_alpha + sw + 0.7, 72_000, "Underfitting\nzone",
        color="#93c5fd", ha="center", fontsize=9, alpha=0.9)

# MAE curves
ax.plot(log_alphas, train_maes, color="#f59e0b", linewidth=2, label="Train MAE", zorder=4)
ax.plot(log_alphas, cv_maes, color="#60a5fa", linewidth=2.5, label="CV MAE (5-fold)", zorder=5)

# Optimal λ line
ax.axvline(optimal_log_alpha, color="#4ade80", linestyle="--", linewidth=1.5, zorder=6,
           label=f"Optimal λ ≈ {alphas[best_idx]:.3f}")

# Target lines
ax.axhline(38_000, color="white", linestyle=":", linewidth=1, alpha=0.6,
           label="$38k target")
ax.axhline(40_000, color="#f59e0b", linestyle=":", linewidth=1, alpha=0.6,
           label="$40k constraint")

ax.set_xlim(log_alphas[0], log_alphas[-1])
ax.set_ylim(30_000, 80_000)
ax.set_xlabel("log₁₀(λ)", color="white", fontsize=12)
ax.set_ylabel("MAE ($)", color="white", fontsize=12)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
ax.set_title("Ridge λ Sweep — California Housing (44 polynomial features)",
             color="white", fontsize=13, pad=12)
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#4a4a6a")

legend = ax.legend(loc="upper right", fontsize=9, framealpha=0.3,
                   labelcolor="white", facecolor="#1a1a2e", edgecolor="#4a4a6a")

plt.tight_layout()
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=150, facecolor="#1a1a2e")
plt.close()
print(f"Saved: {OUT_PATH}")
