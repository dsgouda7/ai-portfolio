"""
gen_ch05_lambda_powers_sweep.py
Generates: ../img/ch05-lambda-powers-sweep.gif

Animation showing how Ridge regularization affects weights across powers of 10
for lambda (0.001, 0.01, 0.1, 1.0, 10, 100, 1000), demonstrating the progression
from minimal regularization to extreme shrinkage.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "img", "ch05-lambda-powers-sweep.gif")

# ── Data ──────────────────────────────────────────────────────────────────────
data = fetch_california_housing()
X, y = data.data, data.target

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
X_scaled = StandardScaler().fit_transform(X_poly)
feature_names = poly.get_feature_names_out(data.feature_names)

# ── Lambda values (powers of 10) ──────────────────────────────────────────────
lambdas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

# ── Track specific features ───────────────────────────────────────────────────
# Build mapping: we want to track key features to show regularization effect
TRACKED_FEATURES_SEARCH = [
    ("MedInc", "#f59e0b"),
    ("Latitude", "#60a5fa"),
    ("AveRooms AveBedrms", "#a78bfa"),
    ("Population AveBedrms", "#f87171"),
    ("AveOccup^2", "#fb923c"),
]

# Find indices - sklearn uses format like "MedInc", "Latitude", "AveRooms AveBedrms", "AveOccup^2"
tracked_indices = {}
tracked_order = []

for search_name, color in TRACKED_FEATURES_SEARCH:
    found = False
    for i, fname in enumerate(feature_names):
        if fname == search_name:
            tracked_indices[search_name] = (i, color)
            tracked_order.append(search_name)
            found = True
            break
    if not found:
        print(f"Warning: Could not find feature '{search_name}' in feature_names")

# If some weren't found, use the first 5 features as fallback
if len(tracked_indices) < 5:
    tracked_indices = {}
    tracked_order = []
    colors = ["#f59e0b", "#60a5fa", "#a78bfa", "#f87171", "#fb923c"]
    for i, fname in enumerate(feature_names[:5]):
        tracked_indices[fname] = (i, colors[i])
        tracked_order.append(fname)

# ── Compute coefficients for each lambda ──────────────────────────────────────
all_coefs = []
tracked_coefs = {name: [] for name in tracked_order}

for lam in lambdas:
    ridge = Ridge(alpha=lam)
    ridge.fit(X_scaled, y)
    coefs = ridge.coef_
    all_coefs.append(coefs)
    
    for name in tracked_order:
        idx, _ = tracked_indices[name]
        tracked_coefs[name].append(coefs[idx])

# ── Animation setup ────────────────────────────────────────────────────────────
fig, (ax_main, ax_bar) = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor("#1a1a2e")

for ax in [ax_main, ax_bar]:
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#4a4a6a")

# Left panel: coefficient trajectories
log_lambdas = np.log10(lambdas)
ax_main.axhline(0, color="white", alpha=0.3, linewidth=1)
ax_main.set_xlabel("log₁₀(λ)", color="white", fontsize=12)
ax_main.set_ylabel("Coefficient value", color="white", fontsize=12)
ax_main.set_title("Ridge Regularization: Weight Shrinkage Across λ",
                  color="white", fontsize=13, pad=12)
ax_main.set_xlim(-3.2, 3.2)
ax_main.set_ylim(-0.8, 0.8)

# Right panel: current weights as bars
ax_bar.set_xlabel("Features", color="white", fontsize=12)
ax_bar.set_ylabel("Coefficient value", color="white", fontsize=12)
ax_bar.set_title("Current Weights (λ = ?)",
                 color="white", fontsize=13, pad=12)
ax_bar.set_ylim(-0.8, 0.8)

# Plot all feature trajectories in background (gray)
for j in range(len(all_coefs[0])):
    vals = [coefs[j] for coefs in all_coefs]
    ax_main.plot(log_lambdas, vals, color="#4a4a6a", alpha=0.2, linewidth=0.8)

# Plot tracked features
lines = {}
for name, (idx, color) in tracked_indices.items():
    vals = tracked_coefs[name]
    line, = ax_main.plot(log_lambdas, vals, color=color, linewidth=2.5, 
                         label=name, alpha=0.8)
    lines[name] = line

ax_main.legend(loc="upper right", fontsize=9, framealpha=0.3,
               labelcolor="white", facecolor="#1a1a2e", edgecolor="#4a4a6a")

# Initialize bar chart
x_pos = np.arange(len(tracked_order))
bars = ax_bar.bar(x_pos, [0] * len(tracked_order), 
                  color=[tracked_indices[n][1] for n in tracked_order],
                  alpha=0.8, edgecolor="white", linewidth=1.5)
ax_bar.set_xticks(x_pos)
ax_bar.set_xticklabels([n.replace(" ", "\n") for n in tracked_order], 
                        rotation=0, fontsize=9, color="white")
ax_bar.axhline(0, color="white", alpha=0.3, linewidth=1)

# Vertical marker on trajectory plot
vline = ax_main.axvline(log_lambdas[0], color="#4ade80", linestyle="--", 
                        linewidth=2, alpha=0.8)

# ── Animation function ────────────────────────────────────────────────────────
def update(frame):
    lam_val = lambdas[frame]
    log_lam = log_lambdas[frame]
    
    # Update vertical line
    vline.set_xdata([log_lam, log_lam])
    
    # Update bar heights
    for i, (name, bar) in enumerate(zip(tracked_order, bars)):
        bar.set_height(tracked_coefs[name][frame])
    
    # Update title
    ax_bar.set_title(f"Current Weights (λ = {lam_val:.3f})",
                     color="white", fontsize=13, pad=12)
    
    return [vline] + list(bars)

# ── Create animation ──────────────────────────────────────────────────────────
anim = FuncAnimation(fig, update, frames=len(lambdas), interval=800, blit=True)

# Save
writer = PillowWriter(fps=1)
anim.save(OUT_PATH, writer=writer)
plt.close()

print(f"✓ Generated: {OUT_PATH}")
