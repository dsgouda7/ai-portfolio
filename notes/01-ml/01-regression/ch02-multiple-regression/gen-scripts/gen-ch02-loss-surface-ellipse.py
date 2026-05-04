"""
gen_ch02_loss_surface_ellipse.py

Generates a 2-panel figure showing MSE loss surface contours for
unscaled vs standardized California Housing features, with gradient
descent paths overlaid.

Output: ../img/loss_surface_ellipse.png
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
data = fetch_california_housing()
X_all = data.data
y = data.target

# Use first 500 samples for speed
X_raw = X_all[:500, [0, 4]]   # MedInc (index 0), Population (index 4)
y_sub = y[:500]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# ---------------------------------------------------------------------------
# MSE helper  (intercept absorbed as b = y_mean - X @ w)
# ---------------------------------------------------------------------------
def mse_grid(X, y, W1, W2):
    """Compute MSE over a meshgrid of (w1, w2) values."""
    n = len(y)
    y_mean = np.mean(y)
    loss = np.empty_like(W1)
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            w = np.array([W1[i, j], W2[i, j]])
            b = y_mean - X @ w
            yhat = X @ w + b
            loss[i, j] = np.mean((yhat - y) ** 2)
    return loss


# ---------------------------------------------------------------------------
# Gradient descent path
# ---------------------------------------------------------------------------
def gradient_descent_path(X, y, n_steps, lr):
    w = np.zeros(2)
    n = len(y)
    path = [w.copy()]
    for _ in range(n_steps):
        yhat = X @ w + np.mean(y - X @ w)
        e = yhat - y
        dw = (2 / n) * X.T @ e
        w = w - lr * dw
        path.append(w.copy())
    return np.array(path)


def clamp_path(path, xlim, ylim):
    """Clamp path coordinates to stay within plot bounds."""
    out = path.copy()
    out[:, 0] = np.clip(out[:, 0], xlim[0], xlim[1])
    out[:, 1] = np.clip(out[:, 1], ylim[0], ylim[1])
    return out


# ---------------------------------------------------------------------------
# Build loss grids
# ---------------------------------------------------------------------------
GRID_N = 200

# Left panel — unscaled
xlim_raw = (-0.1, 0.5)
ylim_raw = (-0.0001, 0.0001)
w1_raw = np.linspace(*xlim_raw, GRID_N)
w2_raw = np.linspace(*ylim_raw, GRID_N)
W1_raw, W2_raw = np.meshgrid(w1_raw, w2_raw)
print("Computing unscaled loss grid …")
Z_raw = mse_grid(X_raw, y_sub, W1_raw, W2_raw)

# Right panel — standardized
xlim_sc = (-1.0, 2.0)
ylim_sc = (-1.5, 1.5)
w1_sc = np.linspace(*xlim_sc, GRID_N)
w2_sc = np.linspace(*ylim_sc, GRID_N)
W1_sc, W2_sc = np.meshgrid(w1_sc, w2_sc)
print("Computing scaled loss grid …")
Z_sc = mse_grid(X_scaled, y_sub, W1_sc, W2_sc)

# ---------------------------------------------------------------------------
# Gradient descent paths
# ---------------------------------------------------------------------------
print("Running gradient descent (unscaled) …")
path_raw = gradient_descent_path(X_raw, y_sub, n_steps=100, lr=1e-6)
path_raw_clamped = clamp_path(path_raw, xlim_raw, ylim_raw)

print("Running gradient descent (scaled) …")
path_sc = gradient_descent_path(X_scaled, y_sub, n_steps=100, lr=0.01)
path_sc_clamped = clamp_path(path_sc, xlim_sc, ylim_sc)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
BG = "#1a1a2e"
LINE_COLOR = "white"
N_CONTOURS = 20

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor(BG)

panel_cfg = [
    dict(
        ax=axes[0],
        W1=W1_raw, W2=W2_raw, Z=Z_raw,
        path=path_raw_clamped,
        xlim=xlim_raw, ylim=ylim_raw,
        title="Unscaled Features\n(Population raw vs MedInc raw)\nEigenvalue ratio ≈ 5,000",
        annot="Zigzag path: √5000 = 70× more iters",
        annot_xy=(0.05, 0.87),
    ),
    dict(
        ax=axes[1],
        W1=W1_sc, W2=W2_sc, Z=Z_sc,
        path=path_sc_clamped,
        xlim=xlim_sc, ylim=ylim_sc,
        title="StandardScaler Applied\n(all features σ=1)\nEigenvalue ratio ≈ 2–10",
        annot="Smooth spiral: fast convergence",
        annot_xy=(0.05, 0.87),
    ),
]

for cfg in panel_cfg:
    ax = cfg["ax"]
    ax.set_facecolor(BG)

    Z = cfg["Z"]
    # Log-scale the loss so both shallow and steep regions are visible
    Z_plot = np.log1p(Z - Z.min())

    ax.contour(
        cfg["W1"], cfg["W2"], Z_plot,
        levels=N_CONTOURS,
        colors=LINE_COLOR,
        linewidths=0.6,
        alpha=0.7,
    )

    # Colour-coded gradient descent path (red → green via RdYlGn)
    path = cfg["path"]
    n_pts = len(path)
    cmap = plt.cm.RdYlGn
    for k in range(n_pts - 1):
        color = cmap(k / max(n_pts - 2, 1))
        ax.plot(
            path[k : k + 2, 0],
            path[k : k + 2, 1],
            color=color,
            linewidth=2.2,
            solid_capstyle="round",
        )

    # Start marker (red circle) and end marker (green star)
    ax.plot(path[0, 0], path[0, 1], "o", color="#ff4444", markersize=9,
            zorder=5, label="Start")
    ax.plot(path[-1, 0], path[-1, 1], "*", color="#44ff88", markersize=13,
            zorder=5, label="End")

    ax.set_xlim(cfg["xlim"])
    ax.set_ylim(cfg["ylim"])
    ax.set_title(cfg["title"], color="white", fontsize=10, pad=8)
    ax.set_xlabel("w₁ (MedInc weight)", color="white", fontsize=9)
    ax.set_ylabel("w₂ (Population weight)", color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#555577")

    # Annotation
    ax.text(
        *cfg["annot_xy"], cfg["annot"],
        transform=ax.transAxes,
        color="#ffdd88", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#2a2a4e", alpha=0.8,
                  edgecolor="#888888"),
    )

fig.suptitle(
    "Loss Surface Shape: Why StandardScaler Matters",
    color="white", fontsize=13, fontweight="bold", y=1.01,
)
plt.tight_layout()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = Path(__file__).parent.parent / "img" / "loss_surface_ellipse.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out_path}")
