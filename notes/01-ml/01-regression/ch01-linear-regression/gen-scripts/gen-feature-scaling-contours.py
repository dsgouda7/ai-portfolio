"""
gen_feature_scaling_contours.py
--------------------------------
Generates img/feature_scaling_contours.png — a two-panel figure showing
loss contours and gradient-descent paths with and without feature scaling.

Left panel  — unscaled features: elongated ellipse contours; zigzag path.
Right panel — standardised features: circular contours; direct path.

Run from the repository root:
    python notes/01-ml/01_regression/ch01_linear_regression/gen_scripts/gen_feature_scaling_contours.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── output path ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(SCRIPT_DIR, "..", "img")
os.makedirs(IMG_DIR, exist_ok=True)
OUT_PATH = os.path.join(IMG_DIR, "feature_scaling_contours.png")

# ── colour palette ────────────────────────────────────────────────────────────
C_CONTOUR = "#94a3b8"
C_PATH    = "#ef4444"   # red path
C_START   = "#1d4ed8"   # blue start dot
C_END     = "#15803d"   # green minimum dot
C_ARROW   = "#f59e0b"   # amber gradient arrow

# ── helpers ───────────────────────────────────────────────────────────────────
def quadratic_loss(w1, w2, A, b_vec, c):
    """L = w^T A w + b^T w + c  (A is 2×2 positive definite)."""
    W = np.stack([w1.ravel(), w2.ravel()], axis=1)   # (N, 2)
    quad = np.einsum("ni,ij,nj->n", W, A, W)
    lin  = W @ b_vec
    return (quad + lin + c).reshape(w1.shape)


def gd_path(w_init, A, b_vec, lr, n_steps):
    """Gradient descent: grad = 2Aw + b."""
    path = [np.array(w_init, dtype=float)]
    w = np.array(w_init, dtype=float)
    for _ in range(n_steps):
        grad = 2.0 * A @ w + b_vec
        w = w - lr * grad
        path.append(w.copy())
    return np.array(path)


# ── problem setup ─────────────────────────────────────────────────────────────
# Unscaled: feature 1 ~ range 3, feature 2 ~ range 37
# This makes the loss surface elongated along axis 1.
# A_unscaled chosen so eigenvalues ratio ≈ (37/3)² ≈ 150 (extreme stretch).
A_unscaled = np.array([[1.0, 0.0],
                        [0.0, 150.0]])
b_unscaled = np.array([-2.0, -300.0])     # minimum near (1, 1)
c_unscaled = 200.0

# Scaled: both features ~ range 1 → symmetric bowl
A_scaled = np.array([[1.0, 0.0],
                     [0.0, 1.0]])
b_scaled = np.array([-2.0, -2.0])
c_scaled = 2.0

# Minima (where 2Aw* + b = 0  →  w* = -A^{-1} b / 2)
w_star_unscaled = -np.linalg.solve(2 * A_unscaled, b_unscaled)
w_star_scaled   = -np.linalg.solve(2 * A_scaled,   b_scaled)

# GD paths
w0_unscaled = np.array([-1.0,  0.5])   # start far from minimum
w0_scaled   = np.array([-1.0, -1.0])

path_unscaled = gd_path(w0_unscaled, A_unscaled, b_unscaled, lr=0.005, n_steps=40)
path_scaled   = gd_path(w0_scaled,   A_scaled,   b_scaled,   lr=0.25,  n_steps=12)

# ── figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
fig.patch.set_facecolor("#0f172a")
for ax in axes:
    ax.set_facecolor("#1e293b")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

# ── shared contour levels ──────────────────────────────────────────────────────
def draw_panel(ax, A, b_vec, c, path, w_star,
               x_lim, y_lim, title, xlabel, ylabel):
    x1 = np.linspace(*x_lim, 400)
    x2 = np.linspace(*y_lim, 400)
    W1, W2 = np.meshgrid(x1, x2)
    Z = quadratic_loss(W1, W2, A, b_vec, c)

    lvls = np.percentile(Z, np.linspace(2, 80, 14))
    lvls = np.unique(np.round(lvls, 4))

    ax.contour(W1, W2, Z, levels=lvls, colors=C_CONTOUR, linewidths=0.7, alpha=0.6)

    # GD path
    ax.plot(path[:, 0], path[:, 1], color=C_PATH, linewidth=1.6,
            zorder=4, label="GD path")
    for k in range(len(path) - 1):
        dx = path[k+1, 0] - path[k, 0]
        dy = path[k+1, 1] - path[k, 1]
        ax.annotate("", xy=(path[k, 0] + dx, path[k, 1] + dy),
                    xytext=(path[k, 0], path[k, 1]),
                    arrowprops=dict(arrowstyle="-|>", color=C_PATH,
                                   lw=1.0, mutation_scale=8))

    # Start / minimum dots
    ax.scatter(*path[0],   s=60, color=C_START, zorder=6, label="start")
    ax.scatter(*w_star,    s=80, color=C_END,   zorder=6, marker="*",
               label="minimum")

    ax.set_xlim(x_lim); ax.set_ylim(y_lim)
    ax.set_xlabel(xlabel, color="#94a3b8", fontsize=9)
    ax.set_ylabel(ylabel, color="#94a3b8", fontsize=9)
    ax.tick_params(colors="#64748b", labelsize=8)
    ax.set_title(title, color="#e2e8f0", fontsize=10, pad=8)

    legend = ax.legend(fontsize=8, framealpha=0.25, labelcolor="#e2e8f0",
                       facecolor="#1e293b", edgecolor="#334155")

draw_panel(
    axes[0], A_unscaled, b_unscaled, c_unscaled, path_unscaled, w_star_unscaled,
    x_lim=(-2.0, 2.5), y_lim=(0.2, 1.8),
    title="Unscaled features\n(elongated ellipse — zigzag path)",
    xlabel="$w_{\\mathrm{MedInc}}$",
    ylabel="$w_{\\mathrm{Latitude}}$",
)

draw_panel(
    axes[1], A_scaled, b_scaled, c_scaled, path_scaled, w_star_scaled,
    x_lim=(-2.0, 2.0), y_lim=(-2.0, 2.0),
    title="Standardised features\n(circular bowl — direct descent)",
    xlabel="$w_{1}$ (standardised)",
    ylabel="$w_{2}$ (standardised)",
)

fig.suptitle(
    "Feature scaling and the loss surface",
    color="#f1f5f9", fontsize=12, fontweight="bold", y=1.01,
)
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved: {OUT_PATH}")
