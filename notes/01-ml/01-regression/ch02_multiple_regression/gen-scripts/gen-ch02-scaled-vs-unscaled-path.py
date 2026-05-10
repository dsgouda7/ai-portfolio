"""
gen_ch02_scaled_vs_unscaled_path.py
Generates an animated GIF showing gradient descent on scaled vs unscaled features.
Output: ../img/scaled_vs_unscaled_path.gif
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from pathlib import Path
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# ── Output path ────────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent.parent / "img"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "scaled_vs_unscaled_path.gif"

# ── Data setup ─────────────────────────────────────────────────────────────────
data = fetch_california_housing()
X_all = data.data[:300, [0, 4]]   # MedInc, Population — 300 samples
y = data.target[:300]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)
X_raw = X_all.copy()
y_mean = np.mean(y)

# ── Gradient descent ───────────────────────────────────────────────────────────
def run_gd(X, y, n_steps, lr):
    w = np.zeros(2)
    path = [w.copy()]
    mses = []
    for _ in range(n_steps):
        b = np.mean(y - X @ w)
        yhat = X @ w + b
        e = yhat - y
        n = len(y)
        dw = (2 / n) * X.T @ e
        w = w - lr * dw
        path.append(w.copy())
        mses.append(np.mean(e**2))
    return np.array(path), mses

N_STEPS = 80
path_raw, mses_raw = run_gd(X_raw, y, N_STEPS, lr=5e-9)
path_scaled, mses_scaled = run_gd(X_scaled, y, N_STEPS, lr=0.05)

# ── Pre-compute loss contours ──────────────────────────────────────────────────
def compute_contour(X, y, w1_range, w2_range):
    W1, W2 = np.meshgrid(w1_range, w2_range)
    Z = np.zeros_like(W1)
    n = len(y)
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            w = np.array([W1[i, j], W2[i, j]])
            b = np.mean(y - X @ w)
            yhat = X @ w + b
            Z[i, j] = np.mean((yhat - y) ** 2)
    return W1, W2, Z

# Unscaled grid
w1_raw = np.linspace(-0.05, 0.6, 60)
w2_raw = np.linspace(-0.0002, 0.0002, 60)
W1_raw, W2_raw, Z_raw = compute_contour(X_raw, y, w1_raw, w2_raw)

# Scaled grid
w1_sc = np.linspace(-0.5, 1.5, 60)
w2_sc = np.linspace(-1.0, 0.5, 60)
W1_sc, W2_sc, Z_sc = compute_contour(X_scaled, y, w1_sc, w2_sc)

# ── Figure setup ───────────────────────────────────────────────────────────────
BG = "#1a1a2e"
fig, (ax_raw, ax_sc) = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor(BG)

for ax in (ax_raw, ax_sc):
    ax.set_facecolor(BG)
    ax.tick_params(colors="#cccccc", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")

# Contour levels (log-spaced for better visual spread)
def make_levels(Z):
    lo, hi = Z.min(), Z.max()
    return np.geomspace(max(lo, 1e-6), hi, 20)

ax_raw.contour(W1_raw, W2_raw, Z_raw, levels=make_levels(Z_raw),
               cmap="plasma", alpha=0.6)
ax_sc.contour(W1_sc, W2_sc, Z_sc, levels=make_levels(Z_sc),
              cmap="plasma", alpha=0.6)

ax_raw.set_xlabel("w₁ (MedInc)", color="#aaaacc", fontsize=9)
ax_raw.set_ylabel("w₂ (Population)", color="#aaaacc", fontsize=9)
ax_raw.set_title("Unscaled Features\n(η=5e-9)", color="#e0e0ff",
                 fontsize=11, fontweight="bold")

ax_sc.set_xlabel("w₁ (MedInc scaled)", color="#aaaacc", fontsize=9)
ax_sc.set_ylabel("w₂ (Population scaled)", color="#aaaacc", fontsize=9)
ax_sc.set_title("StandardScaler Applied\n(η=0.05)", color="#e0e0ff",
                fontsize=11, fontweight="bold")

# Start marker (raw panel only)
ax_raw.plot(0, 0, "o", color="#ff4444", markersize=8, zorder=5, label="start")

# Trail lines (initially empty)
trail_raw, = ax_raw.plot([], [], "-", color="#88ccff", linewidth=1.2,
                         alpha=0.7, zorder=3)
trail_sc, = ax_sc.plot([], [], "-", color="#88ffaa", linewidth=1.2,
                       alpha=0.7, zorder=3)

# Current position dots
dot_raw, = ax_raw.plot([], [], "o", color="#ffffff", markersize=7, zorder=6)
dot_sc, = ax_sc.plot([], [], "o", color="#ffffff", markersize=7, zorder=6)

# Shared bottom text
bottom_text = fig.text(
    0.5, 0.01,
    "Epoch: 0 | MSE_raw: --- | MSE_scaled: ---",
    ha="center", va="bottom", color="#ddddff",
    fontsize=10, fontfamily="monospace"
)

fig.tight_layout(rect=[0, 0.05, 1, 1])

# ── Animation update ───────────────────────────────────────────────────────────
def update(frame):
    # Paths are 0..N_STEPS (N_STEPS+1 points); frame 0 = start
    idx = min(frame, N_STEPS)

    # Trail
    trail_raw.set_data(path_raw[:idx + 1, 0], path_raw[:idx + 1, 1])
    trail_sc.set_data(path_scaled[:idx + 1, 0], path_scaled[:idx + 1, 1])

    # Current dot
    dot_raw.set_data([path_raw[idx, 0]], [path_raw[idx, 1]])
    dot_sc.set_data([path_scaled[idx, 0]], [path_scaled[idx, 1]])

    # MSE values (mses list starts at step 1)
    mse_r = mses_raw[idx - 1] if idx > 0 else float("nan")
    mse_s = mses_scaled[idx - 1] if idx > 0 else float("nan")

    label_r = f"{mse_r:.3f}" if not np.isnan(mse_r) else "---"
    label_s = f"{mse_s:.3f}" if not np.isnan(mse_s) else "---"

    bottom_text.set_text(
        f"Epoch: {idx}  |  MSE_raw: {label_r}  |  MSE_scaled: {label_s}"
    )

    return trail_raw, trail_sc, dot_raw, dot_sc, bottom_text

# ── Save ───────────────────────────────────────────────────────────────────────
ani = animation.FuncAnimation(
    fig, update,
    frames=N_STEPS + 1,
    interval=100,
    blit=True
)

writer = animation.PillowWriter(fps=10)
ani.save(str(OUT_PATH), writer=writer)
plt.close(fig)

print(f"Saved: {OUT_PATH}")
