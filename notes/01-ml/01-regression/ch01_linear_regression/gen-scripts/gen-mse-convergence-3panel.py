"""Generate 3-panel convergence animation for Ch.01.

Demonstrates the 4th (missing) property of MSE gradient descent:
  the gradient self-brakes to 0 as w approaches w* (the minimum).
  MAE's gradient is a sum of sign terms — magnitude stays fixed near min.

Three panels, built epoch-by-epoch:
  1. w and b marching toward optimal w*, b*
  2. |∂L/∂w| decaying to 0 (MSE) vs staying elevated (MAE)
  3. Loss descending each epoch

Outputs:
    ../img/mse_convergence_3panel.gif   — animated GIF (~50 frames)

Run:
    python gen_mse_convergence_3panel.py
"""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# ── Palette (matches repo conventions) ────────────────────────────────────
DARK   = "#1f2937"
BLUE   = "#2563eb"     # MSE
AMBER  = "#d97706"     # MAE
GREEN  = "#16a34a"     # optimal reference
GRID   = "#374151"
LIGHT  = "#e2e8f0"

HERE     = Path(__file__).parent
OUT_DIR  = HERE / ".." / "img"
OUT_DIR.mkdir(parents=True, exist_ok=True)
GIF_PATH = OUT_DIR / "mse_convergence_3panel.gif"

# ── Data ──────────────────────────────────────────────────────────────────
housing = fetch_california_housing(as_frame=True)
X_raw = housing.data["MedInc"].values
y_raw = housing.target.values

# Reproducible subsample for render speed
rng = np.random.default_rng(42)
idx = rng.choice(len(X_raw), size=2000, replace=False)
X_raw, y_raw = X_raw[idx], y_raw[idx]

# Standardize so both axes are on comparable scales
sx = StandardScaler()
sy = StandardScaler()
X = sx.fit_transform(X_raw.reshape(-1, 1)).ravel()
y = sy.fit_transform(y_raw.reshape(-1, 1)).ravel()
N = len(X)

# OLS optimal parameters (reference lines)
cov_xy = float(np.cov(X, y)[0, 1])
var_x  = float(np.var(X, ddof=0))
w_opt  = cov_xy / var_x
b_opt  = float(np.mean(y)) - w_opt * float(np.mean(X))

# ── Gradient descent ──────────────────────────────────────────────────────
EPOCHS = 50
LR_MSE = 0.06
LR_MAE = 0.025


def run_gd(loss_type: str):
    """Return (ws, bs, |grad_w|s, losses) — all length EPOCHS+1.

    Index k holds values *at the start of epoch k* (before the update at
    epoch k is applied), so ws[k] and grads[k] are in sync.
    """
    lrate = LR_MSE if loss_type == "mse" else LR_MAE
    w, b  = 0.0, 0.0
    ws, bs, grads, losses = [w], [b], [], []

    for _ in range(EPOCHS):
        err = w * X + b - y
        if loss_type == "mse":
            gw   = (2 / N) * float(X @ err)
            gb   = (2 / N) * float(err.sum())
            loss = float(np.mean(err ** 2))
        else:
            gw   = float(np.mean(np.sign(err) * X))
            gb   = float(np.mean(np.sign(err)))
            loss = float(np.mean(np.abs(err)))

        grads.append(abs(gw))
        losses.append(loss)
        w -= lrate * gw
        b -= lrate * gb
        ws.append(w)
        bs.append(b)

    # Final point (after last update)
    err = w * X + b - y
    if loss_type == "mse":
        grads.append(abs((2 / N) * float(X @ err)))
        losses.append(float(np.mean(err ** 2)))
    else:
        grads.append(abs(float(np.mean(np.sign(err) * X))))
        losses.append(float(np.mean(np.abs(err))))

    return (np.array(ws), np.array(bs),
            np.array(grads), np.array(losses))


ws_mse, bs_mse, grads_mse, losses_mse = run_gd("mse")
ws_mae, bs_mae, grads_mae, losses_mae = run_gd("mae")

ep_axis = np.arange(EPOCHS + 1)

# ── Fixed axis limits (computed once so the view doesn't jump) ────────────
param_lo = min(0.0, ws_mse.min(), ws_mae.min(),
               bs_mse.min(), bs_mae.min()) - 0.05
param_hi = max(ws_mse.max(), ws_mae.max(), w_opt,
               bs_mse.max(), bs_mae.max(), b_opt) + 0.12

grad_hi  = max(grads_mse.max(), grads_mae.max()) * 1.08
loss_hi  = max(losses_mse.max(), losses_mae.max()) * 1.08


# ── Frame builder ─────────────────────────────────────────────────────────
def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(DARK)
    ax.tick_params(colors=LIGHT, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.xaxis.label.set_color(LIGHT)
    ax.yaxis.label.set_color(LIGHT)
    ax.title.set_color(LIGHT)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.5)


def make_frame(k: int) -> np.ndarray:
    """Render state at epoch k (0 = initialisation, EPOCHS = converged)."""
    ep = ep_axis[: k + 1]

    fig, axes = plt.subplots(
        1, 3, figsize=(14, 4.4), facecolor=DARK,
    )
    fig.subplots_adjust(left=0.07, right=0.97, top=0.84, bottom=0.14, wspace=0.38)

    # ── Panel 1: parameter trajectories ──────────────────────────────────
    ax = axes[0]
    _style(ax)
    ax.plot(ep, ws_mse[:k+1], color=BLUE,  lw=2.0, label="w  (MSE)")
    ax.plot(ep, bs_mse[:k+1], color=BLUE,  lw=1.2, ls="--", alpha=0.65, label="b  (MSE)")
    ax.plot(ep, ws_mae[:k+1], color=AMBER, lw=2.0, label="w  (MAE)")
    ax.plot(ep, bs_mae[:k+1], color=AMBER, lw=1.2, ls="--", alpha=0.65, label="b  (MAE)")
    ax.axhline(w_opt, color=GREEN, lw=1.0, ls=":", label=f"w* = {w_opt:.3f}")
    ax.axhline(b_opt, color=GREEN, lw=0.7, ls=":", alpha=0.45)
    ax.scatter([k], [ws_mse[k]], color=BLUE,  s=60, zorder=6)
    ax.scatter([k], [ws_mae[k]], color=AMBER, s=60, zorder=6)
    ax.set_xlim(0, EPOCHS)
    ax.set_ylim(param_lo, param_hi)
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("Parameter value", fontsize=9)
    ax.set_title("① w, b  →  optimum", fontsize=10, fontweight="bold", pad=5)
    ax.legend(fontsize=7, loc="lower right",
              facecolor="#0f172a", edgecolor=GRID, labelcolor=LIGHT)

    # ── Panel 2: gradient magnitude ───────────────────────────────────────
    ax = axes[1]
    _style(ax)
    ax.plot(ep, grads_mse[:k+1], color=BLUE,  lw=2.0, label="|∂L/∂w|  MSE → 0 ✓")
    ax.plot(ep, grads_mae[:k+1], color=AMBER, lw=2.0, label="|∂L/∂w|  MAE (stays elevated)")
    ax.axhline(0, color=GREEN, lw=1.1, ls=":", label="0  ← minimum signal")
    ax.scatter([k], [grads_mse[k]], color=BLUE,  s=60, zorder=6)
    ax.scatter([k], [grads_mae[k]], color=AMBER, s=60, zorder=6)
    ax.set_xlim(0, EPOCHS)
    ax.set_ylim(-0.01 * grad_hi, grad_hi)
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("|∂L/∂w|", fontsize=9)
    ax.set_title("② Gradient magnitude  (self-brake = MSE only)", fontsize=10,
                 fontweight="bold", pad=5)
    ax.legend(fontsize=7, loc="upper right",
              facecolor="#0f172a", edgecolor=GRID, labelcolor=LIGHT)

    # ── Panel 3: loss ─────────────────────────────────────────────────────
    ax = axes[2]
    _style(ax)
    ax.plot(ep, losses_mse[:k+1], color=BLUE,  lw=2.0, label="MSE loss")
    ax.plot(ep, losses_mae[:k+1], color=AMBER, lw=2.0, label="MAE loss")
    ax.scatter([k], [losses_mse[k]], color=BLUE,  s=60, zorder=6)
    ax.scatter([k], [losses_mae[k]], color=AMBER, s=60, zorder=6)
    ax.set_xlim(0, EPOCHS)
    ax.set_ylim(-0.01 * loss_hi, loss_hi)
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("Loss", fontsize=9)
    ax.set_title("③ Loss descending", fontsize=10, fontweight="bold", pad=5)
    ax.legend(fontsize=7, loc="upper right",
              facecolor="#0f172a", edgecolor=GRID, labelcolor=LIGHT)

    # ── Shared title ──────────────────────────────────────────────────────
    fig.suptitle(
        f"Epoch {k:>2d}/{EPOCHS}   │   "
        f"w (MSE) = {ws_mse[k]:.3f}   "
        f"│grad│ (MSE) = {grads_mse[k]:.4f}   "
        f"│grad│ (MAE) = {grads_mae[k]:.4f}",
        color=LIGHT, fontsize=9.5, fontweight="bold",
    )

    fig.canvas.draw()
    w_px, h_px = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = buf.reshape(h_px, w_px, 4)[..., :3].copy()
    plt.close(fig)
    return img


# ── Render all frames ─────────────────────────────────────────────────────
print("Rendering mse_convergence_3panel frames …")
frames: list[np.ndarray] = []
for k in range(EPOCHS + 1):
    if k % 10 == 0:
        print(f"  epoch {k}/{EPOCHS}")
    frames.append(make_frame(k))

# Hold the final frame for 2 seconds
frames += [frames[-1]] * 24

print(f"Writing {GIF_PATH} …")
imageio.mimsave(GIF_PATH, frames, fps=12, loop=0)
print("Done  →", GIF_PATH)
