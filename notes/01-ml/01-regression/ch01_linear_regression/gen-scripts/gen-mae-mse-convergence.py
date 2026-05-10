"""Generate MAE vs MSE gradient descent convergence comparison for Ch.01 §3.9.

Shows side-by-side that:
  - MSE: gradient magnitude → 0 as W approaches W*  →  smooth convergence
  - MAE: gradient is always ±sign(error)             →  oscillates near minimum

Outputs (relative to this script):
    ../img/mae_mse_convergence.gif          — 4-panel animated GIF
    ../img/mae_mse_convergence_table.png    — static raw-numbers table (both losses)

Run:
    python gen_mae_mse_convergence.py
"""
from __future__ import annotations

import os
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.lines import Line2D

# ── palette ───────────────────────────────────────────────────────────────────
DARK        = "#1f2937"
BLUE        = "#2563eb"    # MSE colour
AMBER       = "#d97706"    # MAE colour
LIGHT_BLUE  = "#93c5fd"
GREEN       = "#16a34a"
RED         = "#dc2626"
GREY        = "#cbd5e1"
LIGHT       = "#f8fafc"

HERE    = Path(__file__).parent
OUT_DIR = HERE / ".." / "img"
OUT_DIR.mkdir(parents=True, exist_ok=True)
GIF_PATH   = OUT_DIR / "mae_mse_convergence.gif"
TABLE_PATH = OUT_DIR / "mae_mse_convergence_table.png"

# ── data ─────────────────────────────────────────────────────────────────────
data = fetch_california_housing()
X = data.data[:, [0]]
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_s = scaler.fit_transform(X_train)
x_s = X_s[:, 0]           # 1-D for scalar gradient loop

n = len(x_s)
mu    = float(scaler.mean_[0])
sigma = float(np.sqrt(scaler.var_[0]))

def to_orig(W_sc: float, b_sc: float) -> tuple[float, float]:
    """Convert scaled-space weights back to original-unit weights."""
    w_o = W_sc / sigma
    b_o = b_sc - W_sc * mu / sigma
    return w_o, b_o


# ── run both GD loops together ───────────────────────────────────────────────
MAX_EPOCHS = 300
ALPHA      = 0.01     # same learning rate for both

# Checkpoints to log (used for static table and first-N animation frames)
LOG_EPOCHS = [0, 1, 2, 3, 5, 10, 15, 25, 50, 100, 150, 200, 250, 300]

# Containers
mse_W, mse_b = 0.0, 0.0
mae_W, mae_b = 0.0, 0.0

mse_history: dict[str, list] = {"epoch": [], "W": [], "grad_W": [], "loss": []}
mae_history: dict[str, list] = {"epoch": [], "W": [], "grad_W": [], "loss": []}

# Records also for the raw table
table_rows: list[dict] = []

for epoch in range(MAX_EPOCHS + 1):
    # ── MSE ──────────────────────────────────────────────────────────────────
    y_hat_mse   = x_s * mse_W + mse_b
    err_mse     = y_hat_mse - y_train
    loss_mse    = float(np.mean(err_mse ** 2))
    grad_W_mse  = float((2.0 / n) * np.dot(x_s, err_mse))
    grad_b_mse  = float((2.0 / n) * np.sum(err_mse))

    # ── MAE (subgradient: sign of error, 0 at exact match) ───────────────────
    y_hat_mae   = x_s * mae_W + mae_b
    err_mae     = y_hat_mae - y_train
    loss_mae    = float(np.mean(np.abs(err_mae)))
    grad_W_mae  = float((1.0 / n) * np.dot(x_s, np.sign(err_mae)))
    grad_b_mae  = float((1.0 / n) * np.sum(np.sign(err_mae)))

    if epoch in LOG_EPOCHS:
        w_o_mse, b_o_mse = to_orig(mse_W, mse_b)
        w_o_mae, b_o_mae = to_orig(mae_W, mae_b)
        table_rows.append({
            "epoch":      epoch,
            "mse_W":      round(mse_W, 5),
            "mse_gradW":  round(grad_W_mse, 5),
            "mse_loss":   round(loss_mse, 5),
            "mae_W":      round(mae_W, 5),
            "mae_gradW":  round(grad_W_mae, 5),
            "mae_loss":   round(loss_mae, 5),
        })

    mse_history["epoch"].append(epoch)
    mse_history["W"].append(mse_W)
    mse_history["grad_W"].append(abs(grad_W_mse))
    mse_history["loss"].append(loss_mse)

    mae_history["epoch"].append(epoch)
    mae_history["W"].append(mae_W)
    mae_history["grad_W"].append(abs(grad_W_mae))
    mae_history["loss"].append(loss_mae)

    # ── weight updates ───────────────────────────────────────────────────────
    mse_W -= ALPHA * grad_W_mse
    mse_b -= ALPHA * grad_b_mse
    mae_W -= ALPHA * grad_W_mae
    mae_b -= ALPHA * grad_b_mae


# ── original-unit x for scatter/lines ────────────────────────────────────────
X_train_orig = X_train[:, 0]

epochs_arr   = np.array(mse_history["epoch"])
mse_W_arr    = np.array(mse_history["W"])
mse_loss_arr = np.array(mse_history["loss"])
mse_grad_arr = np.array(mse_history["grad_W"])

mae_W_arr    = np.array(mae_history["W"])
mae_loss_arr = np.array(mae_history["loss"])
mae_grad_arr = np.array(mae_history["grad_W"])


# ── animation frame generator ────────────────────────────────────────────────
DISPLAY_EPOCHS = (
    list(range(0, 20, 1))       # every epoch 0-19
    + list(range(20, 50, 2))    # every 2nd epoch 20-49
    + list(range(50, 151, 5))   # every 5th epoch 50-150
    + list(range(150, 301, 10)) # every 10th epoch 150-300
)
DISPLAY_EPOCHS = sorted(set(DISPLAY_EPOCHS))


def draw_frame(ep_idx: int) -> np.ndarray:
    ep = DISPLAY_EPOCHS[ep_idx]

    # weights at this epoch (original units)
    mse_w_o, mse_b_o = to_orig(mse_W_arr[ep], mse_b)   # b_final ok for visual
    mae_w_o, mae_b_o = to_orig(mae_W_arr[ep], mae_b)
    # use a rough b_o from the epoch
    mse_w_o, _ = to_orig(mse_W_arr[ep], 0)
    mae_w_o, _ = to_orig(mae_W_arr[ep], 0)
    # just use the scaled-space line for visual consistency
    mse_W_now = mse_W_arr[ep]
    mae_W_now = mae_W_arr[ep]

    fig = plt.figure(figsize=(14, 9), facecolor="white")
    fig.suptitle(
        f"MAE vs MSE gradient descent  —  Epoch {ep} / {MAX_EPOCHS}\n"
        "MSE gradient → 0 at minimum (smooth landing)    |    "
        "MAE gradient stays ±constant (oscillates)",
        fontsize=11, fontweight="bold", color=DARK, y=1.0,
    )
    grid = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.32,
                             left=0.08, right=0.97, top=0.91, bottom=0.08)
    ax_scatter  = fig.add_subplot(grid[0, 0])
    ax_W        = fig.add_subplot(grid[0, 1])
    ax_loss     = fig.add_subplot(grid[1, 0])
    ax_grad     = fig.add_subplot(grid[1, 1])

    xs = np.linspace(X_train_orig.min(), X_train_orig.max(), 200)

    # ── panel 1: scatter + live lines ────────────────────────────────────────
    ax_scatter.scatter(X_train_orig, y_train, s=5, alpha=0.25, color=LIGHT_BLUE, zorder=1)
    ax_scatter.plot(xs, mse_W_now * (xs - mu) / sigma + 0,
                    color=BLUE, lw=2.5, label=f"MSE  w={mse_W_now:.3f}", zorder=3)
    ax_scatter.plot(xs, mae_W_now * (xs - mu) / sigma + 0,
                    color=AMBER, lw=2.5, linestyle="--", label=f"MAE  w={mae_W_now:.3f}", zorder=2)
    ax_scatter.set_xlim(X_train_orig.min() - 0.2, X_train_orig.max() + 0.2)
    ax_scatter.set_ylim(-0.2, 5.5)
    ax_scatter.set_title("Regression fit (both models)", fontsize=10, color=DARK)
    ax_scatter.set_xlabel("MedInc (×$10k)", fontsize=9)
    ax_scatter.set_ylabel("MedHouseVal (×$100k)", fontsize=9)
    ax_scatter.legend(fontsize=8.5, frameon=True)
    ax_scatter.grid(alpha=0.18)

    # ── panel 2: W trajectory ────────────────────────────────────────────────
    ax_W.plot(epochs_arr[:ep+1], mse_W_arr[:ep+1], color=BLUE, lw=2, label="MSE")
    ax_W.plot(epochs_arr[:ep+1], mae_W_arr[:ep+1], color=AMBER, lw=2,
              linestyle="--", label="MAE")
    ax_W.scatter([ep], [mse_W_arr[ep]], color=BLUE, s=60, zorder=5)
    ax_W.scatter([ep], [mae_W_arr[ep]], color=AMBER, s=60, zorder=5)
    ax_W.set_xlim(-5, MAX_EPOCHS + 5)
    ax_W.set_title("W over epochs  (MAE oscillates at convergence)", fontsize=10, color=DARK)
    ax_W.set_xlabel("Epoch", fontsize=9)
    ax_W.set_ylabel("W  (scaled)", fontsize=9)
    ax_W.legend(fontsize=8.5)
    ax_W.grid(alpha=0.18)

    # ── panel 3: loss trajectory ──────────────────────────────────────────────
    ax_loss.plot(epochs_arr[:ep+1], mse_loss_arr[:ep+1], color=BLUE, lw=2, label="MSE loss")
    ax_loss.plot(epochs_arr[:ep+1], mae_loss_arr[:ep+1], color=AMBER, lw=2,
                 linestyle="--", label="MAE loss")
    ax_loss.set_xlim(-5, MAX_EPOCHS + 5)
    ax_loss.set_title("Loss over epochs", fontsize=10, color=DARK)
    ax_loss.set_xlabel("Epoch", fontsize=9)
    ax_loss.set_ylabel("Loss value", fontsize=9)
    ax_loss.legend(fontsize=8.5)
    ax_loss.grid(alpha=0.18)

    # ── panel 4: |gradient_W| trajectory ────────────────────────────────────
    ax_grad.plot(epochs_arr[:ep+1], mse_grad_arr[:ep+1], color=BLUE, lw=2, label="|grad_W| MSE")
    ax_grad.plot(epochs_arr[:ep+1], mae_grad_arr[:ep+1], color=AMBER, lw=2,
                 linestyle="--", label="|grad_W| MAE")
    ax_grad.scatter([ep], [mse_grad_arr[ep]], color=BLUE, s=60, zorder=5)
    ax_grad.scatter([ep], [mae_grad_arr[ep]], color=AMBER, s=60, zorder=5)
    ax_grad.set_xlim(-5, MAX_EPOCHS + 5)
    ax_grad.set_title("|Gradient| — MSE → 0, MAE stays elevated", fontsize=10, color=DARK)
    ax_grad.set_xlabel("Epoch", fontsize=9)
    ax_grad.set_ylabel("|∂L/∂W|", fontsize=9)
    ax_grad.legend(fontsize=8.5)
    ax_grad.grid(alpha=0.18)

    # current-epoch annotations
    for ax, mse_val, mae_val, label in [
        (ax_W,    mse_W_arr[ep],    mae_W_arr[ep],    "W"),
        (ax_loss, mse_loss_arr[ep], mae_loss_arr[ep], "loss"),
        (ax_grad, mse_grad_arr[ep], mae_grad_arr[ep], "|grad|"),
    ]:
        ax.axvline(ep, color=GREY, linewidth=0.9, linestyle=":", zorder=0)

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    frame = np.asarray(buf)[:, :, :3]
    plt.close(fig)
    return frame


print(f"Rendering {len(DISPLAY_EPOCHS)} frames for mae_mse_convergence.gif…")
frames = []
for i, ep in enumerate(DISPLAY_EPOCHS):
    if i % 20 == 0:
        print(f"  frame {i+1}/{len(DISPLAY_EPOCHS)}  (epoch {ep})")
    frames.append(draw_frame(i))

# hold last frame longer
durations = [0.18] * (len(frames) - 8) + [1.2] * 8
imageio.mimsave(str(GIF_PATH), frames, duration=durations, loop=0)
print(f"wrote {GIF_PATH}")


# ── static raw-numbers table ─────────────────────────────────────────────────
fig_t, ax_t = plt.subplots(figsize=(14, len(table_rows) * 0.52 + 1.5), facecolor="white")
ax_t.axis("off")

col_labels = [
    "Epoch",
    "MSE  W", "MSE  ∂L/∂W", "MSE  loss",
    "MAE  W", "MAE  ∂L/∂W", "MAE  loss",
]
cell_text = [
    [
        str(r["epoch"]),
        f"{r['mse_W']:.5f}",  f"{r['mse_gradW']:.5f}",  f"{r['mse_loss']:.5f}",
        f"{r['mae_W']:.5f}",  f"{r['mae_gradW']:.5f}",  f"{r['mae_loss']:.5f}",
    ]
    for r in table_rows
]

# colour-code rows where MAE gradient was large (still oscillating)
row_colours = []
for r in table_rows:
    if r["epoch"] > 50 and abs(r["mae_gradW"]) > abs(r["mse_gradW"]) * 5:
        row_colours.append(["#fff7ed"] * 7)   # amber tint: MAE still bouncing
    else:
        row_colours.append(["white"] * 7)

tbl = ax_t.table(
    cellText=cell_text,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
    cellColours=row_colours,
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.2)
tbl.scale(1, 1.55)

# header colours
for j in range(len(col_labels)):
    cell = tbl[0, j]
    cell.set_facecolor(BLUE if j in (1, 2, 3) else AMBER if j in (4, 5, 6) else DARK)
    cell.set_text_props(color="white", fontweight="bold")

ax_t.set_title(
    "MAE vs MSE gradient descent — raw numbers\n"
    "Orange-tinted rows: MAE gradient still large while MSE gradient has collapsed to near-zero",
    fontsize=10.5, fontweight="bold", color=DARK, pad=14,
)

fig_t.tight_layout(pad=1.2)
fig_t.savefig(str(TABLE_PATH), dpi=130, bbox_inches="tight")
plt.close(fig_t)
print(f"wrote {TABLE_PATH}")
