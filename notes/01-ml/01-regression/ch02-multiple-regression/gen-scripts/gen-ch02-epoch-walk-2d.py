import numpy as np
import matplotlib.pyplot as plt
import io
import imageio.v2 as iio
from pathlib import Path

# Dataset
X = np.array([[0.5, 1.0], [1.5, 0.0], [2.0, -1.0]], dtype=float)
y = np.array([1.5, 2.5, 4.0])
alpha = 0.1
w0 = np.array([0.0, 0.0]); b0 = 0.0

# Epoch 1
yhat1 = X @ w0 + b0;  e1 = yhat1 - y;  mse1 = np.mean(e1**2)
XTe1 = X.T @ e1;  gw1 = (2/3)*XTe1;  gb1 = (2/3)*np.sum(e1)
w1 = w0 - alpha*gw1;  b1 = b0 - alpha*gb1

# Epoch 2
yhat2 = X @ w1 + b1;  e2 = yhat2 - y;  mse2 = np.mean(e2**2)
XTe2 = X.T @ e2;  gw2 = (2/3)*XTe2;  gb2 = (2/3)*np.sum(e2)
w2 = w1 - alpha*gw2;  b2 = b1 - alpha*gb2

# Loss contour (pre-computed once)
w1v = np.linspace(-0.3, 2.0, 60);  w2v = np.linspace(-0.8, 0.5, 60)
W1g, W2g = np.meshgrid(w1v, w2v)
Zgrid = np.zeros_like(W1g)
for ci in range(W1g.shape[0]):
    for cj in range(W1g.shape[1]):
        ww = np.array([W1g[ci, cj], W2g[ci, cj]])
        Zgrid[ci, cj] = np.mean((X @ ww + np.mean(y - X @ ww) - y)**2)

BG = '#1a1a2e'
out_path = Path(__file__).parent.parent / "img" / "epoch_walk_2d.gif"
out_path.parent.mkdir(parents=True, exist_ok=True)


def make_frame(title, left_lines, mid_lines, trail_ws, cur_w):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), facecolor=BG)
    for ax, lines, ptitle in zip(axes[:2], [left_lines, mid_lines], ['Forward Pass', 'Gradient / Update']):
        ax.set_facecolor(BG);  ax.axis('off')
        ax.text(0.5, 0.97, ptitle, color='#aaaaff', fontsize=9, ha='center', va='top',
                transform=ax.transAxes, fontweight='bold')
        yp = 0.88
        for ln in lines:
            grey = ln.startswith('Stage') or ln.startswith('Computing')
            ax.text(0.02, yp, ln, color='#777777' if grey else 'white',
                    fontsize=7.5, va='top', transform=ax.transAxes, fontfamily='monospace')
            yp -= 0.105
    ax3 = axes[2];  ax3.set_facecolor(BG)
    ax3.contourf(W1g, W2g, Zgrid, levels=15, cmap='viridis', alpha=0.7)
    ax3.contour(W1g, W2g, Zgrid, levels=15, colors='white', linewidths=0.4, alpha=0.4)
    trail = np.array(trail_ws)
    if len(trail) > 1:
        ax3.plot(trail[:, 0], trail[:, 1], 'w-', lw=1.2, alpha=0.8)
        ax3.scatter(trail[:-1, 0], trail[:-1, 1], color='white', s=30, zorder=4)
        ax3.annotate('', xy=trail[-1], xytext=trail[-2],
                     arrowprops=dict(arrowstyle='->', color='white', lw=1.2), zorder=6)
    ax3.scatter([cur_w[0]], [cur_w[1]], color='red', s=150, zorder=5)
    ax3.set_xlabel('w₁', color='white', fontsize=9)
    ax3.set_ylabel('w₂', color='white', fontsize=9)
    ax3.tick_params(colors='white', labelsize=7)
    ax3.set_title('Loss Surface', color='white', fontsize=9)
    for sp in ax3.spines.values():
        sp.set_edgecolor('white')
    fig.suptitle(title, color='white', fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor=BG)
    plt.close(fig);  buf.seek(0)
    return iio.imread(buf)


def fwd_rows(yh, e, mse):
    hdr = f"{'i':>2} {'x1':>5} {'x2':>5} {'y':>5} {'ŷ':>7} {'e':>7} {'e²':>7}"
    rows = [hdr, '-' * 47]
    for k in range(3):
        rows.append(f"{k+1:>2} {X[k,0]:>5.1f} {X[k,1]:>5.1f} {y[k]:>5.1f}"
                    f" {yh[k]:>7.3f} {e[k]:>7.3f} {e[k]**2:>7.3f}")
    rows.append(f"  MSE = {mse:.3f}")
    return rows


fT1 = fwd_rows(yhat1, e1, mse1)
fT2 = fwd_rows(yhat2, e2, mse2)

xte1 = [
    f"Xᵀe row 1: (0.5)({e1[0]:.1f})+(1.5)({e1[1]:.1f})+(2.0)({e1[2]:.1f}) = {XTe1[0]:.1f}",
    f"Xᵀe row 2: (1.0)({e1[0]:.1f})+(0.0)({e1[1]:.1f})+(-1.0)({e1[2]:.1f}) = {XTe1[1]:+.1f}",
    f"∂L/∂w = (2/3)×[{XTe1[0]:.2f}, {XTe1[1]:.2f}] = [{gw1[0]:.2f}, {gw1[1]:+.2f}]",
    f"∂L/∂b = (2/3)×({np.sum(e1):.1f}) = {gb1:.2f}",
]
upd1 = [
    f"w ← [0.0, 0.0] − 0.1×[{gw1[0]:.2f}, {gw1[1]:+.2f}]",
    f"  = [{w1[0]:.3f}, {w1[1]:.3f}]",
    f"b ← 0.0 − 0.1×{gb1:.3f} = {b1:.3f}",
]
sum1 = [f"w = [{w1[0]:.3f}, {w1[1]:.3f}]", f"b = {b1:.3f}", f"MSE = {mse2:.3f}"]
xte2 = [
    f"Xᵀe row 1: (0.5)({e2[0]:.3f})+(1.5)({e2[1]:.3f})+(2.0)({e2[2]:.3f}) = {XTe2[0]:.3f}",
    f"Xᵀe row 2: (1.0)({e2[0]:.3f})+(0.0)({e2[1]:.3f})+(-1.0)({e2[2]:.3f}) = {XTe2[1]:.3f}",
    f"∂L/∂w = (2/3)×[{XTe2[0]:.3f}, {XTe2[1]:.3f}] = [{gw2[0]:.3f}, {gw2[1]:.3f}]",
    f"∂L/∂b = (2/3)×({np.sum(e2):.3f}) = {gb2:.3f}",
]
upd2 = [
    f"w ← [{w1[0]:.3f}, {w1[1]:.3f}] − 0.1×[{gw2[0]:.3f}, {gw2[1]:.3f}]",
    f"  = [{w2[0]:.3f}, {w2[1]:.3f}]",
    f"b ← {b1:.3f} − 0.1×{gb2:.3f} = {b2:.3f}",
]
sum2 = [f"w = [{w2[0]:.3f}, {w2[1]:.3f}]", f"b = {b2:.3f}"]

frames = [
    make_frame("Epoch 1, Stage 1: Forward Pass",   fT1,  ["Stage 2 pending..."],    [w0], w0),
    make_frame("Epoch 1, Stage 2: Compute Xᵀe",   fT1,  xte1,                     [w0], w0),
    make_frame("Epoch 1, Stage 3: Update Weights", sum1, upd1,                     [w0, w1], w1),
    make_frame("Epoch 2, Stage 1: Forward Pass",   fT2,  ["Computing new Xᵀe..."], [w0, w1], w1),
    make_frame("Epoch 2, Stage 2: Compute Xᵀe",   fT2,  xte2,                     [w0, w1], w1),
    make_frame("Epoch 2, Stage 3: Update Weights", sum2, upd2,                     [w0, w1, w2], w2),
]

iio.mimsave(str(out_path), frames, fps=1, loop=0)
print(f"Saved → {out_path}")
