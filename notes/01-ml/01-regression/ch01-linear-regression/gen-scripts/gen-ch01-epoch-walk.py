import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
import os

# Output paths
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'img')
os.makedirs(OUT_DIR, exist_ok=True)
PARABOLA_PNG = os.path.join(OUT_DIR, 'loss_parabola_generated.png')
EPOCH_GIF = os.path.join(OUT_DIR, 'epoch_walk_generated.gif')

# Load data
data = fetch_california_housing()
X = data.data[:, [0]]  # MedInc
y = data.target        # MedHouseVal (units: 100k)

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Manual gradient descent (educational)
W = 0.0
b = 0.0
alpha = 0.01
n = len(X_train_s)

# Epochs of interest
epochs_of_interest = [0,1,2,3,5,10,15,25,50,100,200]
max_epoch = 200
records = []
frames = []

# utility to convert scaled params to original-x params
mu = scaler.mean_[0]
sigma = np.sqrt(scaler.var_[0])
def to_orig(W_scaled, b_scaled):
    w_orig = W_scaled / sigma
    b_orig = b_scaled - (W_scaled * mu / sigma)
    return w_orig, b_orig

# function to compute MSE on training set given original w,b
X_train_orig = X_train[:,0]

def mse_on_train(w, b):
    preds = w * X_train_orig + b
    return np.mean((preds - y_train) ** 2)

# Run GD and capture frames
for epoch in range(max_epoch + 1):
    if epoch in epochs_of_interest:
        w_o, b_o = to_orig(W, b)
        mse = mse_on_train(w_o, b_o)
        rmse = np.sqrt(mse)
        records.append((epoch, w_o, b_o, mse, rmse))

        # Build the three-panel frame
        fig, axs = plt.subplots(1,3, figsize=(12,4))
        # Panel 1: scatter + regression line (original units)
        axs[0].scatter(X_train_orig, y_train, s=6, alpha=0.4)
        xs = np.linspace(X_train_orig.min(), X_train_orig.max(), 200)
        axs[0].plot(xs, w_o * xs + b_o, color='red')
        axs[0].set_title(f'Epoch {epoch}: regression line')
        axs[0].set_xlabel('MedInc (×$10k)')
        axs[0].set_ylabel('MedHouseVal (×$100k)')

        # Panel 2: parabola L(w) with current point
        ws = np.linspace(w_o - 1.0, w_o + 1.0, 400)
        Ls = [mse_on_train(wi, b_o) for wi in ws]
        axs[1].plot(ws, Ls)
        axs[1].scatter([w_o], [mse], color='red')
        axs[1].set_title('MSE vs w (b fixed; residuals drive the slope)')
        axs[1].set_xlabel('w')
        axs[1].set_ylabel('MSE')

        # Panel 3: text readout
        axs[2].axis('off')
        txt = (
            "e = y_hat - y\n"
            "L(e) = e^2\n"
            "dL/de = 2e\n\n"
            f"w = {w_o:.4f}\n"
            f"b = {b_o:.4f}\n"
            f"MSE = {mse:.6f}\n"
            f"RMSE = {rmse:.4f}\n"
        )
        axs[2].text(0.05, 0.6, txt, fontsize=12, family='monospace')
        axs[2].set_title('Metrics')

        # Save frame to buffer
        fname = os.path.join(OUT_DIR, f'frame_epoch_{epoch:03d}.png')
        fig.tight_layout()
        fig.savefig(fname)
        plt.close(fig)
        frames.append(imageio.v2.imread(fname))

    # Gradient step on scaled features
    y_hat = X_train_s[:,0] * W + b
    error = y_hat - y_train
    dW = (2.0 / n) * np.dot(X_train_s[:,0], error)
    db = (2.0 / n) * np.sum(error)
    W -= alpha * dW
    b -= alpha * db

# Save GIF
imageio.mimsave(EPOCH_GIF, frames, duration=0.5)
print('Saved', EPOCH_GIF)

# Clean up individual frame PNGs
for epoch in epochs_of_interest:
    fname = os.path.join(OUT_DIR, f'frame_epoch_{epoch:03d}.png')
    if os.path.exists(fname):
        os.remove(fname)

# Generate dual-panel parabola PNG:
#   Left:  MSE(w) parabola with minimum marked and gradient arrows
#   Right: dL/dw vs w — shows it is a straight line crossing zero at w*
final_w_o, final_b_o = to_orig(W, b)
ws = np.linspace(final_w_o - 0.8, final_w_o + 0.8, 400)
Ls = np.array([mse_on_train(wi, final_b_o) for wi in ws])

# Compute linear derivative analytically: dL/dw = 2a*w + b_w
# a = mean(x^2), b_w = 2*mean(x*(b-y))
a_coef = np.mean(X_train_orig ** 2)
bw_coef = 2.0 * np.mean(X_train_orig * (final_b_o - y_train))
dLdws = 2.0 * a_coef * ws + bw_coef   # linear in w

fig, axs = plt.subplots(1, 2, figsize=(10, 4))

# Left panel — parabola
axs[0].plot(ws, Ls, color='steelblue', linewidth=2)
w_star = final_w_o
L_star = mse_on_train(w_star, final_b_o)
axs[0].scatter([w_star], [L_star], color='green', zorder=5, s=80, label=f'w* = {w_star:.3f}')
# Two example gradient arrows on the left and right of the minimum
for w_ex, dx in [(final_w_o - 0.55, +0.12), (final_w_o + 0.55, -0.12)]:
    L_ex = mse_on_train(w_ex, final_b_o)
    axs[0].annotate('', xy=(w_ex + dx, mse_on_train(w_ex + dx, final_b_o)),
                    xytext=(w_ex, L_ex),
                    arrowprops=dict(arrowstyle='->', color='tomato', lw=1.5))
axs[0].set_title('MSE(w) — quadratic loss surface\n(b fixed at converged value; larger residuals give larger slopes)')
axs[0].set_xlabel('w')
axs[0].set_ylabel('MSE')
axs[0].legend(fontsize=9)

# Right panel — linear derivative
axs[1].plot(ws, dLdws, color='darkorange', linewidth=2, label='dL/dw = 2aw + b_w')
axs[1].axhline(0, color='grey', linewidth=0.8, linestyle='--')
axs[1].scatter([w_star], [0], color='green', zorder=5, s=80, label=f'zero at w* = {w_star:.3f}')
axs[1].set_title('dL/dw vs w — linear derivative\n(zero crossing = minimum; |dL/de| grows with |e|)')
axs[1].set_xlabel('w')
axs[1].set_ylabel('dL/dw')
axs[1].legend(fontsize=9)

fig.suptitle('Quadratic MSE loss → linear derivative → unique minimum', fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(PARABOLA_PNG, bbox_inches='tight')
plt.close(fig)
print('Saved', PARABOLA_PNG)

# Print markdown table
print('\n| Epoch | w | b | MSE | RMSE |')
print('|------:|---:|---:|----:|----:|')
for (e,w_o,b_o,mse,rmse) in records:
    print(f'| {e} | {w_o:.4f} | {b_o:.4f} | {mse:.6f} | {rmse:.4f} |')

# Done

