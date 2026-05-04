"""Generate Reference/img/ch16-tensorboard-dashboard.png — annotated mock of the
TensorBoard UI showing the four panels most relevant to diagnosing training:
Scalars, Histograms, Distributions, and Embedding Projector.

This is a schematic illustration (not a screenshot) so it can be rendered
offline and included in the PDF.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

DARK   = "#2C3E50"
BG     = "#F4F6F9"
CARD   = "#FFFFFF"
GREEN  = "#27AE60"
RED    = "#E74C3C"
BLUE   = "#2E86C1"
ORANGE = "#E67E22"
PURPLE = "#8E44AD"
MUTED  = "#7F8C8D"
GRID   = "#E0E4EA"
TBORANGE = "#FF6D00"

rng = np.random.default_rng(3)

fig = plt.figure(figsize=(13.2, 7.4), facecolor=BG)
# No suptitle — the top bar serves as the header, then a small caption below.
fig.text(0.5, 0.985,
         "TensorBoard — the four panels you actually use to debug training",
         ha="center", va="top", fontsize=14, fontweight="bold", color=DARK)

# Top bar (mock header)
head_ax = fig.add_axes([0.01, 0.905, 0.98, 0.048])
head_ax.axis("off")
head_ax.add_patch(mpatches.FancyBboxPatch(
    (0, 0), 1, 1, boxstyle="round,pad=0.01,rounding_size=0.01",
    transform=head_ax.transAxes,
    facecolor="#263238", edgecolor="none"))
head_ax.text(0.015, 0.5, "TensorBoard", color="white",
             fontsize=12, fontweight="bold", va="center",
             transform=head_ax.transAxes)
tabs = [("SCALARS", TBORANGE), ("HISTOGRAMS", "white"),
        ("DISTRIBUTIONS", "white"), ("PROJECTOR", "white"),
        ("IMAGES", MUTED), ("GRAPHS", MUTED), ("PROFILE", MUTED)]
x0 = 0.16
for name, col in tabs:
    head_ax.text(x0, 0.5, name, color=col, fontsize=8.5,
                 fontweight="bold", va="center",
                 transform=head_ax.transAxes)
    x0 += 0.10
head_ax.text(0.965, 0.5, "logs/ch16_*", color="#B0BEC5", fontsize=8,
             ha="right", va="center", transform=head_ax.transAxes)

# Four panels in a 2×2 grid
# Left-top: Scalars. Right-top: Histograms.
# Left-bottom: Distributions. Right-bottom: Embedding Projector.

# ── Panel 1: SCALARS (loss / val_loss) ─────────────────────────────────────
ax1 = fig.add_axes([0.035, 0.49, 0.455, 0.37])
ax1.set_facecolor(CARD)
ax1.set_title("Scalars — train / val loss", fontsize=11,
              fontweight="bold", color=DARK, loc="left", pad=8)

epochs = np.arange(0, 50)
train_loss = 1.6 * np.exp(-epochs / 14) + 0.08 + rng.normal(0, 0.018, size=epochs.shape)
val_loss_a = 1.6 * np.exp(-epochs / 14) + 0.12 + rng.normal(0, 0.02, size=epochs.shape)
# Second run (orange) that overfits — val goes up after epoch ~25
val_loss_b = np.where(
    epochs < 25,
    1.6 * np.exp(-epochs / 13) + 0.12,
    1.6 * np.exp(-25 / 13) + 0.12 + 0.02 * (epochs - 25),
) + rng.normal(0, 0.02, size=epochs.shape)
train_loss_b = 1.6 * np.exp(-epochs / 11) + 0.05 + rng.normal(0, 0.018, size=epochs.shape)

ax1.plot(epochs, train_loss,  color=BLUE,   linewidth=2.0, label="run_A/train")
ax1.plot(epochs, val_loss_a,  color=BLUE,   linewidth=1.6, linestyle="--", label="run_A/val")
ax1.plot(epochs, train_loss_b, color=ORANGE, linewidth=2.0, label="run_B/train")
ax1.plot(epochs, val_loss_b,  color=ORANGE, linewidth=1.6, linestyle="--", label="run_B/val")
ax1.axvline(25, color=RED, linestyle=":", linewidth=1.1, alpha=0.7)
ax1.annotate("overfitting:\nval climbs while\ntrain keeps falling",
             xy=(32, val_loss_b[32]), xytext=(27, 1.15),
             fontsize=8.5, color=RED,
             bbox=dict(boxstyle="round,pad=0.25", facecolor="#FDECEA",
                       edgecolor=RED, linewidth=0.9),
             arrowprops=dict(arrowstyle="->", color=RED, lw=0.9))
ax1.set_xlabel("epoch", fontsize=9, color=MUTED)
ax1.set_ylabel("loss", fontsize=9, color=MUTED)
ax1.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
for spine in ax1.spines.values():
    spine.set_color(GRID)
ax1.legend(loc="upper right", fontsize=8, frameon=True, facecolor="white",
           edgecolor=GRID)

# ── Panel 2: HISTOGRAMS (weight distribution over epochs) ──────────────────
ax2 = fig.add_axes([0.52, 0.49, 0.455, 0.37], projection="3d")
ax2.set_title("Histograms — hidden_1/kernel over epochs",
              fontsize=11, fontweight="bold", color=DARK, loc="left", pad=8)

xs = np.linspace(-0.8, 0.8, 60)
epoch_slices = [1, 5, 10, 20, 30, 45]
cmap = plt.cm.viridis
for i, ep in enumerate(epoch_slices):
    # Start narrow, spread and re-centre slightly as training proceeds
    width  = 0.12 + 0.03 * i
    centre = -0.05 + 0.02 * (i - 2)
    hist = np.exp(-0.5 * ((xs - centre) / width) ** 2)
    hist = hist / hist.max()
    ax2.bar(xs, hist, zs=ep, zdir="y", width=0.025,
            color=cmap(i / (len(epoch_slices) - 1)), alpha=0.85,
            edgecolor="none")
ax2.set_xlabel("weight value", fontsize=8.5, color=MUTED, labelpad=4)
ax2.set_ylabel("epoch", fontsize=8.5, color=MUTED, labelpad=4)
ax2.set_zlabel("density", fontsize=8.5, color=MUTED, labelpad=4)
ax2.set_yticks(epoch_slices)
ax2.tick_params(axis="both", labelsize=7.5, colors=MUTED)
ax2.view_init(elev=24, azim=-62)
ax2.set_facecolor(CARD)
ax2.xaxis.pane.set_edgecolor(GRID); ax2.yaxis.pane.set_edgecolor(GRID)
ax2.zaxis.pane.set_edgecolor(GRID)
ax2.xaxis.pane.set_facecolor((1, 1, 1, 0.0))
ax2.yaxis.pane.set_facecolor((1, 1, 1, 0.0))
ax2.zaxis.pane.set_facecolor((1, 1, 1, 0.0))

# Annotation outside the 3D axes
fig.text(0.965, 0.52,
         "healthy: distribution\nshifts & spreads\nover training\n\nfrozen at ep.1 ⇒\nvanishing gradient",
         fontsize=8.5, color=DARK, ha="right", va="bottom",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF8E1",
                   edgecolor="#F9A825", linewidth=0.9))

# ── Panel 3: DISTRIBUTIONS (percentile ribbons) ────────────────────────────
ax3 = fig.add_axes([0.035, 0.06, 0.455, 0.35])
ax3.set_facecolor(CARD)
ax3.set_title("Distributions — gradients/hidden_2/kernel",
              fontsize=11, fontweight="bold", color=DARK, loc="left", pad=8)

steps = np.arange(0, 500)
median = -0.01 * np.ones_like(steps, dtype=float)
for p, alpha in [(0.50, 0.32), (0.30, 0.22), (0.12, 0.12)]:
    spread = 0.22 * (1 + 0.4 * np.sin(steps / 55)) * (1 - p + 0.2)
    upper  = median + spread
    lower  = median - spread
    ax3.fill_between(steps, lower, upper, color=PURPLE, alpha=alpha,
                     linewidth=0)
ax3.plot(steps, median, color=PURPLE, linewidth=1.6)
ax3.axhline(0, color=MUTED, linewidth=0.7, alpha=0.6, linestyle="--")
ax3.set_xlabel("step", fontsize=9, color=MUTED)
ax3.set_ylabel("gradient value", fontsize=9, color=MUTED)
ax3.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
for spine in ax3.spines.values():
    spine.set_color(GRID)
# Legend-like ribbon key — placed inside panel 3 bounds
ax3.set_ylim(-0.38, 0.38)
ax3.text(250, 0.33,
         "ribbons = 50/70/88 percentiles of gradient distribution  •  "
         "symmetric ≈ 0 = healthy  •  collapse to 0 ⇒ vanishing gradient",
         fontsize=7.8, color=DARK, ha="center", va="center",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#F3E5F5",
                   edgecolor=PURPLE, linewidth=0.9))

# ── Panel 4: EMBEDDING PROJECTOR ───────────────────────────────────────────
ax4 = fig.add_axes([0.52, 0.06, 0.455, 0.35])
ax4.set_facecolor(CARD)
ax4.set_title("Projector — hidden_3 activations (t-SNE, 3 classes)",
              fontsize=11, fontweight="bold", color=DARK, loc="left", pad=8)

# Three well-separated clusters
n_pts = 80
def cluster(centre, spread, col, label):
    x = rng.normal(centre[0], spread, size=n_pts)
    y = rng.normal(centre[1], spread, size=n_pts)
    ax4.scatter(x, y, s=18, c=col, alpha=0.75,
                edgecolors="white", linewidths=0.4, label=label)

cluster((-2.2,  1.6), 0.45, GREEN,  "class 0")
cluster(( 2.4,  1.3), 0.55, BLUE,   "class 1")
cluster(( 0.1, -2.0), 0.50, ORANGE, "class 2")

ax4.set_xticks([]); ax4.set_yticks([])
for spine in ax4.spines.values():
    spine.set_color(GRID)
ax4.legend(loc="upper right", fontsize=8, frameon=True, facecolor="white",
           edgecolor=GRID)
ax4.text(-3.4, -3.1,
         "clean separation in the final hidden layer\n"
         "⇒ representation is semantically meaningful",
         fontsize=8.5, color=DARK,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9",
                   edgecolor=GREEN, linewidth=0.9))
ax4.set_xlim(-4.0, 4.2); ax4.set_ylim(-3.5, 3.0)

out = Path(__file__).resolve().parent / "ch16-tensorboard-dashboard.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"wrote {out}")
