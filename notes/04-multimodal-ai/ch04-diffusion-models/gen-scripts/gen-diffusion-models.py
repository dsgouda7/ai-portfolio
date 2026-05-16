from pathlib import Path
"""Generate Diffusion Models.png — 4-panel xkcd-style concept sketch.

Panels:
  (1) Forward noising x0 -> xT
  (2) Reverse denoising xT -> x0
  (3) Noise schedule beta_t / alpha_bar_t
  (4) Training loss (predict noise eps)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.xkcd(scale=0.8, length=100, randomness=2)

fig = plt.figure(figsize=(18, 11), facecolor="white")
fig.suptitle("Diffusion Models — Forward Noise, Reverse Denoise",
             fontsize=20, fontweight="bold", y=0.98)

gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.28,
                      left=0.05, right=0.97, top=0.91, bottom=0.06)
ax_fw = fig.add_subplot(gs[0, 0])
ax_rv = fig.add_subplot(gs[0, 1])
ax_sch = fig.add_subplot(gs[1, 0])
ax_ls = fig.add_subplot(gs[1, 1])

BLUE, ORANGE, GREEN, RED, PURPLE, DARK, GREY = (
    "#2E86C1", "#E67E22", "#27AE60", "#E74C3C", "#8E44AD", "#2C3E50", "#BDC3C7")

rng = np.random.default_rng(1)

def make_thumb(ax, cx, cy, size, noise_level, seed):
    g = 8
    r = np.random.default_rng(seed)
    base = np.zeros((g, g))
    # "fox" shape: oval blob
    for i in range(g):
        for j in range(g):
            d = ((i - 3.5)**2 / 4 + (j - 3.5)**2 / 6)
            base[i, j] = np.exp(-d) * 0.9
    noise = r.normal(0, 1, (g, g))
    img = (1 - noise_level) * base + noise_level * (0.5 + 0.5 * noise)
    img = np.clip(img, 0, 1)
    for i in range(g):
        for j in range(g):
            ax.add_patch(plt.Rectangle(
                (cx - size/2 + j * size/g, cy + size/2 - (i+1) * size/g),
                size/g, size/g,
                facecolor=plt.cm.Oranges(img[i, j]),
                edgecolor="none"))

# ═══════════════════════ PANEL 1 — Forward ═════════════════════════
ax_fw.set_title("1 · Forward — Gradually Add Gaussian Noise",
                fontsize=13, fontweight="bold", color=DARK)
ax_fw.set_xlim(0, 10); ax_fw.set_ylim(0, 10); ax_fw.axis("off")
labels = ["x0", "x_t/4", "x_t/2", "x_3t/4", "xT"]
for k, (lbl, nl) in enumerate(zip(labels, [0.0, 0.25, 0.5, 0.75, 1.0])):
    cx = 1.0 + k * 2.0
    make_thumb(ax_fw, cx, 6.0, 1.4, nl, seed=k + 10)
    ax_fw.add_patch(plt.Rectangle((cx - 0.7, 5.3), 1.4, 1.4,
                                  fill=False, edgecolor=DARK, lw=1.2))
    ax_fw.text(cx, 4.3, lbl, ha="center", fontsize=10,
               color=DARK, family="monospace")
    if k < 4:
        ax_fw.annotate("", xy=(cx + 1.3, 6.0), xytext=(cx + 0.7, 6.0),
                       arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.5))

ax_fw.text(5, 2.8,
           r"$q(x_t|x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}\, x_{t-1},\ \beta_t I)$",
           ha="center", fontsize=13, color=DARK)
ax_fw.text(5, 1.5,
           "Fixed, no learning. Large T -> xT is pure noise.",
           ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 2 — Reverse ═════════════════════════
ax_rv.set_title("2 · Reverse — Learn to Denoise",
                fontsize=13, fontweight="bold", color=DARK)
ax_rv.set_xlim(0, 10); ax_rv.set_ylim(0, 10); ax_rv.axis("off")
labels = ["xT", "x_3t/4", "x_t/2", "x_t/4", "x0"]
for k, (lbl, nl) in enumerate(zip(labels, [1.0, 0.75, 0.5, 0.25, 0.0])):
    cx = 1.0 + k * 2.0
    make_thumb(ax_rv, cx, 6.0, 1.4, nl, seed=k + 10)
    ax_rv.add_patch(plt.Rectangle((cx - 0.7, 5.3), 1.4, 1.4,
                                  fill=False, edgecolor=DARK, lw=1.2))
    ax_rv.text(cx, 4.3, lbl, ha="center", fontsize=10,
               color=DARK, family="monospace")
    if k < 4:
        ax_rv.annotate("", xy=(cx + 1.3, 6.0), xytext=(cx + 0.7, 6.0),
                       arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.5))

ax_rv.text(5, 2.8,
           r"$p_\theta(x_{t-1}|x_t) = \mathcal{N}(\mu_\theta(x_t, t),\ \Sigma_t)$",
           ha="center", fontsize=13, color=DARK)
ax_rv.text(5, 1.5,
           "Network = UNet (or Transformer) predicts noise, conditioned on t.",
           ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 3 — Schedule ═════════════════════════
ax_sch.set_title("3 · Noise Schedule",
                 fontsize=13, fontweight="bold", color=DARK)
T = np.linspace(0, 1000, 200)
beta = 1e-4 + (0.02 - 1e-4) * (T / 1000)
alpha_bar = np.cumprod(1 - beta)

ax_sch.plot(T, beta * 50, "-", color=RED, lw=2.5,
            label=r"$\beta_t$  (scaled x50)")
ax_sch.plot(T, alpha_bar, "-", color=BLUE, lw=2.5,
            label=r"$\bar\alpha_t = \prod(1-\beta_s)$")

ax_sch.set_xlabel("timestep  t", fontsize=10, color=DARK)
ax_sch.set_ylabel("value", fontsize=10, color=DARK)
ax_sch.legend(fontsize=10, loc="center right", framealpha=0.9)
ax_sch.set_ylim(0, 1.1)

ax_sch.text(0.5, -0.25,
            r"$x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \epsilon$"
            "\n-> closed-form sampling at any t.",
            transform=ax_sch.transAxes, ha="center",
            fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 4 — Loss ═════════════════════════════
ax_ls.set_title("4 · Training Loss — Predict the Noise",
                fontsize=13, fontweight="bold", color=DARK)
ax_ls.set_xlim(0, 10); ax_ls.set_ylim(0, 10); ax_ls.axis("off")

steps = [
    (8.6, "sample (x0, t, eps)",                  BLUE),
    (7.0, "build x_t via closed form",            ORANGE),
    (5.4, "predict eps_theta(x_t, t)",            PURPLE),
    (3.8, "loss = || eps - eps_theta ||^2",       RED),
    (2.2, "backprop -> update UNet weights",      GREEN),
]
for y, name, c in steps:
    ax_ls.add_patch(FancyBboxPatch((0.5, y - 0.55), 9.0, 1.1,
                                   boxstyle="round,pad=0.05",
                                   facecolor=c, edgecolor="white", lw=2))
    ax_ls.text(5, y, name, ha="center", va="center",
               color="white", fontweight="bold", fontsize=10.5)
for y1, y2 in [(8.6, 7.0), (7.0, 5.4), (5.4, 3.8), (3.8, 2.2)]:
    ax_ls.annotate("", xy=(5, y2 + 0.55), xytext=(5, y1 - 0.55),
                   arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.3))
ax_ls.text(5, 0.6,
           "Clean L2. No tricky adversarial training. This is why diffusion scales.",
           ha="center", fontsize=9, color="#555", style="italic")

out = str(Path(__file__).resolve().parent.parent / "img" / "Diffusion Models.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out}")
