"""Generate Reference/img/ch15-prior-posterior.png — MAP with Gaussian vs Laplace prior.

Two-panel figure showing how the choice of prior shape induces either L2 (Gaussian)
or L1 (Laplace) regularisation when maximising the posterior.

Panel A: prior densities (Gaussian vs Laplace) over weight w.
Panel B: the resulting regularisation term as a function of w — quadratic for
         Gaussian prior (L2), absolute for Laplace prior (L1).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BLUE   = "#2E86C1"
ORANGE = "#E67E22"
DARK   = "#2C3E50"
GREY   = "#7F8C8D"
GREEN  = "#27AE60"
RED    = "#E74C3C"
MLE    = "#8E5A9E"

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), facecolor="white")
fig.suptitle(
    r"MAP = MLE + prior: Gaussian prior $\Rightarrow$ L2,  Laplace prior $\Rightarrow$ L1",
    fontsize=13, fontweight="bold", color=DARK, y=0.99,
)

w = np.linspace(-4, 4, 1000)

# ── LEFT: prior densities ─────────────────────────────────────────────────
ax = axes[0]
sigma = 1.0
gauss = np.exp(-0.5 * (w / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
b = 1.0
laplace = np.exp(-np.abs(w) / b) / (2 * b)

ax.plot(w, gauss, color=BLUE, linewidth=2.6, label=r"Gaussian prior  $w\sim\mathcal{N}(0,\sigma^2)$")
ax.fill_between(w, 0, gauss, color=BLUE, alpha=0.10)
ax.plot(w, laplace, color=ORANGE, linewidth=2.6, label=r"Laplace prior  $w\sim\mathrm{Lap}(0,b)$")
ax.fill_between(w, 0, laplace, color=ORANGE, alpha=0.10)

ax.axvline(0, color=DARK, linewidth=0.8, alpha=0.6)
ax.set_title("Prior density  $p(w)$", fontsize=11, color=DARK)
ax.set_xlabel(r"weight $w$", fontsize=11, color=DARK)
ax.set_ylabel(r"$p(w)$", fontsize=11, color=DARK)
ax.grid(True, alpha=0.25)
ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

ax.annotate("Laplace: sharp peak at 0\n→ pushes weights exactly to 0 (sparsity)",
            xy=(0.02, laplace[np.argmin(np.abs(w - 0.02))]),
            xytext=(-3.85, 0.12),
            fontsize=9, color=ORANGE,
            bbox=dict(boxstyle="round,pad=0.32", facecolor="#FFF3E0",
                      edgecolor=ORANGE, linewidth=1.0),
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.0,
                            connectionstyle="arc3,rad=-0.15"))
ax.annotate("Gaussian: smooth bell\n→ shrinks all weights proportionally",
            xy=(-0.9, gauss[np.argmin(np.abs(w - (-0.9)))]),
            xytext=(1.25, 0.30),
            fontsize=9, color=BLUE,
            bbox=dict(boxstyle="round,pad=0.32", facecolor="#E3F2FD",
                      edgecolor=BLUE, linewidth=1.0),
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0,
                            connectionstyle="arc3,rad=0.15"))

# ── RIGHT: corresponding regularisation terms −log p(w) ───────────────────
ax = axes[1]
# −log Gaussian ∝ w² / (2σ²)  → L2 regulariser, shape w²
# −log Laplace  ∝ |w| / b     → L1 regulariser, shape |w|
l2 = 0.5 * w ** 2
l1 = np.abs(w)

ax.plot(w, l2, color=BLUE, linewidth=2.6,
        label=r"$-\log p_{\mathcal{N}}(w) \;\propto\; w^2$   (L2)")
ax.plot(w, l1, color=ORANGE, linewidth=2.6,
        label=r"$-\log p_{\mathrm{Lap}}(w) \;\propto\; |w|$   (L1)")
ax.fill_between(w, 0, l2, color=BLUE, alpha=0.06)
ax.fill_between(w, 0, l1, color=ORANGE, alpha=0.08)

ax.axvline(0, color=DARK, linewidth=0.8, alpha=0.6)
ax.set_title(r"Regularisation term  $-\log p(w)$", fontsize=11, color=DARK)
ax.set_xlabel(r"weight $w$", fontsize=11, color=DARK)
ax.set_ylabel(r"penalty added to NLL", fontsize=11, color=DARK)
ax.grid(True, alpha=0.25)
ax.legend(loc="upper center", fontsize=9, framealpha=0.95)
ax.set_ylim(0, 4.5)

ax.annotate(
    "|w| is non-differentiable at 0\n→ gradient sign is ±1 → drives w to 0",
    xy=(0.0, 0.0),
    xytext=(1.05, 1.4),
    fontsize=9, color=ORANGE,
    bbox=dict(boxstyle="round,pad=0.32", facecolor="#FFF3E0",
              edgecolor=ORANGE, linewidth=1.0),
    arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.0,
                    connectionstyle="arc3,rad=-0.2"),
)
ax.annotate(
    "w² grows quadratically\n→ gradient ∝ w → smooth shrinkage",
    xy=(-2.2, 0.5 * 2.2 ** 2),
    xytext=(-3.95, 0.35),
    fontsize=9, color=BLUE,
    bbox=dict(boxstyle="round,pad=0.32", facecolor="#E3F2FD",
              edgecolor=BLUE, linewidth=1.0),
    arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0,
                    connectionstyle="arc3,rad=-0.2"),
)

# Footer equation showing MAP (plain braces — matplotlib mathtext lacks \Big)
fig.text(
    0.5, -0.02,
    r"$\hat{\theta}_{\mathrm{MAP}} \;=\; \arg\min_\theta \;\{\; "
    r"-\sum_i \log p(y_i\mid x_i;\theta) \;+\; \lambda\,[-\log p(\theta)] \;\}$"
    "\n"
    r"MLE term (MSE or BCE)   +   prior term (L2 if Gaussian, L1 if Laplace)",
    ha="center", va="top", fontsize=10, color=DARK,
)

fig.tight_layout(rect=(0, 0.03, 1, 0.95))
out = Path(__file__).resolve().parent / "ch15-prior-posterior.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")
