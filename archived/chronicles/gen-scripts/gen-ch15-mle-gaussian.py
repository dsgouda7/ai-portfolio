"""Generate Reference/img/ch15-mle-gaussian.png — Gaussian likelihood surface.

Two-panel figure:
  (left)  Gaussian likelihoods over (μ, σ²) parameter space, with the MLE peak
          marked at the sample-mean / sample-variance point.
  (right) Cross-section showing −log-likelihood as a convex bowl over μ at
          σ² fixed — the NLL that becomes MSE once constants drop.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BLUE  = "#2E86C1"
RED   = "#E74C3C"
GREEN = "#27AE60"
DARK  = "#2C3E50"
GOLD  = "#F39C12"
MLE   = "#8E5A9E"

rng = np.random.default_rng(7)
true_mu, true_sigma = 2.0, 0.8
n = 80
data = rng.normal(true_mu, true_sigma, size=n)
xbar = float(data.mean())
s2   = float(data.var(ddof=0))

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), facecolor="white")
fig.suptitle(r"MLE — Gaussian likelihood $\mathcal{L}(\mu,\sigma^2)$ peaks at $(\bar{x}, s^2)$",
             fontsize=13, fontweight="bold", color=DARK, y=0.99)

# ── LEFT: log-likelihood surface over (μ, σ²) ──────────────────────────────
ax = axes[0]
mus    = np.linspace(xbar - 1.4, xbar + 1.4, 220)
sigmas = np.linspace(max(0.15, s2 ** 0.5 - 0.6), s2 ** 0.5 + 0.7, 220)
MU, SG = np.meshgrid(mus, sigmas)
# Log-likelihood per (μ, σ): sum over data
# log p(x|μ,σ) = -0.5*log(2π σ²) - (x-μ)² / (2 σ²)
sq = (data[:, None, None] - MU[None, :, :]) ** 2
LL = (-0.5 * np.log(2 * np.pi * SG ** 2) - sq / (2 * SG ** 2)).sum(axis=0)
# Shift so max is 0 for clean contour levels
LL -= LL.max()

cf = ax.contourf(MU, SG, LL, levels=20, cmap="viridis", alpha=0.92)
ax.contour(MU, SG, LL, levels=10, colors="white", linewidths=0.5, alpha=0.5)
ax.plot(xbar, s2 ** 0.5, marker="*", markersize=22,
        markerfacecolor=GOLD, markeredgecolor="white", markeredgewidth=1.5,
        zorder=5, label=r"MLE: $(\bar{x}, \sqrt{s^2})$")
ax.axvline(true_mu, color="white", linestyle=":", linewidth=1, alpha=0.6)
ax.axhline(true_sigma, color="white", linestyle=":", linewidth=1, alpha=0.6)
ax.text(true_mu + 0.03, sigmas[-1] - 0.05, r"true $\mu$", color="white",
        fontsize=9, alpha=0.9)
ax.text(mus[-1] - 0.28, true_sigma + 0.02, r"true $\sigma$", color="white",
        fontsize=9, alpha=0.9)

ax.set_xlabel(r"$\mu$  (mean parameter)", fontsize=11, color=DARK)
ax.set_ylabel(r"$\sigma$  (std parameter)", fontsize=11, color=DARK)
ax.set_title(r"Log-likelihood surface  $\ell(\mu,\sigma) = \sum_i \log p(x_i\mid\mu,\sigma)$",
             fontsize=10, color=DARK)
ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(r"$\ell - \ell_{\max}$", fontsize=9, color=DARK)

# ── RIGHT: NLL bowl cross-section over μ at σ fixed → derives MSE ──────────
ax = axes[1]
mu_line = np.linspace(xbar - 1.6, xbar + 1.6, 400)
sigma_fix = s2 ** 0.5
# NLL(μ) = (1/(2σ²)) * Σ (x - μ)² + const — a parabola in μ, minimum at x̄
nll = ((data[:, None] - mu_line[None, :]) ** 2).sum(axis=0) / (2 * sigma_fix ** 2)
nll -= nll.min()
ax.plot(mu_line, nll, color=MLE, linewidth=2.6,
        label=r"$-\ell(\mu)$  (NLL, $\sigma$ fixed)")
# Shade MSE-equivalent as parabola
ax.fill_between(mu_line, 0, nll, color=MLE, alpha=0.08)
# Mark the minimum
ax.axvline(xbar, color=GOLD, linewidth=1.4, linestyle="--", alpha=0.9)
ax.plot(xbar, 0, marker="*", markersize=20, markerfacecolor=GOLD,
        markeredgecolor="white", markeredgewidth=1.4, zorder=5)
ax.text(xbar + 0.04, nll.max() * 0.06,
        r"$\hat{\mu}_{\mathrm{MLE}} = \bar{x}$", color=DARK, fontsize=10)

# Annotate the algebraic reduction to MSE
ax.annotate(
    r"minimising $-\ell(\mu) = \dfrac{1}{2\sigma^2}\sum_i (x_i-\mu)^2$"
    "\n"
    r"$\;\Longleftrightarrow\;$ minimising $\mathrm{MSE}(\mu)$",
    xy=(xbar + 0.85, nll[np.argmin(np.abs(mu_line - (xbar + 0.85)))]),
    xytext=(xbar + 0.35, nll.max() * 0.72),
    fontsize=10, color=DARK,
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF8E1",
              edgecolor=GOLD, linewidth=1.1),
    arrowprops=dict(arrowstyle="->", color=DARK, lw=1.1,
                    connectionstyle="arc3,rad=-0.2"),
)

ax.set_xlabel(r"$\mu$", fontsize=11, color=DARK)
ax.set_ylabel(r"$-\ell(\mu)$  (negative log-likelihood)", fontsize=11, color=DARK)
ax.set_title("Cross-section: NLL is a parabola in μ — reduces to MSE",
             fontsize=10, color=DARK)
ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.25)
ax.set_xlim(mu_line[0], mu_line[-1])
ax.set_ylim(0, nll.max() * 1.05)

fig.tight_layout(rect=(0, 0, 1, 0.95))
out = Path(__file__).resolve().parent / "ch15-mle-gaussian.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")
