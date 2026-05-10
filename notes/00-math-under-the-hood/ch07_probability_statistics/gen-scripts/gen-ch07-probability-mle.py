"""Ch.7 hero image: Gaussian PDF, CLT histogram, MLE = MSE likelihood surface."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DARK, BLUE, GREEN, PURPLE = "#2C3E50", "#2E86C1", "#27AE60", "#8E44AD"
ORANGE, RED, GREY, GOLD = "#E67E22", "#E74C3C", "#7F8C8D", "#F39C12"

OUT = Path(__file__).parent / "ch07-probability-mle.png"

fig, axes = plt.subplots(1, 3, figsize=(14.8, 5.2))
fig.patch.set_facecolor("white")

# ---------- Panel 1: the three workhorse distributions ----------
ax = axes[0]
# Bernoulli (make/miss): single shot success prob p=0.72
ax.bar([0, 1], [0.28, 0.72], width=0.28, color=[RED, GREEN],
       edgecolor=DARK, alpha=0.85, label="Bernoulli(p=0.72)")
ax.text(0, 0.30, "miss\n0.28", ha="center", fontsize=9, color=DARK)
ax.text(1, 0.74, "make\n0.72", ha="center", fontsize=9, color=DARK)

# Gaussian for release-height uncertainty
x = np.linspace(-2, 3, 400)
mu, sigma = 0.5, 0.5
pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
ax.plot(x, pdf, color=BLUE, lw=2.2, label=r"$\mathcal{N}(\mu=0.5,\sigma=0.5)$")
ax.fill_between(x, 0, pdf, color=BLUE, alpha=0.15)

# Binomial for 10 shots: number of makes
from math import comb
k = np.arange(0, 11)
n_trials, p = 10, 0.72
binom = np.array([comb(n_trials, kk) * p**kk * (1-p)**(n_trials-kk) for kk in k])
ax.bar(k - 0.3, binom, width=0.18, color=PURPLE, alpha=0.75,
       edgecolor=DARK, label="Binomial(n=10, p=0.72)")

ax.set_xlim(-1.2, 11.3); ax.set_ylim(0, 0.95)
ax.set_xlabel("outcome"); ax.set_ylabel("probability / density")
ax.set_title("Three distributions, three free-throw questions",
             fontsize=11, color=DARK)
ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
ax.grid(alpha=0.25)

# ---------- Panel 2: Central Limit Theorem in action ----------
ax = axes[1]
rng = np.random.default_rng(7)
# Underlying distribution: skewed (exponential-like). Sample means → Gaussian.
N_SAMPLES, N_REPS = 50, 10_000
lam = 1.0
pop = rng.exponential(scale=1/lam, size=(N_REPS, N_SAMPLES))
means = pop.mean(axis=1)

ax.hist(means, bins=60, density=True, color=GREEN, alpha=0.65,
        edgecolor=DARK, linewidth=0.4, label=f"{N_REPS:,} sample means\n(n={N_SAMPLES} each)")

# Overlay predicted Gaussian: mean=1/lam, std=(1/lam)/sqrt(n)
mu_clt = 1 / lam
sd_clt = (1 / lam) / np.sqrt(N_SAMPLES)
xx = np.linspace(means.min(), means.max(), 300)
pred = (1 / (sd_clt * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((xx - mu_clt)/sd_clt)**2)
ax.plot(xx, pred, color=RED, lw=2.4,
        label=fr"CLT: $\mathcal{{N}}({mu_clt:.2f}, {sd_clt:.3f}^2)$")

ax.axvline(mu_clt, color=DARK, linestyle=":", lw=1.2)
ax.set_xlabel("sample mean"); ax.set_ylabel("density")
ax.set_title("Central Limit Theorem:\nskewed source → Gaussian means",
             fontsize=11, color=DARK)
ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
ax.grid(alpha=0.25)

# ---------- Panel 3: likelihood surface — MLE lands on MSE fit ----------
ax = axes[2]
# Data: noisy linear relation y = a*x + b + noise.
a_true, b_true, sigma_noise = 1.8, -0.5, 0.4
xs = np.linspace(0, 2, 30)
ys = a_true * xs + b_true + rng.normal(0, sigma_noise, size=xs.shape)

# Grid in (a, b), plot log-likelihood.
a_grid = np.linspace(0.8, 2.8, 160)
b_grid = np.linspace(-1.8, 0.8, 160)
A, B = np.meshgrid(a_grid, b_grid)
# log-lik under Gaussian noise is  -SSE/(2σ²) + const. Drop const.
SSE = np.zeros_like(A)
for xi, yi in zip(xs, ys):
    SSE += (yi - (A * xi + B)) ** 2
loglik = -SSE / (2 * sigma_noise ** 2)

cf = ax.contourf(A, B, loglik, levels=20, cmap="Purples")
ax.contour(A, B, loglik, levels=20, colors=GREY, linewidths=0.4, alpha=0.6)

# MLE = argmin SSE (closed-form OLS)
X_mat = np.column_stack([xs, np.ones_like(xs)])
w_ols, *_ = np.linalg.lstsq(X_mat, ys, rcond=None)
a_hat, b_hat = w_ols
ax.plot(a_hat, b_hat, "*", color=GOLD, markersize=20,
        markeredgecolor=DARK, markeredgewidth=1.2,
        label=fr"MLE = OLS  $(\hat a={a_hat:.2f},\,\hat b={b_hat:.2f})$")
ax.plot(a_true, b_true, "o", color=RED, markersize=10,
        markeredgecolor=DARK, markeredgewidth=1.2,
        label=fr"truth  $(a={a_true},\,b={b_true})$")

ax.set_xlabel("slope  a"); ax.set_ylabel("intercept  b")
ax.set_title("Gaussian log-likelihood = $-\\mathrm{SSE}/(2\\sigma^2)$\n"
             "MLE and least squares coincide",
             fontsize=11, color=DARK)
ax.legend(loc="lower left", fontsize=8, framealpha=0.95)

plt.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}")
