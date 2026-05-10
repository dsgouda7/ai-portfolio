"""
gen_ch03_text_to_graphs.py
Convert ASCII-art scatter diagrams in ch03 README to actual matplotlib figures.

Outputs (saved to ../img/):
  ch03-mi-case1-ushape.png          — Case 1: U-shaped relationship
  ch03-mi-case2-threshold.png       — Case 2: threshold / step relationship
  ch03-broken-ruler-parabola.png    — y = x² parabola (Broken Ruler)
  ch03-univariate-scatter.png       — MedInc vs HouseAge comparison
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DARK_BG  = "#1a1a2e"
TEXT_CLR = "#e2e8f0"
TEAL     = "#4ecdc4"
GOLD     = "#f7b731"
RED      = "#e94560"
PURPLE   = "#a29bfe"
OUT      = Path("../img")

np.random.seed(42)


def style_ax(ax, xlabel="", ylabel="", title=""):
    ax.set_facecolor(DARK_BG)
    ax.set_title(title, color=TEXT_CLR, fontsize=10, fontweight="bold", pad=8)
    ax.tick_params(colors=TEXT_CLR, labelsize=8)
    ax.set_xlabel(xlabel, color=TEXT_CLR, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT_CLR, fontsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333355")


def annotate_box(ax, x, y, text, color, coords="axes fraction"):
    ax.text(x, y, text, transform=ax.transAxes,
            color=color, fontsize=8.5, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e",
                      edgecolor=color, alpha=0.92))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Case 1 — U-shaped relationship
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 4.2), facecolor=DARK_BG)
style_ax(ax, xlabel="x  (e.g. dose level)",
         ylabel="y  (e.g. response)",
         title="Case 1 — U-shaped Relationship")

x_u = np.random.uniform(0.2, 5.8, 100)
y_u = -(x_u - 3)**2 + 9 + np.random.normal(0, 0.7, 100)

ax.scatter(x_u, y_u, s=28, color=TEAL, alpha=0.72, lw=0)

x_sm = np.linspace(0, 6, 300)
ax.plot(x_sm, -(x_sm - 3)**2 + 9, color=PURPLE, lw=1.8, alpha=0.55, linestyle="--")

annotate_box(ax, 0.04, 0.97,
             "Pearson ρ ≈ 0\n(symmetric deviations cancel)", RED)
annotate_box(ax, 0.04, 0.70,
             "MI > 0\n(x predicts y strongly)", TEAL)

plt.tight_layout()
out1 = OUT / "ch03-mi-case1-ushape.png"
plt.savefig(out1, dpi=130, facecolor=DARK_BG, bbox_inches="tight")
plt.close()
print(f"Saved {out1}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Case 2 — Threshold / step relationship
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 4.2), facecolor=DARK_BG)
style_ax(ax, xlabel="x  (e.g. income level)",
         ylabel="y  (e.g. house price)",
         title="Case 2 — Threshold / Step Relationship")

x_th = np.random.uniform(0, 10, 160)
y_th = np.where(x_th < 5, 1.5, 4.1) + np.random.normal(0, 0.32, 160)

ax.scatter(x_th, y_th, s=28, color=GOLD, alpha=0.72, lw=0)
ax.axvline(x=5, color=RED, linestyle="--", lw=1.8, alpha=0.85)
ax.text(5.15, 2.6, "threshold", color=RED, fontsize=8.5, va="center")

annotate_box(ax, 0.04, 0.97,
             "Pearson ρ = moderate\n(partial signal only)", RED)
annotate_box(ax, 0.55, 0.32,
             "MI = high\n(captures full jump)", GOLD)

plt.tight_layout()
out2 = OUT / "ch03-mi-case2-threshold.png"
plt.savefig(out2, dpi=130, facecolor=DARK_BG, bbox_inches="tight")
plt.close()
print(f"Saved {out2}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Broken Ruler — y = x² parabola
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 4.5), facecolor=DARK_BG)
style_ax(ax, xlabel="x", ylabel="y = x²",
         title="The Broken Ruler: $y = x^2$")

x_p = np.linspace(-3, 3, 300)
y_p = x_p**2

ax.scatter(x_p[::2], y_p[::2], s=20, color=TEAL, alpha=0.65, lw=0)
ax.plot(x_p, y_p, color=PURPLE, lw=1.8, alpha=0.50)

# Show mean line to illustrate why deviations cancel
ax.axhline(y=np.mean(y_p), color=RED, linestyle=":", lw=1.4, alpha=0.6)
ax.text(2.3, np.mean(y_p) + 0.15, "mean(y)", color=RED, fontsize=7.5, alpha=0.85)

annotate_box(ax, 0.04, 0.97, "Pearson ρ = 0.000\n→ ruler: no relationship", RED)
annotate_box(ax, 0.04, 0.73, "MI score = 0.95+\n→ detective: strong link", TEAL)

ax.text(0.50, 0.04,
        "Left and right deviations from mean cancel — Pearson reads zero.\n"
        "But every x maps to a unique y — MI sees the full pattern.",
        transform=ax.transAxes, color=TEXT_CLR, fontsize=7.2,
        ha="center", va="bottom", style="italic", alpha=0.80)

plt.tight_layout()
out3 = OUT / "ch03-broken-ruler-parabola.png"
plt.savefig(out3, dpi=130, facecolor=DARK_BG, bbox_inches="tight")
plt.close()
print(f"Saved {out3}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Univariate scatter — MedInc vs HouseAge side-by-side
# ─────────────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=DARK_BG)
fig.patch.set_facecolor(DARK_BG)

n = 400

# Panel 1: strong positive correlation (ρ ≈ 0.69)
x_inc = np.random.gamma(3, 1.2, n)
x_inc = np.clip(x_inc, 0.5, 12)
y_inc = 0.42 * x_inc + 0.5 + np.random.normal(0, 0.65, n)
y_inc = np.clip(y_inc, 0.5, 5.0)

style_ax(ax1, xlabel="MedInc  (low → high)",
         ylabel="MedHouseVal  (×$100k)",
         title="MedInc → MedHouseVal     ρ = 0.69, R² = 0.47")
ax1.scatter(x_inc, y_inc, s=14, color=TEAL, alpha=0.38, lw=0)
m, b = np.polyfit(x_inc, y_inc, 1)
x_fit = np.linspace(x_inc.min(), x_inc.max(), 200)
ax1.plot(x_fit, m * x_fit + b, color=PURPLE, lw=2.0, alpha=0.90)
annotate_box(ax1, 0.05, 0.97, "Tight upward band\n→ R² = 0.47", TEAL)

# Panel 2: no correlation (ρ ≈ 0)
x_age = np.random.uniform(1, 52, n)
y_age = np.clip(np.random.normal(2.0, 1.0, n), 0.5, 5.0)

style_ax(ax2, xlabel="HouseAge  (years)",
         ylabel="MedHouseVal  (×$100k)",
         title="HouseAge → MedHouseVal     ρ ≈ 0, R² ≈ 0")
ax2.scatter(x_age, y_age, s=14, color=GOLD, alpha=0.38, lw=0)
annotate_box(ax2, 0.05, 0.97, "Uniform cloud\n→ R² ≈ 0", GOLD)

fig.subplots_adjust(wspace=0.38)
out4 = OUT / "ch03-univariate-scatter.png"
plt.savefig(out4, dpi=130, facecolor=DARK_BG, bbox_inches="tight")
plt.close()
print(f"Saved {out4}")
