from pathlib import Path
"""Generate Generative Evaluation.png — 4-panel xkcd-style concept sketch.

Panels:
  (1) FID - distribution overlap of Inception features
  (2) CLIPScore - text-image alignment
  (3) Metric coverage: what each metric sees and misses
  (4) Human evaluation pipeline
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse
import numpy as np

plt.xkcd(scale=0.8, length=100, randomness=2)

fig = plt.figure(figsize=(18, 11), facecolor="white")
fig.suptitle("Generative Evaluation — Measuring What You Made",
             fontsize=20, fontweight="bold", y=0.98)

gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.28,
                      left=0.05, right=0.97, top=0.91, bottom=0.06)
ax_fid = fig.add_subplot(gs[0, 0])
ax_clip = fig.add_subplot(gs[0, 1])
ax_cov = fig.add_subplot(gs[1, 0])
ax_hum = fig.add_subplot(gs[1, 1])

BLUE, ORANGE, GREEN, RED, PURPLE, DARK, GREY = (
    "#2E86C1", "#E67E22", "#27AE60", "#E74C3C", "#8E44AD", "#2C3E50", "#BDC3C7")

# ═══════════════════════ PANEL 1 — FID ═════════════════════════════
ax_fid.set_title(r"1 · FID — Distance Between Feature Gaussians",
                 fontsize=13, fontweight="bold", color=DARK)

x = np.linspace(-4, 6, 400)
real = np.exp(-0.5 * (x - 0)**2) / np.sqrt(2 * np.pi)
gen_close = np.exp(-0.5 * (x - 0.7)**2 / 1.1) / np.sqrt(2.2 * np.pi)
gen_far = 0.55 * np.exp(-0.5 * (x - 3)**2 / 1.8) / np.sqrt(3.6 * np.pi)

ax_fid.fill_between(x, real, alpha=0.4, color=GREEN, label="real features")
ax_fid.fill_between(x, gen_close, alpha=0.4, color=BLUE,
                    label="good model (FID ~5)")
ax_fid.fill_between(x, gen_far, alpha=0.4, color=RED,
                    label="bad model (FID ~60)")

ax_fid.set_xlabel("Inception feature axis (PCA projection)",
                  fontsize=10, color=DARK)
ax_fid.set_ylabel("density", fontsize=10, color=DARK)
ax_fid.legend(fontsize=9, loc="upper right", framealpha=0.9)

ax_fid.text(0.5, -0.25,
            r"FID = $||\mu_r - \mu_g||^2 + Tr(\Sigma_r + \Sigma_g - 2\sqrt{\Sigma_r \Sigma_g})$"
            "\nLower = closer distributions.",
            transform=ax_fid.transAxes, ha="center",
            fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 2 — CLIPScore ═══════════════════════
ax_clip.set_title("2 · CLIPScore — Prompt Alignment",
                  fontsize=13, fontweight="bold", color=DARK)
ax_clip.set_xlim(0, 10); ax_clip.set_ylim(0, 10); ax_clip.axis("off")

# text
ax_clip.add_patch(FancyBboxPatch((0.3, 5.8), 4.0, 1.6,
                                 boxstyle="round,pad=0.05",
                                 facecolor=GREEN, edgecolor="white", lw=2))
ax_clip.text(2.3, 6.6, '"a red fox in snow"',
             ha="center", va="center", color="white",
             fontweight="bold", fontsize=10)

# image (box)
ax_clip.add_patch(FancyBboxPatch((5.7, 5.8), 4.0, 1.6,
                                 boxstyle="round,pad=0.05",
                                 facecolor=BLUE, edgecolor="white", lw=2))
ax_clip.text(7.7, 6.6, "[generated image]",
             ha="center", va="center", color="white",
             fontweight="bold", fontsize=10)

# CLIP encoders
ax_clip.annotate("", xy=(3.5, 4.5), xytext=(2.3, 5.7),
                 arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.4))
ax_clip.annotate("", xy=(6.5, 4.5), xytext=(7.7, 5.7),
                 arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.4))

ax_clip.add_patch(FancyBboxPatch((3.0, 3.3), 4.0, 1.2,
                                 boxstyle="round,pad=0.1",
                                 facecolor=PURPLE, edgecolor="white", lw=2))
ax_clip.text(5, 3.9, "CLIP text + image encoder",
             ha="center", va="center", color="white",
             fontweight="bold", fontsize=10)

# result
ax_clip.add_patch(FancyBboxPatch((3.0, 1.3), 4.0, 1.4,
                                 boxstyle="round,pad=0.1",
                                 facecolor=ORANGE, edgecolor="white", lw=2))
ax_clip.text(5, 2.0, "CLIPScore = max(0, cos(t, i)) * 2.5",
             ha="center", va="center", color="white",
             fontweight="bold", fontsize=10, family="monospace")
ax_clip.annotate("", xy=(5, 1.3), xytext=(5, 3.3),
                 arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.4))

ax_clip.text(5, 0.4,
             "Good for alignment. Blind to quality / realism.",
             ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 3 — Coverage ════════════════════════
ax_cov.set_title("3 · What Each Metric Catches (and Misses)",
                 fontsize=13, fontweight="bold", color=DARK)
ax_cov.set_xlim(0, 10); ax_cov.set_ylim(0, 10); ax_cov.axis("off")

metrics = ["FID", "IS", "CLIP-S", "LPIPS", "human"]
criteria = ["realism", "diversity", "text align", "fine detail", "bias"]
# rows = metric, cols = criteria. Y / N / partial
matrix = [
    ["y", "p", "n", "p", "n"],
    ["p", "y", "n", "n", "n"],
    ["n", "n", "y", "n", "n"],
    ["p", "n", "n", "y", "n"],
    ["y", "y", "y", "y", "y"],
]
col_x = [2.5, 4.0, 5.5, 7.0, 8.5]
for i, c in enumerate(criteria):
    ax_cov.text(col_x[i], 8.5, c, ha="center", fontsize=9.5,
                fontweight="bold", color=DARK, rotation=20)
ax_cov.axhline(8.0, xmin=0.05, xmax=0.95, color=DARK, lw=0.8)

for j, metric in enumerate(metrics):
    y = 7.2 - j * 1.1
    ax_cov.text(1.0, y, metric, fontweight="bold", fontsize=11, color=DARK)
    for i, v in enumerate(matrix[j]):
        color = {"y": GREEN, "p": ORANGE, "n": RED}[v]
        sym = {"y": "YES", "p": "so-so", "n": "NO"}[v]
        ax_cov.add_patch(FancyBboxPatch((col_x[i] - 0.45, y - 0.3),
                                        0.9, 0.6,
                                        boxstyle="round,pad=0.03",
                                        facecolor=color, edgecolor="white", lw=1))
        ax_cov.text(col_x[i], y, sym, ha="center", va="center",
                    color="white", fontweight="bold", fontsize=8)

ax_cov.text(5, 0.6,
            "Use 2-3 metrics + spot-check humans. One number never tells the whole story.",
            ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 4 — Human eval ══════════════════════
ax_hum.set_title("4 · Human Evaluation Pipeline",
                 fontsize=13, fontweight="bold", color=DARK)
ax_hum.set_xlim(0, 10); ax_hum.set_ylim(0, 10); ax_hum.axis("off")

stages = [
    (8.3, "pick prompts\n(realistic mix)",      BLUE),
    (6.7, "sample N images\nfrom each model",   ORANGE),
    (5.1, "pairwise A/B,\nblind + randomised",  PURPLE),
    (3.5, "~3 raters per item\n+ inter-rater kappa", GREEN),
    (1.9, "win-rate + CI\nper prompt bucket",   RED),
]
for y, name, c in stages:
    ax_hum.add_patch(FancyBboxPatch((0.5, y - 0.6), 9.0, 1.2,
                                    boxstyle="round,pad=0.05",
                                    facecolor=c, edgecolor="white", lw=2))
    ax_hum.text(5, y, name, ha="center", va="center",
                color="white", fontweight="bold", fontsize=10)

for y1, y2 in [(8.3, 6.7), (6.7, 5.1), (5.1, 3.5), (3.5, 1.9)]:
    ax_hum.annotate("", xy=(5, y2 + 0.6), xytext=(5, y1 - 0.6),
                    arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.3))

ax_hum.text(5, 0.6,
            "Instruction clarity + calibration tasks are half the battle.",
            ha="center", fontsize=9, color="#555", style="italic")

out = str(Path(__file__).resolve().parent.parent / "img" / "Generative Evaluation.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out}")
