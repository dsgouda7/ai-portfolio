from pathlib import Path
"""Generate Latent Diffusion.png — 4-panel xkcd-style concept sketch.

Panels:
  (1) Full pipeline: VAE encode -> latent diffuse -> VAE decode
  (2) Pixel vs latent space shape comparison (8x compression)
  (3) Compute savings bar chart
  (4) VAE reconstruction fidelity tradeoff
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.xkcd(scale=0.8, length=100, randomness=2)

fig = plt.figure(figsize=(18, 11), facecolor="white")
fig.suptitle("Latent Diffusion — Compress, Diffuse, Decode",
             fontsize=20, fontweight="bold", y=0.98)

gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.28,
                      left=0.05, right=0.97, top=0.91, bottom=0.06)
ax_pipe = fig.add_subplot(gs[0, 0])
ax_sh = fig.add_subplot(gs[0, 1])
ax_cost = fig.add_subplot(gs[1, 0])
ax_rec = fig.add_subplot(gs[1, 1])

BLUE, ORANGE, GREEN, RED, PURPLE, DARK, GREY = (
    "#2E86C1", "#E67E22", "#27AE60", "#E74C3C", "#8E44AD", "#2C3E50", "#BDC3C7")

# ═══════════════════════ PANEL 1 — Pipeline ════════════════════════
ax_pipe.set_title("1 · Pipeline — Diffusion in Latent Space",
                  fontsize=13, fontweight="bold", color=DARK)
ax_pipe.set_xlim(0, 10); ax_pipe.set_ylim(0, 10); ax_pipe.axis("off")

stages = [
    (1.2, "image\n512x512x3",       BLUE),
    (3.4, "VAE\nencoder",           ORANGE),
    (5.6, "latent\n64x64x4",        PURPLE),
    (7.8, "VAE\ndecoder",           ORANGE),
    (9.2, "image",                  BLUE),
]
for x, name, c in stages:
    ax_pipe.add_patch(FancyBboxPatch((x - 0.85, 5.5), 1.7, 2.0,
                                     boxstyle="round,pad=0.05",
                                     facecolor=c, edgecolor="white", lw=2))
    ax_pipe.text(x, 6.5, name, ha="center", va="center",
                 color="white", fontweight="bold", fontsize=9.5)

for i in range(4):
    x1, x2 = stages[i][0] + 0.9, stages[i + 1][0] - 0.9
    ax_pipe.annotate("", xy=(x2, 6.5), xytext=(x1, 6.5),
                     arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.4))

# Diffusion loop on the latent
ax_pipe.add_patch(FancyBboxPatch((4.3, 2.6), 2.6, 1.8,
                                 boxstyle="round,pad=0.1",
                                 facecolor=GREEN, edgecolor="white", lw=2))
ax_pipe.text(5.6, 3.5, "UNet loop\n(denoise latent)",
             ha="center", va="center",
             color="white", fontweight="bold", fontsize=10)
ax_pipe.annotate("", xy=(5.6, 4.3), xytext=(5.6, 5.4),
                 arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1.5))

ax_pipe.text(5, 1.2,
             "Train the VAE ONCE. Diffuse in the 8x smaller latent.\n"
             "Decoder renders pixels on demand.",
             ha="center", fontsize=9.5, color="#555", style="italic")

# ═══════════════════════ PANEL 2 — Shape compare ═══════════════════
ax_sh.set_title("2 · Pixel vs Latent Shape",
                fontsize=13, fontweight="bold", color=DARK)
ax_sh.set_xlim(0, 10); ax_sh.set_ylim(0, 10); ax_sh.axis("off")

# Pixel grid
ax_sh.text(2.5, 9.3, "pixel space", ha="center",
           fontweight="bold", color=BLUE, fontsize=11)
for i in range(16):
    for j in range(16):
        ax_sh.add_patch(plt.Rectangle((0.2 + j*0.24, 8.4 - i*0.24),
                                      0.22, 0.22,
                                      facecolor=plt.cm.Blues(0.3 + 0.03*((i+j) % 8)),
                                      edgecolor="none"))
ax_sh.text(2.5, 4.1, "512 x 512 x 3\n= 786,432 values",
           ha="center", fontsize=10, color=DARK, family="monospace")

# Latent grid
ax_sh.text(7.5, 9.3, "latent space", ha="center",
           fontweight="bold", color=PURPLE, fontsize=11)
for i in range(8):
    for j in range(8):
        ax_sh.add_patch(plt.Rectangle((5.4 + j*0.45, 7.95 - i*0.45),
                                      0.42, 0.42,
                                      facecolor=plt.cm.Purples(0.4 + 0.06*((i+j) % 6)),
                                      edgecolor="none"))
ax_sh.text(7.5, 4.1, "64 x 64 x 4\n= 16,384 values",
           ha="center", fontsize=10, color=DARK, family="monospace")

ax_sh.text(5, 2.4,
           "~48x fewer values per step.",
           ha="center", color=GREEN, fontweight="bold", fontsize=12)
ax_sh.text(5, 1.2,
           "Encoder throws out high-freq noise and JPEG grain — things diffusion doesn't need.",
           ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 3 — Compute savings ═════════════════
ax_cost.set_title("3 · Compute Savings per Training Step",
                  fontsize=13, fontweight="bold", color=DARK)
labels = ["pixel diffusion\n(512^2)", "latent diffusion\n(64^2)"]
flops  = [100, 6]
mem    = [100, 12]

x = np.arange(len(labels))
w = 0.35
ax_cost.bar(x - w/2, flops, w, color=BLUE,   edgecolor="white",
            label="FLOPs (rel)")
ax_cost.bar(x + w/2, mem,   w, color=ORANGE, edgecolor="white",
            label="VRAM (rel)")

for i, (f, m) in enumerate(zip(flops, mem)):
    ax_cost.text(i - w/2, f + 2, str(f), ha="center",
                 fontsize=10, fontweight="bold", color=BLUE)
    ax_cost.text(i + w/2, m + 2, str(m), ha="center",
                 fontsize=10, fontweight="bold", color=ORANGE)

ax_cost.set_xticks(x); ax_cost.set_xticklabels(labels, fontsize=10)
ax_cost.set_ylim(0, 120)
ax_cost.legend(fontsize=10, loc="upper right", framealpha=0.9)

ax_cost.text(0.5, -0.25,
             "~15x cheaper training + fits on a single consumer GPU.\n"
             "This is what made Stable Diffusion possible.",
             transform=ax_cost.transAxes, ha="center",
             fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 4 — VAE tradeoff ════════════════════
ax_rec.set_title("4 · VAE — Reconstruction vs Compression",
                 fontsize=13, fontweight="bold", color=DARK)
ax_rec.set_xlim(0, 10); ax_rec.set_ylim(0, 10); ax_rec.axis("off")

ratios = [4, 8, 16, 32]
fid = [4.5, 5.8, 9.2, 16.0]     # reconstruction error goes up
savings = [25, 48, 120, 320]    # compute savings grow

# Bar chart style
for i, r in enumerate(ratios):
    y = 8.5 - i * 1.7
    ax_rec.text(0.7, y, f"{r}x", fontweight="bold", fontsize=11,
                color=DARK)
    # recon bar (red = bad)
    ax_rec.add_patch(FancyBboxPatch((1.8, y - 0.3), fid[i] * 0.4, 0.6,
                                    boxstyle="round,pad=0.02",
                                    facecolor=RED, edgecolor="white", lw=1))
    ax_rec.text(1.9 + fid[i] * 0.4, y, f"recon err {fid[i]:.1f}",
                va="center", fontsize=9, color=DARK)

ax_rec.text(5, 1.7,
            'SD uses 8x. Sweet spot: "barely visible loss" at huge compute savings.',
            ha="center", fontsize=9.5, color="#555", style="italic",
            fontweight="bold")
ax_rec.text(5, 0.7,
            "Go too aggressive and fine details (text, faces) collapse.",
            ha="center", fontsize=9, color="#555", style="italic")

out = str(Path(__file__).resolve().parent.parent / "img" / "Latent Diffusion.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out}")
