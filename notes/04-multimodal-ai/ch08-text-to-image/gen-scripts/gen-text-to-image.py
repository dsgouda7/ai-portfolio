from pathlib import Path
"""Generate Text to Image.png — 4-panel xkcd-style concept sketch.

Panels:
  (1) txt2img pipeline
  (2) img2img with strength / start-noise level
  (3) Inpainting with mask
  (4) ControlNet conditioning stack
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.xkcd(scale=0.8, length=100, randomness=2)

fig = plt.figure(figsize=(18, 11), facecolor="white")
fig.suptitle("Text-to-Image — Prompts, img2img, Inpainting, ControlNet",
             fontsize=20, fontweight="bold", y=0.98)

gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.28,
                      left=0.05, right=0.97, top=0.91, bottom=0.06)
ax_t2i = fig.add_subplot(gs[0, 0])
ax_i2i = fig.add_subplot(gs[0, 1])
ax_inp = fig.add_subplot(gs[1, 0])
ax_cn = fig.add_subplot(gs[1, 1])

BLUE, ORANGE, GREEN, RED, PURPLE, DARK, GREY = (
    "#2E86C1", "#E67E22", "#27AE60", "#E74C3C", "#8E44AD", "#2C3E50", "#BDC3C7")

# ═══════════════════════ PANEL 1 — txt2img ═════════════════════════
ax_t2i.set_title("1 · txt2img — Start from Pure Noise",
                 fontsize=13, fontweight="bold", color=DARK)
ax_t2i.set_xlim(0, 10); ax_t2i.set_ylim(0, 10); ax_t2i.axis("off")

stages = [
    (1.2, "prompt",       GREEN,  '"a red fox"'),
    (3.2, "CLIP text",    BLUE,   "77x768"),
    (5.2, "random noise", GREY,   "xT ~ N(0,I)"),
    (7.2, "UNet x T",     PURPLE, "denoise loop"),
    (9.2, "VAE decode",   ORANGE, "image"),
]
for x, name, c, body in stages:
    ax_t2i.add_patch(FancyBboxPatch((x - 0.75, 5.0), 1.5, 2.2,
                                    boxstyle="round,pad=0.05",
                                    facecolor=c, edgecolor="white", lw=2))
    ax_t2i.text(x, 6.5, name, ha="center", va="center",
                color="white", fontweight="bold", fontsize=9.5)
    ax_t2i.text(x, 5.4, body, ha="center", fontsize=8.5, color="white",
                family="monospace")
for i in range(4):
    x1, x2 = stages[i][0] + 0.8, stages[i + 1][0] - 0.8
    ax_t2i.annotate("", xy=(x2, 6.0), xytext=(x1, 6.0),
                    arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.4))

ax_t2i.text(5, 3.2,
            "Text guides every denoising step via cross-attention.",
            ha="center", fontsize=10, color=DARK, fontweight="bold")
ax_t2i.text(5, 1.8,
            "Full T noise -> pure creative freedom\n"
            "but anchored by the prompt.",
            ha="center", fontsize=9.5, color="#555", style="italic")

# ═══════════════════════ PANEL 2 — img2img ═════════════════════════
ax_i2i.set_title("2 · img2img — Start Partially Noised",
                 fontsize=13, fontweight="bold", color=DARK)
ax_i2i.set_xlim(0, 10); ax_i2i.set_ylim(0, 10); ax_i2i.axis("off")

# Input image
ax_i2i.add_patch(FancyBboxPatch((0.3, 6.0), 2.0, 2.0,
                                boxstyle="round,pad=0.05",
                                facecolor=BLUE, edgecolor="white", lw=2))
ax_i2i.text(1.3, 7.0, "input\nimage", ha="center", va="center",
            color="white", fontweight="bold", fontsize=10)

# Partial noise
ax_i2i.annotate("", xy=(3.3, 7.0), xytext=(2.4, 7.0),
                arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.4))
ax_i2i.text(2.85, 7.4, "+noise", fontsize=9, color=DARK, ha="center")
ax_i2i.add_patch(FancyBboxPatch((3.3, 6.0), 2.0, 2.0,
                                boxstyle="round,pad=0.05",
                                facecolor=GREY, edgecolor="white", lw=2))
ax_i2i.text(4.3, 7.0, "x_t\n(partial)", ha="center", va="center",
            color="white", fontweight="bold", fontsize=10)

# UNet loop
ax_i2i.annotate("", xy=(6.3, 7.0), xytext=(5.4, 7.0),
                arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.4))
ax_i2i.add_patch(FancyBboxPatch((6.3, 6.0), 1.8, 2.0,
                                boxstyle="round,pad=0.05",
                                facecolor=PURPLE, edgecolor="white", lw=2))
ax_i2i.text(7.2, 7.0, "UNet loop", ha="center", va="center",
            color="white", fontweight="bold", fontsize=9.5)

# Out
ax_i2i.annotate("", xy=(9.2, 7.0), xytext=(8.3, 7.0),
                arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.4))
ax_i2i.add_patch(FancyBboxPatch((8.2, 6.0), 1.6, 2.0,
                                boxstyle="round,pad=0.05",
                                facecolor=ORANGE, edgecolor="white", lw=2))
ax_i2i.text(9.0, 7.0, "output", ha="center", va="center",
            color="white", fontweight="bold", fontsize=10)

# Strength axis
ax_i2i.annotate("", xy=(9.4, 3.0), xytext=(0.6, 3.0),
                arrowprops=dict(arrowstyle="-|>", color=DARK, lw=2))
ax_i2i.text(0.6, 3.5, "strength 0.1", fontsize=9, color=DARK)
ax_i2i.text(9.4, 3.5, "strength 0.95", fontsize=9, color=DARK, ha="right")
ax_i2i.text(5, 3.5, "0.5", fontsize=9, color=DARK, ha="center",
            fontweight="bold")
ax_i2i.text(5, 2.1,
            "strength = fraction of T to start from noise",
            ha="center", fontsize=10, color=DARK, fontweight="bold")
ax_i2i.text(5, 1.0,
            "low -> tiny edits, keeps composition.  high -> near txt2img.",
            ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 3 — Inpainting ══════════════════════
ax_inp.set_title("3 · Inpainting — Regenerate Under a Mask",
                 fontsize=13, fontweight="bold", color=DARK)
ax_inp.set_xlim(0, 10); ax_inp.set_ylim(0, 10); ax_inp.axis("off")

# Original image
ax_inp.text(2.0, 9.3, "original", ha="center", fontweight="bold",
            fontsize=11, color=BLUE)
for i in range(6):
    for j in range(6):
        ax_inp.add_patch(plt.Rectangle((0.6 + j*0.5, 7.8 - i*0.5), 0.48, 0.48,
                                       facecolor=plt.cm.Blues(0.35 + 0.05*((i+j) % 5)),
                                       edgecolor="white", lw=0.4))

# Mask
ax_inp.text(5.0, 9.3, "mask", ha="center", fontweight="bold",
            fontsize=11, color=RED)
for i in range(6):
    for j in range(6):
        in_mask = (2 <= i <= 3 and 2 <= j <= 4)
        c = RED if in_mask else "white"
        ax_inp.add_patch(plt.Rectangle((3.6 + j*0.5, 7.8 - i*0.5), 0.48, 0.48,
                                       facecolor=c, edgecolor=DARK, lw=0.4))

# Result
ax_inp.text(8.0, 9.3, "result", ha="center", fontweight="bold",
            fontsize=11, color=GREEN)
for i in range(6):
    for j in range(6):
        in_mask = (2 <= i <= 3 and 2 <= j <= 4)
        if in_mask:
            c = plt.cm.Greens(0.5 + 0.08*((i+j) % 5))
        else:
            c = plt.cm.Blues(0.35 + 0.05*((i+j) % 5))
        ax_inp.add_patch(plt.Rectangle((6.6 + j*0.5, 7.8 - i*0.5), 0.48, 0.48,
                                       facecolor=c, edgecolor="white", lw=0.4))

ax_inp.text(5, 3.8,
            'At each step: new latent INSIDE mask, original latent OUTSIDE.',
            ha="center", fontsize=10, color=DARK, fontweight="bold")
ax_inp.text(5, 2.5,
            'x_t := m * x_t^new + (1 - m) * x_t^orig',
            ha="center", fontsize=10, color=DARK, family="monospace")
ax_inp.text(5, 1.0,
            "Seam blending avoids hard edges around the mask boundary.",
            ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 4 — ControlNet stack ═════════════════
ax_cn.set_title("4 · ControlNet — Structural Conditioning",
                fontsize=13, fontweight="bold", color=DARK)
ax_cn.set_xlim(0, 10); ax_cn.set_ylim(0, 10); ax_cn.axis("off")

# Original UNet at the top
ax_cn.add_patch(FancyBboxPatch((2.5, 7.0), 5.0, 1.6,
                               boxstyle="round,pad=0.05",
                               facecolor=PURPLE, edgecolor="white", lw=2))
ax_cn.text(5.0, 7.8, "frozen UNet (trained on big data)",
           ha="center", va="center", color="white",
           fontweight="bold", fontsize=10)

# Trainable copy
ax_cn.add_patch(FancyBboxPatch((0.3, 4.5), 3.2, 1.6,
                               boxstyle="round,pad=0.05",
                               facecolor=GREEN, edgecolor="white", lw=2))
ax_cn.text(1.9, 5.3, "trainable\nencoder copy",
           ha="center", va="center", color="white",
           fontweight="bold", fontsize=10)

# Zero-conv
ax_cn.add_patch(FancyBboxPatch((4.0, 4.5), 2.0, 1.6,
                               boxstyle="round,pad=0.05",
                               facecolor=DARK, edgecolor="white", lw=2))
ax_cn.text(5.0, 5.3, "zero conv",
           ha="center", va="center", color="white",
           fontweight="bold", fontsize=10)

# Hint
ax_cn.add_patch(FancyBboxPatch((6.6, 4.5), 3.2, 1.6,
                               boxstyle="round,pad=0.05",
                               facecolor=ORANGE, edgecolor="white", lw=2))
ax_cn.text(8.2, 5.3, "hint\n(canny / depth / pose)",
           ha="center", va="center", color="white",
           fontweight="bold", fontsize=9.5)

# Arrows
ax_cn.annotate("", xy=(3.9, 5.3), xytext=(3.5, 5.3),
               arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.4))
ax_cn.annotate("", xy=(6.5, 5.3), xytext=(6.1, 5.3),
               arrowprops=dict(arrowstyle="<-", color=DARK, lw=1.4))
ax_cn.annotate("", xy=(5, 6.9), xytext=(5, 6.2),
               arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.8))
ax_cn.text(5.5, 6.55, "add", fontsize=9, color=DARK)

# Output
ax_cn.add_patch(FancyBboxPatch((3.0, 1.5), 4.0, 1.6,
                               boxstyle="round,pad=0.05",
                               facecolor=BLUE, edgecolor="white", lw=2))
ax_cn.text(5.0, 2.3, "conditioned image",
           ha="center", va="center", color="white",
           fontweight="bold", fontsize=10)
ax_cn.annotate("", xy=(5, 3.1), xytext=(5, 3.7),
               arrowprops=dict(arrowstyle="<-", color=DARK, lw=1.4))

ax_cn.text(5, 0.6,
           "Freeze original. Train only the copy + zero-convs. Stable & small.",
           ha="center", fontsize=9, color="#555", style="italic")

out = str(Path(__file__).resolve().parent.parent / "img" / "Text to Image.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out}")
