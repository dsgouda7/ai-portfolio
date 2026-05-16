from pathlib import Path
"""Generate CLIP.png — 4-panel xkcd-style concept sketch.

Panels:
  (1) Dual-encoder architecture (image tower + text tower)
  (2) Contrastive similarity matrix (diagonal = positives)
  (3) Zero-shot classification via text prompts
  (4) Shared embedding space 2D sketch
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.xkcd(scale=0.8, length=100, randomness=2)

fig = plt.figure(figsize=(18, 11), facecolor="white")
fig.suptitle("CLIP — Contrastive Language-Image Pretraining",
             fontsize=20, fontweight="bold", y=0.98)

gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.28,
                      left=0.05, right=0.97, top=0.91, bottom=0.06)
ax_arch = fig.add_subplot(gs[0, 0])
ax_m = fig.add_subplot(gs[0, 1])
ax_zs = fig.add_subplot(gs[1, 0])
ax_sp = fig.add_subplot(gs[1, 1])

BLUE, ORANGE, GREEN, RED, PURPLE, DARK, GREY = (
    "#2E86C1", "#E67E22", "#27AE60", "#E74C3C", "#8E44AD", "#2C3E50", "#BDC3C7")

# ═══════════════════════ PANEL 1 — Dual encoders ═══════════════════
ax_arch.set_title("1 · Dual-Encoder Architecture",
                  fontsize=13, fontweight="bold", color=DARK)
ax_arch.set_xlim(0, 10); ax_arch.set_ylim(0, 10); ax_arch.axis("off")

# image tower
ax_arch.add_patch(FancyBboxPatch((0.4, 6.5), 2.0, 1.2,
                                 boxstyle="round,pad=0.05",
                                 facecolor=GREY, edgecolor="white", lw=1.5))
ax_arch.text(1.4, 7.1, "image", ha="center", va="center",
             color="white", fontweight="bold", fontsize=10)
ax_arch.annotate("", xy=(1.4, 5.9), xytext=(1.4, 6.4),
                 arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.4))
ax_arch.add_patch(FancyBboxPatch((0.3, 4.2), 2.2, 1.6,
                                 boxstyle="round,pad=0.05",
                                 facecolor=BLUE, edgecolor="white", lw=2))
ax_arch.text(1.4, 5.0, "ViT /\nResNet", ha="center", va="center",
             color="white", fontweight="bold", fontsize=10)
ax_arch.annotate("", xy=(1.4, 3.5), xytext=(1.4, 4.1),
                 arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.4))
ax_arch.add_patch(FancyBboxPatch((0.5, 2.6), 1.8, 0.9,
                                 boxstyle="round,pad=0.05",
                                 facecolor=PURPLE, edgecolor="white", lw=1.5))
ax_arch.text(1.4, 3.05, "proj -> d", ha="center", va="center",
             color="white", fontweight="bold", fontsize=9.5)

# text tower
ax_arch.add_patch(FancyBboxPatch((7.6, 6.5), 2.0, 1.2,
                                 boxstyle="round,pad=0.05",
                                 facecolor=GREY, edgecolor="white", lw=1.5))
ax_arch.text(8.6, 7.1, '"a red fox"', ha="center", va="center",
             color="white", fontweight="bold", fontsize=10)
ax_arch.annotate("", xy=(8.6, 5.9), xytext=(8.6, 6.4),
                 arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.4))
ax_arch.add_patch(FancyBboxPatch((7.5, 4.2), 2.2, 1.6,
                                 boxstyle="round,pad=0.05",
                                 facecolor=GREEN, edgecolor="white", lw=2))
ax_arch.text(8.6, 5.0, "Text\nTransformer", ha="center", va="center",
             color="white", fontweight="bold", fontsize=10)
ax_arch.annotate("", xy=(8.6, 3.5), xytext=(8.6, 4.1),
                 arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.4))
ax_arch.add_patch(FancyBboxPatch((7.7, 2.6), 1.8, 0.9,
                                 boxstyle="round,pad=0.05",
                                 facecolor=PURPLE, edgecolor="white", lw=1.5))
ax_arch.text(8.6, 3.05, "proj -> d", ha="center", va="center",
             color="white", fontweight="bold", fontsize=9.5)

# shared space
ax_arch.add_patch(FancyBboxPatch((3.2, 0.7), 3.6, 2.0,
                                 boxstyle="round,pad=0.1",
                                 facecolor=ORANGE, edgecolor="white", lw=2))
ax_arch.text(5.0, 1.7, "cosine similarity\n+  InfoNCE loss",
             ha="center", va="center", color="white",
             fontweight="bold", fontsize=11)

ax_arch.annotate("", xy=(3.3, 1.9), xytext=(2.3, 2.7),
                 arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.4))
ax_arch.annotate("", xy=(6.7, 1.9), xytext=(7.7, 2.7),
                 arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.4))

# ═══════════════════════ PANEL 2 — Similarity matrix ═══════════════
ax_m.set_title("2 · Contrastive Similarity Matrix (batch of N)",
               fontsize=13, fontweight="bold", color=DARK)
N = 6
mat = 0.15 + 0.1 * np.random.default_rng(1).random((N, N))
for i in range(N):
    mat[i, i] = 0.9
im = ax_m.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="equal")
ax_m.set_xticks(range(N))
ax_m.set_yticks(range(N))
ax_m.set_xticklabels([f"T{i+1}" for i in range(N)], fontsize=9)
ax_m.set_yticklabels([f"I{i+1}" for i in range(N)], fontsize=9)
ax_m.set_xlabel("text embeddings", fontsize=10, color=DARK)
ax_m.set_ylabel("image embeddings", fontsize=10, color=DARK)

# highlight diagonal
for i in range(N):
    ax_m.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                 fill=False, edgecolor=DARK, lw=2))
ax_m.text(0.5, -1.4,
          "Maximise diagonal (matched pairs), minimise everything else.\n"
          "Loss = symmetric cross-entropy over rows and columns.",
          transform=ax_m.transAxes, ha="center",
          fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 3 — Zero-shot ═══════════════════════
ax_zs.set_title("3 · Zero-Shot Classification",
                fontsize=13, fontweight="bold", color=DARK)
ax_zs.set_xlim(0, 10); ax_zs.set_ylim(0, 10); ax_zs.axis("off")

# image on left
ax_zs.add_patch(FancyBboxPatch((0.3, 4.3), 2.5, 2.5,
                               boxstyle="round,pad=0.05",
                               facecolor=BLUE, edgecolor="white", lw=2))
ax_zs.text(1.55, 5.55, "[image]", ha="center", va="center",
           color="white", fontweight="bold", fontsize=11)

# text prompts
prompts = [
    (6.5, 8.2,  '"a photo of a cat"',  0.08),
    (6.5, 7.0,  '"a photo of a dog"',  0.15),
    (6.5, 5.8,  '"a photo of a fox"',  0.72),
    (6.5, 4.6,  '"a photo of a car"',  0.03),
    (6.5, 3.4,  '"a photo of a bird"', 0.02),
]
for x, y, text, score in prompts:
    c = GREEN if score > 0.5 else GREY
    ax_zs.add_patch(FancyBboxPatch((4.0, y - 0.35), 4.0, 0.7,
                                   boxstyle="round,pad=0.05",
                                   facecolor=c, edgecolor="white", lw=1.5))
    ax_zs.text(4.2, y, text, va="center", color="white",
               fontweight="bold", fontsize=9.5)
    ax_zs.text(7.9, y, f"{score:.2f}", va="center", ha="right",
               color="white", fontweight="bold", fontsize=10)

ax_zs.annotate("", xy=(3.9, 5.5), xytext=(2.9, 5.5),
               arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.5))
ax_zs.text(3.4, 5.9, "cos sim", fontsize=9, color=DARK, ha="center")

ax_zs.text(5, 1.8, "argmax = fox",
           ha="center", color=GREEN, fontweight="bold", fontsize=13)
ax_zs.text(5, 0.7,
           "No fine-tuning. Swap class names -> new classifier.",
           ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 4 — Shared space ═════════════════════
ax_sp.set_title("4 · Shared Embedding Space (2D sketch)",
                fontsize=13, fontweight="bold", color=DARK)
ax_sp.set_xlim(-3, 3); ax_sp.set_ylim(-3, 3)
ax_sp.set_xticks([]); ax_sp.set_yticks([])

rng = np.random.default_rng(4)
clusters = [
    ((-1.5, 1.5), "foxes",  RED),
    (( 1.5, 1.5), "cats",   ORANGE),
    (( 0.0, -1.6), "cars",  BLUE),
]
for (cx, cy), name, c in clusters:
    img_x = cx + rng.normal(0, 0.25, 5)
    img_y = cy + rng.normal(0, 0.25, 5)
    ax_sp.scatter(img_x, img_y, c=c, s=90, marker="o",
                  edgecolors="white", lw=1.2, label=name + " (img)")
    txt_x = cx + rng.normal(0, 0.18, 3)
    txt_y = cy + rng.normal(0, 0.18, 3)
    ax_sp.scatter(txt_x, txt_y, c=c, s=100, marker="^",
                  edgecolors=DARK, lw=1)
    ax_sp.text(cx, cy + 0.7, name, ha="center", fontweight="bold",
               color=c, fontsize=10)

ax_sp.scatter([], [], c=GREY, s=90, marker="o",
              edgecolors="white", label="image")
ax_sp.scatter([], [], c=GREY, s=100, marker="^",
              edgecolors=DARK, label="text")
ax_sp.legend(loc="lower right", fontsize=9, framealpha=0.9)

ax_sp.text(0, -3.4,
           "Image & text of same concept co-locate. Distance = semantic similarity.",
           transform=ax_sp.transData, ha="center",
           fontsize=9, color="#555", style="italic")

out = str(Path(__file__).resolve().parent.parent / "img" / "CLIP.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out}")
