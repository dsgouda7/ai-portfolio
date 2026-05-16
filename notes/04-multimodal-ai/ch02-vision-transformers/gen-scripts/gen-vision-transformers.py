from pathlib import Path
"""Generate Vision Transformers.png — 4-panel xkcd-style concept sketch.

Panels:
  (1) Image -> patches -> token sequence
  (2) ViT block diagram (patch embed + pos + encoder + head)
  (3) CNN vs ViT receptive field at layer 1
  (4) Scale curve: accuracy vs data regime
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.xkcd(scale=0.8, length=100, randomness=2)

fig = plt.figure(figsize=(18, 11), facecolor="white")
fig.suptitle("Vision Transformers — How Images Become Sequences",
             fontsize=20, fontweight="bold", y=0.98)

gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.28,
                      left=0.05, right=0.97, top=0.91, bottom=0.06)
ax_patch = fig.add_subplot(gs[0, 0])
ax_arch = fig.add_subplot(gs[0, 1])
ax_rf = fig.add_subplot(gs[1, 0])
ax_scale = fig.add_subplot(gs[1, 1])

BLUE, ORANGE, GREEN, RED, PURPLE, DARK, GREY = (
    "#2E86C1", "#E67E22", "#27AE60", "#E74C3C", "#8E44AD", "#2C3E50", "#BDC3C7")

# ═══════════════════════ PANEL 1 — Patchify ═════════════════════════
ax_patch.set_title("1 · Image -> 16x16 Patches -> Token Sequence",
                   fontsize=13, fontweight="bold", color=DARK)
ax_patch.set_xlim(0, 10); ax_patch.set_ylim(0, 10); ax_patch.axis("off")

# The image as 4x4 patches
n = 4
rng = np.random.default_rng(2)
colors = rng.uniform(0.3, 0.9, (n, n))
for i in range(n):
    for j in range(n):
        ax_patch.add_patch(plt.Rectangle((0.4 + j*0.55, 7.5 - i*0.55), 0.5, 0.5,
                                         facecolor=plt.cm.Blues(colors[i, j]),
                                         edgecolor="white", lw=0.6))
ax_patch.text(1.6, 9.4, "224x224", ha="center", fontsize=10,
              color=DARK, family="monospace")

ax_patch.annotate("", xy=(4.0, 8.1), xytext=(3.0, 8.1),
                  arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.5))
ax_patch.text(3.5, 8.5, "flatten", fontsize=9, color=DARK, ha="center")

# Token sequence
for k in range(8):
    ax_patch.add_patch(FancyBboxPatch((4.1 + k*0.65, 7.4), 0.55, 1.0,
                                      boxstyle="round,pad=0.03",
                                      facecolor=plt.cm.Blues(0.4 + 0.05*k),
                                      edgecolor="white", lw=1))
    ax_patch.text(4.1 + k*0.65 + 0.27, 7.9, f"p{k+1}", ha="center",
                  va="center", color="white", fontsize=8)
ax_patch.text(6.7, 6.9, "... 196 patch tokens", ha="center",
              fontsize=9.5, color=DARK)

# + [CLS] + pos
ax_patch.add_patch(FancyBboxPatch((0.4, 4.3), 1.1, 1.0,
                                  boxstyle="round,pad=0.03",
                                  facecolor=PURPLE, edgecolor="white", lw=1.5))
ax_patch.text(0.95, 4.8, "[CLS]", ha="center", va="center",
              color="white", fontweight="bold", fontsize=9)
ax_patch.text(0.95, 3.6, "learnable", ha="center", fontsize=8.5, color=DARK)

ax_patch.text(2.0, 4.8, "+", fontsize=14, color=DARK, fontweight="bold")

ax_patch.add_patch(FancyBboxPatch((2.4, 4.3), 2.5, 1.0,
                                  boxstyle="round,pad=0.05",
                                  facecolor=ORANGE, edgecolor="white", lw=1.5))
ax_patch.text(3.65, 4.8, "linear embed", ha="center", va="center",
              color="white", fontweight="bold", fontsize=10)

ax_patch.text(5.2, 4.8, "+", fontsize=14, color=DARK, fontweight="bold")

ax_patch.add_patch(FancyBboxPatch((5.6, 4.3), 2.2, 1.0,
                                  boxstyle="round,pad=0.05",
                                  facecolor=GREEN, edgecolor="white", lw=1.5))
ax_patch.text(6.7, 4.8, "pos embed", ha="center", va="center",
              color="white", fontweight="bold", fontsize=10)

ax_patch.text(5, 2.8,
              "Output: (197, D) — treat like any text sequence.",
              ha="center", fontsize=10, color=DARK, fontweight="bold")
ax_patch.text(5, 1.5,
              "[CLS] pools global info. Pos-embedding is the ONLY place spatial order lives.",
              ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 2 — ViT block ═══════════════════════
ax_arch.set_title("2 · ViT Architecture",
                  fontsize=13, fontweight="bold", color=DARK)
ax_arch.set_xlim(0, 10); ax_arch.set_ylim(0, 10); ax_arch.axis("off")

tiers = [
    (9.0, "patch + pos embed",    ORANGE),
    (7.5, "Nx { LN -> MHSA -> +res -> LN -> MLP -> +res }", BLUE),
    (6.0, "(repeated 12 / 24 / 32 times)",                  GREY),
    (4.5, "final LayerNorm",                                PURPLE),
    (3.0, "pluck [CLS] token",                              DARK),
    (1.5, "MLP head -> classes / embedding",                GREEN),
]
for y, name, c in tiers:
    ax_arch.add_patch(FancyBboxPatch((0.5, y - 0.55), 9.0, 1.1,
                                     boxstyle="round,pad=0.05",
                                     facecolor=c, edgecolor="white", lw=2))
    ax_arch.text(5, y, name, ha="center", va="center",
                 color="white", fontweight="bold", fontsize=10)
for y1, y2 in [(9.0, 7.5), (7.5, 4.5), (4.5, 3.0), (3.0, 1.5)]:
    ax_arch.annotate("", xy=(5, y2 + 0.55), xytext=(5, y1 - 0.55),
                     arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.2))
ax_arch.text(5, 0.4,
             "Same transformer as GPT — just different input pipeline.",
             ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 3 — Receptive field ═════════════════
ax_rf.set_title("3 · Receptive Field at Layer 1",
                fontsize=13, fontweight="bold", color=DARK)
ax_rf.set_xlim(0, 10); ax_rf.set_ylim(0, 10); ax_rf.axis("off")

# CNN side
ax_rf.text(2.5, 9.3, "CNN (3x3 conv)", ha="center",
           fontweight="bold", color=BLUE, fontsize=11)
grid = 7
for i in range(grid):
    for j in range(grid):
        color = "white"
        if 2 <= i <= 4 and 2 <= j <= 4:
            color = "#FADBD8"
        if i == 3 and j == 3:
            color = RED
        ax_rf.add_patch(plt.Rectangle((0.4 + j*0.45, 7.8 - i*0.45),
                                      0.42, 0.42,
                                      facecolor=color, edgecolor=BLUE, lw=0.8))
ax_rf.text(2.5, 3.8, "looks at 3x3\nlocal neighbourhood",
           ha="center", fontsize=9, color=DARK)

# ViT side
ax_rf.text(7.5, 9.3, "ViT (self-attention)", ha="center",
           fontweight="bold", color=PURPLE, fontsize=11)
for i in range(grid):
    for j in range(grid):
        color = "#F4ECF7"
        if i == 3 and j == 3:
            color = RED
        ax_rf.add_patch(plt.Rectangle((5.4 + j*0.45, 7.8 - i*0.45),
                                      0.42, 0.42,
                                      facecolor=color, edgecolor=PURPLE, lw=0.8))
ax_rf.text(7.5, 3.8, "every token sees\nevery other token",
           ha="center", fontsize=9, color=DARK)

ax_rf.text(5, 2.0,
           "ViT = global receptive field from layer 1. CNN grows it slowly with depth.",
           ha="center", fontsize=9.5, color="#555", style="italic")
ax_rf.text(5, 0.8,
           "Trade-off: ViT needs MUCH more data (or pretraining) to learn locality.",
           ha="center", fontsize=9, color=RED, fontweight="bold")

# ═══════════════════════ PANEL 4 — Scale curve ═════════════════════
ax_scale.set_title("4 · Accuracy vs Training-Set Size",
                   fontsize=13, fontweight="bold", color=DARK)

data = np.array([1e6, 1e7, 1e8, 1e9])
cnn = [72, 78, 81, 82]
vit = [55, 74, 83, 88]

ax_scale.plot(data, cnn, "-o", color=BLUE,   lw=2.5, markersize=8,
              label="ResNet (CNN)")
ax_scale.plot(data, vit, "-s", color=PURPLE, lw=2.5, markersize=8,
              label="ViT")

# Crossover annotation
ax_scale.axvline(3e7, color=GREY, ls="--", lw=1, alpha=0.7)
ax_scale.text(3e7, 60, "crossover\n~30M imgs", ha="center",
              fontsize=9, color=GREY)

ax_scale.set_xscale("log")
ax_scale.set_xlabel("pretraining images (log scale)", fontsize=10, color=DARK)
ax_scale.set_ylabel("top-1 accuracy (%)", fontsize=10, color=DARK)
ax_scale.set_ylim(50, 95)
ax_scale.legend(fontsize=10, loc="lower right", framealpha=0.9)

ax_scale.text(0.5, -0.25,
              "Small data -> CNN wins. Big data -> ViT dominates.\n"
              "This is why ViTs pair with CLIP-scale pretraining.",
              transform=ax_scale.transAxes, ha="center",
              fontsize=9, color="#555", style="italic")

out = str(Path(__file__).resolve().parent.parent / "img" / "Vision Transformers.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out}")
