from pathlib import Path
"""Generate Text to Video.png — 4-panel xkcd-style concept sketch.

Panels:
  (1) Add the time dimension: video tensor (T, H, W, C)
  (2) Spatial + temporal attention factorisation
  (3) Temporal consistency: naive per-frame vs video model
  (4) Compute scaling: frames x steps x resolution
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.xkcd(scale=0.8, length=100, randomness=2)

fig = plt.figure(figsize=(18, 11), facecolor="white")
fig.suptitle("Text-to-Video — Adding the Temporal Dimension",
             fontsize=20, fontweight="bold", y=0.98)

gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.28,
                      left=0.05, right=0.97, top=0.91, bottom=0.06)
ax_tens = fig.add_subplot(gs[0, 0])
ax_att = fig.add_subplot(gs[0, 1])
ax_cons = fig.add_subplot(gs[1, 0])
ax_cost = fig.add_subplot(gs[1, 1])

BLUE, ORANGE, GREEN, RED, PURPLE, DARK, GREY = (
    "#2E86C1", "#E67E22", "#27AE60", "#E74C3C", "#8E44AD", "#2C3E50", "#BDC3C7")

# ═══════════════════════ PANEL 1 — Video tensor ═════════════════════
ax_tens.set_title("1 · Video = Stack of Frames Over Time",
                  fontsize=13, fontweight="bold", color=DARK)
ax_tens.set_xlim(0, 10); ax_tens.set_ylim(0, 10); ax_tens.axis("off")

# Draw stacked frames
for k in range(5):
    x0 = 1.0 + k * 0.6
    y0 = 5.0 + k * 0.6
    ax_tens.add_patch(plt.Rectangle((x0, y0), 2.5, 2.5,
                                    facecolor=plt.cm.Blues(0.4 + 0.1*k),
                                    edgecolor="white", lw=2))
ax_tens.text(2.5, 4.3, "T frames", ha="center", fontsize=10,
             color=DARK, fontweight="bold")

# Shape label
ax_tens.text(7.0, 7.5, "(T, H, W, C)", ha="center",
             fontsize=14, color=DARK, family="monospace",
             fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#EAEDED",
                       edgecolor=DARK, lw=1))
ax_tens.text(7.0, 6.0, "or in latent:\n(T, H/8, W/8, C')",
             ha="center", fontsize=10, color=DARK, family="monospace")

ax_tens.text(5, 2.8,
             "Same diffusion math - but noise + UNet operate across T too.",
             ha="center", fontsize=10, color=DARK, fontweight="bold")
ax_tens.text(5, 1.5,
             "VAE often encodes in both space AND time (3D VAE / causal VAE).",
             ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 2 — Attention factorisation ═════════
ax_att.set_title("2 · Spatial + Temporal Attention",
                 fontsize=13, fontweight="bold", color=DARK)
ax_att.set_xlim(0, 10); ax_att.set_ylim(0, 10); ax_att.axis("off")

# spatial attention row
ax_att.text(2.5, 9.0, "spatial attention\n(within each frame)",
            ha="center", fontweight="bold", color=BLUE, fontsize=10)
for i in range(4):
    for j in range(4):
        ax_att.add_patch(plt.Rectangle((0.5 + j*0.5, 7.3 - i*0.5), 0.48, 0.48,
                                       facecolor=plt.cm.Blues(0.4),
                                       edgecolor="white", lw=0.5))
# highlight centre
ax_att.add_patch(plt.Rectangle((0.5 + 1.5, 7.3 - 1.5), 0.48, 0.48,
                               facecolor=RED, edgecolor="white", lw=0.8))

# temporal attention col
ax_att.text(7.5, 9.0, "temporal attention\n(same pixel across T)",
            ha="center", fontweight="bold", color=PURPLE, fontsize=10)
for k in range(6):
    ax_att.add_patch(plt.Rectangle((7.2, 7.3 - k*0.5), 0.48, 0.48,
                                   facecolor=plt.cm.Purples(0.4),
                                   edgecolor="white", lw=0.5))
# highlight centre
ax_att.add_patch(plt.Rectangle((7.2, 7.3 - 2.5), 0.48, 0.48,
                               facecolor=RED, edgecolor="white", lw=0.8))

# Combine
ax_att.add_patch(FancyBboxPatch((1.5, 2.4), 7.0, 1.6,
                                boxstyle="round,pad=0.1",
                                facecolor=GREEN, edgecolor="white", lw=2))
ax_att.text(5, 3.2,
            "interleave spatial & temporal blocks  (cheaper than full 3D att.)",
            ha="center", va="center", color="white",
            fontweight="bold", fontsize=10)

ax_att.text(5, 1.3,
            "Full 3D attention cost ~ (T*H*W)^2 -> infeasible.\n"
            "Factorisation drops it to T*(HW)^2 + HW*T^2.",
            ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 3 — Consistency ═════════════════════
ax_cons.set_title("3 · Temporal Consistency",
                  fontsize=13, fontweight="bold", color=DARK)
ax_cons.set_xlim(0, 10); ax_cons.set_ylim(0, 10); ax_cons.axis("off")

# Naive per-frame
ax_cons.text(2.5, 9.3, "per-frame txt2img", ha="center",
             fontweight="bold", color=RED, fontsize=11)
colors_n = [plt.cm.Reds(0.3), plt.cm.Reds(0.5), plt.cm.Reds(0.4),
            plt.cm.Reds(0.6), plt.cm.Reds(0.35)]
for k, c in enumerate(colors_n):
    ax_cons.add_patch(plt.Rectangle((0.4 + k*0.85, 6.8), 0.75, 1.6,
                                    facecolor=c, edgecolor="white", lw=1))
ax_cons.text(2.5, 6.2, "each frame independent",
             ha="center", fontsize=9, color=DARK)
ax_cons.text(2.5, 5.4, "flickers, objects appear/disappear",
             ha="center", fontsize=9, color=RED, fontweight="bold")

# Video model
ax_cons.text(7.5, 9.3, "video model", ha="center",
             fontweight="bold", color=GREEN, fontsize=11)
colors_v = [plt.cm.Greens(0.5), plt.cm.Greens(0.52), plt.cm.Greens(0.54),
            plt.cm.Greens(0.56), plt.cm.Greens(0.58)]
for k, c in enumerate(colors_v):
    ax_cons.add_patch(plt.Rectangle((5.4 + k*0.85, 6.8), 0.75, 1.6,
                                    facecolor=c, edgecolor="white", lw=1))
ax_cons.text(7.5, 6.2, "temporal attention stitches frames",
             ha="center", fontsize=9, color=DARK)
ax_cons.text(7.5, 5.4, "stable identity, smooth motion",
             ha="center", fontsize=9, color=GREEN, fontweight="bold")

ax_cons.text(5, 3.5, "Metrics that matter:",
             ha="center", fontweight="bold", fontsize=11, color=DARK)
ax_cons.text(5, 2.6, "- FVD (video-level FID)",
             ha="center", fontsize=9.5, color=DARK)
ax_cons.text(5, 2.0, "- CLIP-T (prompt alignment per frame)",
             ha="center", fontsize=9.5, color=DARK)
ax_cons.text(5, 1.4, "- temporal flicker score",
             ha="center", fontsize=9.5, color=DARK)

ax_cons.text(5, 0.4,
             "Human eval is still the gold standard.",
             ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 4 — Compute ═════════════════════════
ax_cost.set_title("4 · Why Video Is ~100x More Expensive",
                  fontsize=13, fontweight="bold", color=DARK)
dims = ["frames\n(T=16)", "steps\n(T=30)", "resolution\n(1024^2)", "total"]
rel = [16, 30, 16, 7680]   # product
colors = [BLUE, ORANGE, PURPLE, RED]
x = np.arange(len(dims))

ax_cost.bar(x, rel, color=colors, edgecolor="white", width=0.6)
ax_cost.set_yscale("log")
ax_cost.set_xticks(x); ax_cost.set_xticklabels(dims, fontsize=10)
ax_cost.set_ylabel("relative cost (log)", fontsize=10, color=DARK)

for i, v in enumerate(rel):
    ax_cost.text(i, v * 1.3, f"x{v}", ha="center", fontsize=10,
                 fontweight="bold", color=colors[i])

ax_cost.text(0.5, -0.27,
             "Mitigations: latent VAE in time, cascaded models (low-res -> up),\n"
             "flow matching, distilled 4-8 step samplers.",
             transform=ax_cost.transAxes, ha="center",
             fontsize=9, color="#555", style="italic")

out = str(Path(__file__).resolve().parent.parent / "img" / "Text to Video.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out}")
