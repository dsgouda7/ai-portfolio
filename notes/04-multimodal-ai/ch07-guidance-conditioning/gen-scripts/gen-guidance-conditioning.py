from pathlib import Path
"""Generate Guidance and Conditioning.png — 4-panel xkcd-style concept sketch.

Panels:
  (1) Unconditional vs conditional branches (CFG eps combination)
  (2) Guidance-scale sweep: too low / right / too high
  (3) Cross-attention between text tokens and UNet
  (4) ControlNet / IP-Adapter / T2I-Adapter conditioning types
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.xkcd(scale=0.8, length=100, randomness=2)

fig = plt.figure(figsize=(18, 11), facecolor="white")
fig.suptitle("Guidance & Conditioning — Making Diffusion Follow Instructions",
             fontsize=20, fontweight="bold", y=0.98)

gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.28,
                      left=0.05, right=0.97, top=0.91, bottom=0.06)
ax_cfg = fig.add_subplot(gs[0, 0])
ax_sw = fig.add_subplot(gs[0, 1])
ax_xa = fig.add_subplot(gs[1, 0])
ax_ctrl = fig.add_subplot(gs[1, 1])

BLUE, ORANGE, GREEN, RED, PURPLE, DARK, GREY = (
    "#2E86C1", "#E67E22", "#27AE60", "#E74C3C", "#8E44AD", "#2C3E50", "#BDC3C7")

# ═══════════════════════ PANEL 1 — CFG branches ════════════════════
ax_cfg.set_title("1 · Classifier-Free Guidance",
                 fontsize=13, fontweight="bold", color=DARK)
ax_cfg.set_xlim(0, 10); ax_cfg.set_ylim(0, 10); ax_cfg.axis("off")

# Conditional branch
ax_cfg.add_patch(FancyBboxPatch((0.4, 6.8), 4.0, 1.6,
                                boxstyle="round,pad=0.05",
                                facecolor=BLUE, edgecolor="white", lw=2))
ax_cfg.text(2.4, 7.6, "UNet(x_t, t, c)", ha="center", va="center",
            color="white", fontweight="bold", fontsize=11)
ax_cfg.text(2.4, 6.4, "eps_cond", ha="center", fontsize=10,
            color=DARK, family="monospace")

# Unconditional branch
ax_cfg.add_patch(FancyBboxPatch((5.6, 6.8), 4.0, 1.6,
                                boxstyle="round,pad=0.05",
                                facecolor=GREY, edgecolor="white", lw=2))
ax_cfg.text(7.6, 7.6, "UNet(x_t, t, null)", ha="center", va="center",
            color="white", fontweight="bold", fontsize=11)
ax_cfg.text(7.6, 6.4, "eps_uncond", ha="center", fontsize=10,
            color=DARK, family="monospace")

# Combine
ax_cfg.annotate("", xy=(5, 4.8), xytext=(2.4, 6.3),
                arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.4))
ax_cfg.annotate("", xy=(5, 4.8), xytext=(7.6, 6.3),
                arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.4))

ax_cfg.add_patch(FancyBboxPatch((1.8, 3.2), 6.4, 1.6,
                                boxstyle="round,pad=0.1",
                                facecolor=PURPLE, edgecolor="white", lw=2))
ax_cfg.text(5, 4.0,
            "eps = eps_uncond + w * (eps_cond - eps_uncond)",
            ha="center", va="center",
            color="white", fontweight="bold", fontsize=11,
            family="monospace")

ax_cfg.text(5, 2.1,
            "w = 1 -> plain conditional.  w > 1 -> amplify text signal.",
            ha="center", fontsize=10, color=DARK)
ax_cfg.text(5, 0.9,
            "Cost = 2x model evals per step (uncond + cond).",
            ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 2 — Scale sweep ══════════════════════
ax_sw.set_title("2 · Guidance Scale Sweep (prompt: 'a red fox')",
                fontsize=13, fontweight="bold", color=DARK)
ax_sw.set_xlim(0, 10); ax_sw.set_ylim(0, 10); ax_sw.axis("off")

scales = [(1.3, "w=1",  "#FDEBD0", "ignores prompt\n(generic fox)"),
          (3.8, "w=3",  "#F5B041", "faithful + natural"),
          (6.3, "w=7.5", "#E67E22", "sweet spot"),
          (8.8, "w=20", "#B9770E", "oversaturated,\nblurry, plastic")]

for x, lbl, c, txt in scales:
    ax_sw.add_patch(FancyBboxPatch((x - 0.9, 5.0), 1.8, 3.2,
                                   boxstyle="round,pad=0.05",
                                   facecolor=c, edgecolor="white", lw=2))
    ax_sw.text(x, 8.6, lbl, ha="center", fontweight="bold",
               color=DARK, fontsize=11)
    ax_sw.text(x, 3.8, txt, ha="center", fontsize=9, color=DARK)

# Arrow underneath
ax_sw.annotate("", xy=(9.4, 2.2), xytext=(0.6, 2.2),
               arrowprops=dict(arrowstyle="-|>", color=DARK, lw=2))
ax_sw.text(5, 1.3,
           "increasing w -> more prompt adherence, less variety, more artefacts",
           ha="center", fontsize=9.5, color=DARK)
ax_sw.text(5, 0.4,
           "Typical SD defaults: 7 - 9.",
           ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 3 — Cross-attention ═════════════════
ax_xa.set_title("3 · Cross-Attention — Text -> UNet",
                fontsize=13, fontweight="bold", color=DARK)
ax_xa.set_xlim(0, 10); ax_xa.set_ylim(0, 10); ax_xa.axis("off")

# Text tokens on the left
tokens = ["[a]", "[red]", "[fox]", "[jumps]"]
for i, t in enumerate(tokens):
    y = 8.0 - i * 1.5
    ax_xa.add_patch(FancyBboxPatch((0.4, y - 0.35), 1.8, 0.7,
                                   boxstyle="round,pad=0.05",
                                   facecolor=GREEN, edgecolor="white", lw=1.5))
    ax_xa.text(1.3, y, t, ha="center", va="center",
               color="white", fontweight="bold", fontsize=10)

# UNet latent positions
latent_positions = [(6.0, 8.0), (6.0, 6.5), (6.0, 5.0), (6.0, 3.5),
                    (7.5, 8.0), (7.5, 6.5), (7.5, 5.0), (7.5, 3.5),
                    (9.0, 8.0), (9.0, 6.5), (9.0, 5.0), (9.0, 3.5)]
for x, y in latent_positions:
    ax_xa.add_patch(plt.Circle((x, y), 0.2, facecolor=PURPLE,
                               edgecolor="white", lw=1))

# Draw attention for "fox" token (y=5.0) -> multiple latent positions
rng = np.random.default_rng(9)
for lx, ly in latent_positions:
    w = rng.uniform(0.0, 1.0)
    if w < 0.5:
        continue
    ax_xa.plot([2.2, lx - 0.2], [5.0, ly], "-",
               color=ORANGE, lw=0.4 + w*1.3, alpha=0.3 + w*0.5)

ax_xa.text(7.5, 2.2, "latent pixels", ha="center",
           fontsize=10, fontweight="bold", color=PURPLE)
ax_xa.text(1.3, 0.6, "text embeddings\n(fixed per prompt)",
           ha="center", fontsize=9, color=GREEN)
ax_xa.text(7.5, 0.6,
           "each pixel attends to every token\n(Q = pixel, K/V = text)",
           ha="center", fontsize=9, color=DARK)

# ═══════════════════════ PANEL 4 — ControlNet family ═══════════════
ax_ctrl.set_title("4 · Beyond Text — Structural Conditioning",
                  fontsize=13, fontweight="bold", color=DARK)
ax_ctrl.set_xlim(0, 10); ax_ctrl.set_ylim(0, 10); ax_ctrl.axis("off")

cards = [
    (8.2, "ControlNet",    BLUE,
     "canny / depth / pose\n-> constrain composition"),
    (6.3, "T2I-Adapter",   GREEN,
     "lighter than ControlNet\nsame idea, ~1/5 size"),
    (4.4, "IP-Adapter",    PURPLE,
     "image as a visual prompt\n(style / subject transfer)"),
    (2.5, "LoRA",          ORANGE,
     "low-rank weight patch\nstyles, characters, concepts"),
]
for y, name, c, body in cards:
    ax_ctrl.add_patch(FancyBboxPatch((0.5, y - 0.8), 9.0, 1.6,
                                     boxstyle="round,pad=0.05",
                                     facecolor=c, edgecolor="white", lw=2))
    ax_ctrl.text(2.2, y, name, ha="center", va="center",
                 color="white", fontweight="bold", fontsize=11)
    ax_ctrl.text(6.3, y, body, ha="center", va="center",
                 color="white", fontsize=9.5)

ax_ctrl.text(5, 0.8,
             "Stack them: text + ControlNet + LoRA at inference time.",
             ha="center", fontsize=9, color="#555", style="italic")

out = str(Path(__file__).resolve().parent.parent / "img" / "Guidance Conditioning.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out}")
