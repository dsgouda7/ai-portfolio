from pathlib import Path
"""Generate Fine-Tuning.png — 4-panel xkcd-style concept sketch.

Panels:
  (1) Full fine-tune vs LoRA matrix decomposition W + BA
  (2) Decision tree: when to prompt / RAG / fine-tune
  (3) QLoRA stack (4-bit quantised base + LoRA adapters)
  (4) Catastrophic forgetting / overfitting pitfalls
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.xkcd(scale=0.8, length=100, randomness=2)

fig = plt.figure(figsize=(18, 11), facecolor="white")
fig.suptitle("Fine-Tuning — LoRA & QLoRA", fontsize=22, fontweight="bold", y=0.98)

gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.28,
                      left=0.05, right=0.97, top=0.91, bottom=0.06)
ax_lora = fig.add_subplot(gs[0, 0])
ax_tree = fig.add_subplot(gs[0, 1])
ax_qlora = fig.add_subplot(gs[1, 0])
ax_fail = fig.add_subplot(gs[1, 1])

BLUE, ORANGE, GREEN, RED, PURPLE, DARK, GREY = (
    "#2E86C1", "#E67E22", "#27AE60", "#E74C3C", "#8E44AD", "#2C3E50", "#BDC3C7")

# ═══════════════════════ PANEL 1 — LoRA decomposition ═══════════════
ax_lora.set_title("1 · Full FT vs LoRA — Low-Rank Adaptation",
                  fontsize=13, fontweight="bold", color=DARK)
ax_lora.set_xlim(0, 10); ax_lora.set_ylim(0, 10); ax_lora.axis("off")

# Full FT (big dense matrix)
ax_lora.text(2, 9.0, "Full Fine-Tune", ha="center", fontweight="bold",
             fontsize=11, color=RED)
ax_lora.add_patch(FancyBboxPatch((0.5, 4.0), 3.0, 4.0,
                                 boxstyle="round,pad=0.05",
                                 facecolor=RED, edgecolor="white", lw=2))
ax_lora.text(2, 6.0, "W_new\n(d x d)\n~100%\nparams", ha="center", va="center",
             color="white", fontweight="bold", fontsize=11)
ax_lora.text(2, 3.2, "7B model ->\n~14 GB grads", ha="center",
             fontsize=9, color=RED)

# LoRA: W_frozen + B (d x r) A (r x d)
ax_lora.text(7, 9.0, "LoRA:  W + B A", ha="center", fontweight="bold",
             fontsize=11, color=GREEN)
ax_lora.add_patch(FancyBboxPatch((5, 4.0), 3.0, 4.0,
                                 boxstyle="round,pad=0.05",
                                 facecolor=GREY, edgecolor="white", lw=2))
ax_lora.text(6.5, 6.0, "W\nfrozen", ha="center", va="center",
             color="white", fontweight="bold", fontsize=11)
# thin B and A
ax_lora.add_patch(FancyBboxPatch((8.2, 6.0), 0.4, 2.0,
                                 boxstyle="round,pad=0.02",
                                 facecolor=ORANGE, edgecolor="white", lw=1.5))
ax_lora.text(8.4, 7.0, "B", ha="center", va="center", color="white",
             fontweight="bold", fontsize=10)
ax_lora.add_patch(FancyBboxPatch((8.9, 7.6), 2.0, 0.4,
                                 boxstyle="round,pad=0.02",
                                 facecolor=PURPLE, edgecolor="white", lw=1.5))
ax_lora.text(9.9, 7.8, "A", ha="center", va="center", color="white",
             fontweight="bold", fontsize=10, clip_on=False)
ax_lora.text(7.5, 3.2, "r=8 -> ~0.1%\nparams trained", ha="center",
             fontsize=9, color=GREEN)

ax_lora.text(5, 1.8,
             "LoRA: dW = B A  with B in R(dxr), A in R(rxd), r<<d.",
             ha="center", fontsize=10, color=DARK)
ax_lora.text(5, 0.8,
             "Merge at inference -> zero latency cost.  Swap adapters per task.",
             ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 2 — Decision tree ════════════════════
ax_tree.set_title("2 · When to Fine-Tune — Decision Tree",
                  fontsize=13, fontweight="bold", color=DARK)
ax_tree.set_xlim(0, 10); ax_tree.set_ylim(0, 10); ax_tree.axis("off")

def dbox(x, y, txt, c, w=3.0, h=1.1):
    ax_tree.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
                                     boxstyle="round,pad=0.1",
                                     facecolor=c, edgecolor="white", lw=1.5))
    ax_tree.text(x, y, txt, ha="center", va="center",
                 color="white", fontweight="bold", fontsize=9)

dbox(5, 9.0, "Need new behaviour?", DARK)
dbox(2, 6.8, "no -> prompt eng.", BLUE)
dbox(8, 6.8, "yes: need new facts?", DARK)
dbox(5.5, 4.6, "yes -> RAG", GREEN)
dbox(9.3, 4.6, "no", DARK, w=1.4)
dbox(9.3, 2.4, "format/style\nmatter?", DARK, w=2.4, h=1.3)
dbox(7.3, 0.6, "no -> prompt", BLUE, w=2.3)
dbox(5.0, 0.6, "yes -> LoRA/QLoRA", ORANGE, w=2.6)

edges = [((5,8.5),(2,7.3)), ((5,8.5),(8,7.3)),
         ((8,6.3),(5.5,5.1)), ((8,6.3),(9.3,5.1)),
         ((9.3,4.1),(9.3,3.0)),
         ((9.3,1.8),(7.3,1.1)), ((9.3,1.8),(5.0,1.1))]
for (x1,y1),(x2,y2) in edges:
    ax_tree.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.5))

# ═══════════════════════ PANEL 3 — QLoRA stack ══════════════════════
ax_qlora.set_title("3 · QLoRA — 4-bit base + LoRA adapters",
                   fontsize=13, fontweight="bold", color=DARK)
ax_qlora.set_xlim(0, 10); ax_qlora.set_ylim(0, 10); ax_qlora.axis("off")

layers = [
    (7.0, "Base weights",    BLUE,   "W frozen, 4-bit NF4\n~3.5 GB for 7B"),
    (4.5, "Dequantise",      PURPLE, "on-the-fly -> bf16"),
    (2.0, "LoRA adapters",   ORANGE, "r=8-64 per layer\ntrainable, bf16"),
]
for y, name, c, body in layers:
    ax_qlora.add_patch(FancyBboxPatch((1.0, y - 0.9), 8.0, 1.8,
                                      boxstyle="round,pad=0.05",
                                      facecolor=c, edgecolor="white", lw=2))
    ax_qlora.text(2.5, y, name, ha="center", va="center",
                  color="white", fontweight="bold", fontsize=11)
    ax_qlora.text(6.5, y, body, ha="center", va="center",
                  color="white", fontsize=9)

ax_qlora.text(5, 0.4,
              "7B model fine-tunes in ~10 GB GPU RAM (single 3090/4090).\n"
              "Near-full-FT quality on most tasks.",
              ha="center", fontsize=9.5, color="#555", style="italic")

# ═══════════════════════ PANEL 4 — Failure modes ════════════════════
ax_fail.set_title("4 · Failure Modes", fontsize=13,
                  fontweight="bold", color=DARK)
epochs = np.linspace(0, 10, 100)
train = 1.5 * np.exp(-epochs / 2.5) + 0.1
# val: goes down then up (overfit)
val = 1.5 * np.exp(-epochs / 2.2) + 0.15 + 0.02 * np.clip(epochs - 4, 0, None) ** 1.5

ax_fail.plot(epochs, train, "-", color=BLUE, lw=2.5, label="train loss")
ax_fail.plot(epochs, val, "-", color=RED, lw=2.5, label="val loss")
best = np.argmin(val)
ax_fail.plot(epochs[best], val[best], "o", color=GREEN, markersize=14,
             markeredgecolor="white", markeredgewidth=2, zorder=5)
ax_fail.annotate("early-stop\nhere", (epochs[best], val[best]),
                 textcoords="offset points", xytext=(15, 10),
                 fontsize=10, color=GREEN, fontweight="bold")
ax_fail.axvspan(epochs[best], 10, alpha=0.12, color=RED)
ax_fail.text(8, 1.2, "overfitting /\ncatastrophic\nforgetting",
             color=RED, fontsize=10, fontweight="bold", ha="center")
ax_fail.set_xlabel("epochs", fontsize=10, color=DARK)
ax_fail.set_ylabel("loss", fontsize=10, color=DARK)
ax_fail.legend(fontsize=9, framealpha=0.9)
ax_fail.text(0.5, -0.28,
             "Always hold out a general-capability eval. "
             "LoRA forgets less than full FT.",
             transform=ax_fail.transAxes, ha="center",
             fontsize=9, color="#555", style="italic")

out = str(Path(__file__).resolve().parent.parent / "img" / "Fine-Tuning.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out}")
