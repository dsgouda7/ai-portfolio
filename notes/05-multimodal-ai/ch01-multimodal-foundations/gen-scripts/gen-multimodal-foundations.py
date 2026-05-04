from pathlib import Path
"""Generate Multimodal Foundations.png — 4-panel xkcd-style concept sketch.

Panels:
  (1) Raw signals -> tensors (image / audio / text)
  (2) Patch / window / token shapes side-by-side
  (3) Embedding projectors into a shared D-dim space
  (4) Modality alignment via contrastive pulls
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.xkcd(scale=0.8, length=100, randomness=2)

fig = plt.figure(figsize=(18, 11), facecolor="white")
fig.suptitle("Multimodal Foundations — How Raw Signals Become Tensors",
             fontsize=20, fontweight="bold", y=0.98)

gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.28,
                      left=0.05, right=0.97, top=0.91, bottom=0.06)
ax_raw = fig.add_subplot(gs[0, 0])
ax_sh = fig.add_subplot(gs[0, 1])
ax_proj = fig.add_subplot(gs[1, 0])
ax_al = fig.add_subplot(gs[1, 1])

BLUE, ORANGE, GREEN, RED, PURPLE, DARK, GREY = (
    "#2E86C1", "#E67E22", "#27AE60", "#E74C3C", "#8E44AD", "#2C3E50", "#BDC3C7")

# ═══════════════════════ PANEL 1 — Raw -> tensors ═══════════════════
ax_raw.set_title("1 · Raw Signals -> Tensors",
                 fontsize=13, fontweight="bold", color=DARK)
ax_raw.set_xlim(0, 10); ax_raw.set_ylim(0, 10); ax_raw.axis("off")

# Image
ax_raw.text(1.7, 9.2, "image", ha="center", fontweight="bold",
            color=BLUE, fontsize=11)
for i in range(4):
    for j in range(4):
        ax_raw.add_patch(plt.Rectangle((0.3 + j*0.35, 6.8 - i*0.35), 0.3, 0.3,
                                       facecolor=plt.cm.Blues(0.3 + 0.1*((i+j) % 4)),
                                       edgecolor="white", lw=0.5))
ax_raw.annotate("", xy=(3.5, 7.5), xytext=(2.0, 7.5),
                arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.5))
ax_raw.text(4.2, 7.5, "(H, W, C)", fontsize=10, color=DARK,
            family="monospace")

# Audio
ax_raw.text(1.7, 5.5, "audio", ha="center", fontweight="bold",
            color=ORANGE, fontsize=11)
t = np.linspace(0, 4, 200)
wave = 0.4 * np.sin(6*t) * np.exp(-0.3*t)
ax_raw.plot(0.3 + t*0.9, 4.3 + wave, color=ORANGE, lw=1.5)
ax_raw.annotate("", xy=(5.5, 4.3), xytext=(4.1, 4.3),
                arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.5))
ax_raw.text(6.2, 4.3, "(T, mels)", fontsize=10, color=DARK,
            family="monospace")

# Text
ax_raw.text(1.7, 2.8, "text", ha="center", fontweight="bold",
            color=GREEN, fontsize=11)
ax_raw.text(1.7, 2.0, '"a fox jumps"', ha="center", fontsize=10,
            color=DARK, family="monospace",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#EAF7EC",
                      edgecolor=GREEN))
ax_raw.annotate("", xy=(5.0, 2.0), xytext=(3.2, 2.0),
                arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.5))
ax_raw.text(6.0, 2.0, "(L,) token ids", fontsize=10, color=DARK,
            family="monospace")

ax_raw.text(5, 0.4,
            "Each modality gets its own tokenizer. Shapes differ, but all end as integer/float tensors.",
            ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 2 — Shapes ═══════════════════════════
ax_sh.set_title("2 · Shapes After Tokenization",
                fontsize=13, fontweight="bold", color=DARK)
ax_sh.set_xlim(0, 10); ax_sh.set_ylim(0, 10); ax_sh.axis("off")

rows = [
    (8.0, "image", BLUE,
     "224x224 x3  -> 14x14 patches -> 196 tokens"),
    (6.0, "audio", ORANGE,
     "30s mel      -> 3000 frames  -> 1500 tokens"),
    (4.0, "text",  GREEN,
     "\"a fox...\"  -> BPE          -> 8 tokens"),
    (2.0, "video", PURPLE,
     "T frames     -> tube-lets    -> ~T*196 tokens"),
]
for y, name, c, body in rows:
    ax_sh.add_patch(FancyBboxPatch((0.5, y - 0.7), 9.0, 1.4,
                                   boxstyle="round,pad=0.05",
                                   facecolor=c, edgecolor="white", lw=2))
    ax_sh.text(1.8, y, name, ha="center", va="center",
               color="white", fontweight="bold", fontsize=11)
    ax_sh.text(6.2, y, body, ha="center", va="center",
               color="white", fontsize=9.5, family="monospace")
ax_sh.text(5, 0.6,
           "Every modality becomes a SEQUENCE of tokens. Transformers don't care which kind.",
           ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 3 — Projection ═══════════════════════
ax_proj.set_title("3 · Projecting Into a Shared D-Dim Space",
                  fontsize=13, fontweight="bold", color=DARK)
ax_proj.set_xlim(0, 10); ax_proj.set_ylim(0, 10); ax_proj.axis("off")

boxes = [
    (1.5, 7.5, "image tokens",  BLUE),
    (1.5, 5.0, "audio tokens",  ORANGE),
    (1.5, 2.5, "text tokens",   GREEN),
]
for x, y, name, c in boxes:
    ax_proj.add_patch(FancyBboxPatch((x - 1.1, y - 0.5), 2.2, 1.0,
                                     boxstyle="round,pad=0.05",
                                     facecolor=c, edgecolor="white", lw=2))
    ax_proj.text(x, y, name, ha="center", va="center",
                 color="white", fontweight="bold", fontsize=10)

# projectors
for y in [7.5, 5.0, 2.5]:
    ax_proj.annotate("", xy=(5.0, y), xytext=(2.7, y),
                     arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.5))
    ax_proj.text(3.85, y + 0.3, "Wx", fontsize=9, color=DARK,
                 family="monospace", ha="center")

# Shared space
ax_proj.add_patch(FancyBboxPatch((5.0, 2.0), 4.6, 6.0,
                                 boxstyle="round,pad=0.15",
                                 facecolor="#F4ECF7", edgecolor=PURPLE, lw=2))
ax_proj.text(7.3, 7.5, "shared R^D", ha="center",
             color=PURPLE, fontweight="bold", fontsize=12)
rng = np.random.default_rng(7)
xs = rng.uniform(5.4, 9.2, 18); ys = rng.uniform(2.5, 7.0, 18)
cs = [BLUE]*6 + [ORANGE]*6 + [GREEN]*6
ax_proj.scatter(xs, ys, c=cs, s=60, alpha=0.85, edgecolors="white")

ax_proj.text(5, 0.8,
             "A linear (or MLP) projector per modality maps different shapes to the SAME D-dim vector.",
             ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 4 — Alignment ═══════════════════════
ax_al.set_title("4 · Alignment — Pull Matched Pairs Together",
                fontsize=13, fontweight="bold", color=DARK)
ax_al.set_xlim(0, 10); ax_al.set_ylim(0, 10); ax_al.axis("off")

rng = np.random.default_rng(3)
pairs = [
    ("fox photo",      (2.5, 7.8), BLUE,   "'a red fox'",   (4.8, 7.5),  GREEN),
    ("cat photo",      (2.2, 5.5), BLUE,   "'a ginger cat'",(5.0, 5.3),  GREEN),
    ("dog bark .wav",  (2.0, 3.2), ORANGE, "'dog barking'", (5.2, 3.0),  GREEN),
    ("rain .wav",      (2.8, 1.3), ORANGE, "'heavy rain'",  (5.5, 1.5),  GREEN),
]
for a_name, (ax1, ay1), ac, b_name, (bx1, by1), bc in pairs:
    ax_al.add_patch(plt.Circle((ax1, ay1), 0.35, facecolor=ac,
                               edgecolor="white", lw=1.5))
    ax_al.text(ax1 - 0.6, ay1 + 0.5, a_name, fontsize=8.5, color=DARK)
    ax_al.add_patch(plt.Circle((bx1, by1), 0.35, facecolor=bc,
                               edgecolor="white", lw=1.5))
    ax_al.text(bx1 + 0.4, by1 + 0.5, b_name, fontsize=8.5, color=DARK)
    ax_al.annotate("", xy=(bx1 - 0.4, by1), xytext=(ax1 + 0.4, ay1),
                   arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1.3))

# Push-apart dashed line
ax_al.plot([8.0, 9.0], [6.0, 2.0], ls="--", color=RED, lw=1.2)
ax_al.text(8.5, 4.0, "push\napart", color=RED, fontsize=9,
           fontweight="bold", ha="center", rotation=-70)

ax_al.text(5, 0.3,
           "Contrastive training aligns modalities in one geometry -> enables CLIP, cross-modal search, MM-LLMs.",
           ha="center", fontsize=9, color="#555", style="italic")

out = str(Path(__file__).resolve().parent.parent / "img" / "Multimodal Foundations.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out}")
