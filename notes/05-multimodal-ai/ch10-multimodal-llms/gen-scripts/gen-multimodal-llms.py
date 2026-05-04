from pathlib import Path
"""Generate Multimodal LLMs.png — 4-panel xkcd-style concept sketch.

Panels:
  (1) MM-LLM architecture: vision encoder + projector + LLM
  (2) Token-level interleaving of image + text tokens
  (3) Training stages: pretrain / align / SFT / RLHF
  (4) Capability matrix: VQA / OCR / doc / chart / grounding
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.xkcd(scale=0.8, length=100, randomness=2)

fig = plt.figure(figsize=(18, 11), facecolor="white")
fig.suptitle("Multimodal LLMs — When Language Models Can See",
             fontsize=20, fontweight="bold", y=0.98)

gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.28,
                      left=0.05, right=0.97, top=0.91, bottom=0.06)
ax_arch = fig.add_subplot(gs[0, 0])
ax_int = fig.add_subplot(gs[0, 1])
ax_stg = fig.add_subplot(gs[1, 0])
ax_cap = fig.add_subplot(gs[1, 1])

BLUE, ORANGE, GREEN, RED, PURPLE, DARK, GREY = (
    "#2E86C1", "#E67E22", "#27AE60", "#E74C3C", "#8E44AD", "#2C3E50", "#BDC3C7")

# ═══════════════════════ PANEL 1 — Architecture ════════════════════
ax_arch.set_title("1 · Architecture — Vision + Projector + LLM",
                  fontsize=13, fontweight="bold", color=DARK)
ax_arch.set_xlim(0, 10); ax_arch.set_ylim(0, 10); ax_arch.axis("off")

# image in
ax_arch.add_patch(FancyBboxPatch((0.3, 7.0), 1.8, 1.6,
                                 boxstyle="round,pad=0.05",
                                 facecolor=BLUE, edgecolor="white", lw=2))
ax_arch.text(1.2, 7.8, "image", ha="center", va="center",
             color="white", fontweight="bold", fontsize=11)

ax_arch.annotate("", xy=(3.2, 7.8), xytext=(2.2, 7.8),
                 arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.4))

# vision encoder
ax_arch.add_patch(FancyBboxPatch((3.2, 7.0), 2.1, 1.6,
                                 boxstyle="round,pad=0.05",
                                 facecolor=PURPLE, edgecolor="white", lw=2))
ax_arch.text(4.25, 7.8, "ViT / CLIP-ViT\n(frozen)",
             ha="center", va="center", color="white",
             fontweight="bold", fontsize=9.5)

ax_arch.annotate("", xy=(6.4, 7.8), xytext=(5.4, 7.8),
                 arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.4))

# projector
ax_arch.add_patch(FancyBboxPatch((6.4, 7.0), 3.4, 1.6,
                                 boxstyle="round,pad=0.05",
                                 facecolor=ORANGE, edgecolor="white", lw=2))
ax_arch.text(8.1, 7.8, "projector  (MLP / Q-Former)",
             ha="center", va="center", color="white",
             fontweight="bold", fontsize=9.5)

# LLM
ax_arch.add_patch(FancyBboxPatch((1.5, 3.0), 7.0, 1.8,
                                 boxstyle="round,pad=0.1",
                                 facecolor=GREEN, edgecolor="white", lw=2))
ax_arch.text(5, 3.9, "LLM  (Llama / Mistral / ...)",
             ha="center", va="center", color="white",
             fontweight="bold", fontsize=12)

ax_arch.annotate("", xy=(5, 4.8), xytext=(5, 6.9),
                 arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.5))
ax_arch.text(5.6, 5.8, "image tokens", fontsize=9, color=DARK)

# text
ax_arch.add_patch(FancyBboxPatch((0.6, 0.8), 4.0, 1.3,
                                 boxstyle="round,pad=0.05",
                                 facecolor=BLUE, edgecolor="white", lw=2))
ax_arch.text(2.6, 1.45, '"what do you see?"',
             ha="center", va="center", color="white",
             fontweight="bold", fontsize=10)
ax_arch.annotate("", xy=(2.6, 2.9), xytext=(2.6, 2.1),
                 arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.4))

# output
ax_arch.add_patch(FancyBboxPatch((5.4, 0.8), 4.0, 1.3,
                                 boxstyle="round,pad=0.05",
                                 facecolor=RED, edgecolor="white", lw=2))
ax_arch.text(7.4, 1.45, '"a red fox in snow..."',
             ha="center", va="center", color="white",
             fontweight="bold", fontsize=10)
ax_arch.annotate("", xy=(7.4, 0.8 + 1.3 + 0.1 + 0.6), xytext=(7.4, 2.95),
                 arrowprops=dict(arrowstyle="-", color=DARK, lw=0.0))
ax_arch.annotate("", xy=(7.4, 2.15), xytext=(7.4, 2.9),
                 arrowprops=dict(arrowstyle="<-", color=DARK, lw=1.4))

# ═══════════════════════ PANEL 2 — Token interleave ═══════════════
ax_int.set_title("2 · Token Stream — Interleaved Image + Text",
                 fontsize=13, fontweight="bold", color=DARK)
ax_int.set_xlim(0, 10); ax_int.set_ylim(0, 10); ax_int.axis("off")

# Simulated stream
tokens = [
    ("<img>",     DARK),
    ("img_1",     BLUE), ("img_2", BLUE), ("img_3", BLUE),
    ("...",       GREY),
    ("img_576",   BLUE),
    ("</img>",    DARK),
    ("what",      GREEN), ("do",   GREEN), ("you", GREEN),
    ("see",       GREEN), ("?",    GREEN),
]
x = 0.4
for tok, c in tokens:
    w = 0.85 if len(tok) < 6 else 1.15
    ax_int.add_patch(FancyBboxPatch((x, 6.0), w, 1.0,
                                    boxstyle="round,pad=0.03",
                                    facecolor=c, edgecolor="white", lw=1.2))
    ax_int.text(x + w/2, 6.5, tok, ha="center", va="center",
                color="white", fontweight="bold", fontsize=8)
    x += w + 0.1

ax_int.text(5, 4.8,
            "The LLM consumes image features AS TOKENS, position-embedded like text.",
            ha="center", fontsize=10, color=DARK, fontweight="bold")

ax_int.text(5, 3.4, "Token budget (LLaVA-style):", ha="center",
            fontsize=10, fontweight="bold", color=DARK)
ax_int.text(5, 2.6, "1 image (336x336) -> 576 tokens",
            ha="center", fontsize=9.5, color=DARK, family="monospace")
ax_int.text(5, 1.9, "Many MM-LLMs cap images to avoid blowing context",
            ha="center", fontsize=9, color="#555", style="italic")
ax_int.text(5, 0.9,
            "Token-cost is why high-res / long-video inputs are expensive.",
            ha="center", fontsize=9, color=RED, fontweight="bold")

# ═══════════════════════ PANEL 3 — Training stages ═════════════════
ax_stg.set_title("3 · Training Stages",
                 fontsize=13, fontweight="bold", color=DARK)
ax_stg.set_xlim(0, 10); ax_stg.set_ylim(0, 10); ax_stg.axis("off")

stages = [
    (8.4, "1. Vision pretrain",     BLUE,
     "CLIP-ViT on web images"),
    (6.6, "2. Projector align",     ORANGE,
     "train projector only; LLM frozen"),
    (4.8, "3. Visual SFT",          GREEN,
     "instruction data with images"),
    (3.0, "4. Preference (RLHF/DPO)", PURPLE,
     "human rankings on VQA / safety"),
    (1.2, "5. Eval + redteam",      RED,
     "MMBench / MME / POPE / hallucination"),
]
for y, name, c, body in stages:
    ax_stg.add_patch(FancyBboxPatch((0.5, y - 0.7), 9.0, 1.4,
                                    boxstyle="round,pad=0.05",
                                    facecolor=c, edgecolor="white", lw=2))
    ax_stg.text(2.2, y, name, ha="center", va="center",
                color="white", fontweight="bold", fontsize=10.5)
    ax_stg.text(6.5, y, body, ha="center", va="center",
                color="white", fontsize=9.5)

ax_stg.text(5, 0.3,
            "Most quality gains come from stage 3's data curation.",
            ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 4 — Capabilities ═════════════════════
ax_cap.set_title("4 · What MM-LLMs Can Do",
                 fontsize=13, fontweight="bold", color=DARK)
ax_cap.set_xlim(0, 10); ax_cap.set_ylim(0, 10); ax_cap.axis("off")

caps = [
    (8.5, "VQA",        BLUE,   "'what's the fox holding?'"),
    (7.0, "OCR",        GREEN,  "read text in image"),
    (5.5, "Doc / chart",PURPLE, "tables, plots, scans"),
    (4.0, "Grounding",  ORANGE, "bboxes, click points"),
    (2.5, "Code from image", RED, "screenshot -> HTML / SVG"),
    (1.0, "Video QA (frames)", DARK, "long-context video tokens"),
]
for y, name, c, body in caps:
    ax_cap.add_patch(FancyBboxPatch((0.5, y - 0.5), 9.0, 1.0,
                                    boxstyle="round,pad=0.05",
                                    facecolor=c, edgecolor="white", lw=2))
    ax_cap.text(2.0, y, name, ha="center", va="center",
                color="white", fontweight="bold", fontsize=10)
    ax_cap.text(6.3, y, body, ha="center", va="center",
                color="white", fontsize=9.5)
ax_cap.text(5, 0.2,
            "Still weak at: counting, fine spatial reasoning, precise OCR on small text.",
            ha="center", fontsize=9, color="#555", style="italic")

out = str(Path(__file__).resolve().parent.parent / "img" / "Multimodal LLMs.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out}")
