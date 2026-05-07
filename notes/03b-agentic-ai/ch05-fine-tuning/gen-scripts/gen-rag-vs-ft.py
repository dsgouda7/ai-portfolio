"""Generate RAG-vs-FT.png — 4-panel xkcd-style comparison figure.

Panels:
  (1) RAG-only architecture   — frozen model, retrieval feeds context
  (2) Fine-tuning architecture — weight update, no retrieval needed
  (3) Combined pattern        — fine-tuned model + RAG on top
  (4) Decision matrix         — failure mode → right tool
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.xkcd(scale=0.8, length=100, randomness=2)

fig = plt.figure(figsize=(18, 11), facecolor="white")
fig.suptitle("RAG vs Fine-Tuning — What Each Approach Actually Changes",
             fontsize=20, fontweight="bold", y=0.98)

gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.32,
                      left=0.04, right=0.97, top=0.91, bottom=0.05)
ax_rag  = fig.add_subplot(gs[0, 0])
ax_ft   = fig.add_subplot(gs[0, 1])
ax_both = fig.add_subplot(gs[1, 0])
ax_dec  = fig.add_subplot(gs[1, 1])

BLUE, ORANGE, GREEN, RED, PURPLE, DARK, GREY, TEAL = (
    "#2E86C1", "#E67E22", "#27AE60", "#E74C3C",
    "#8E44AD", "#2C3E50", "#BDC3C7", "#17A589")


def box(ax, x, y, txt, color, w=2.6, h=0.85, fontsize=9):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.08",
                                facecolor=color, edgecolor="white", lw=2,
                                zorder=3))
    ax.text(x, y, txt, ha="center", va="center",
            color="white", fontweight="bold", fontsize=fontsize, zorder=4)


def arrow(ax, x1, y1, x2, y2, color=DARK):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8),
                zorder=2)


# ═══════════════════════════════════════════════════════════════════
# PANEL 1 — RAG-only
# ═══════════════════════════════════════════════════════════════════
ax_rag.set_title("1 · RAG — Model Weights Stay Frozen",
                 fontsize=12, fontweight="bold", color=DARK)
ax_rag.set_xlim(0, 10); ax_rag.set_ylim(0, 10); ax_rag.axis("off")

# corpus
ax_rag.add_patch(FancyBboxPatch((0.2, 6.8), 2.2, 2.4,
                                boxstyle="round,pad=0.1",
                                facecolor=BLUE, edgecolor="white", lw=2))
ax_rag.text(1.3, 8.0, "Private\nCorpus", ha="center", va="center",
            color="white", fontweight="bold", fontsize=10)
ax_rag.text(1.3, 7.0, "menu, FAQs,\ndocs…", ha="center", va="center",
            color="white", fontsize=8)

# retriever
box(ax_rag, 5.0, 8.0, "Retriever\n(vector search)", TEAL, w=2.6)
arrow(ax_rag, 2.4, 8.0, 3.7, 8.0)

# chunks → context
box(ax_rag, 5.0, 6.0, "Relevant\nchunks", ORANGE, w=2.2, h=0.8)
arrow(ax_rag, 5.0, 7.55, 5.0, 6.4)

# user query
ax_rag.text(0.5, 4.3, "User\nquery", ha="center", va="center",
            fontsize=9, color=DARK, fontweight="bold")
arrow(ax_rag, 1.3, 4.3, 3.5, 5.1, color=GREEN)

# LLM (frozen)
box(ax_rag, 5.0, 4.3, "LLM\n(FROZEN)", GREY, w=2.4)
arrow(ax_rag, 3.75, 5.6, 4.5, 4.7)   # chunks into LLM
arrow(ax_rag, 2.6, 4.3, 3.7, 4.3)    # query into LLM

# answer
box(ax_rag, 8.5, 4.3, "Answer", GREEN, w=1.8, h=0.8)
arrow(ax_rag, 6.2, 4.3, 7.6, 4.3)

# frozen annotation
ax_rag.text(5.0, 3.1, "⚠ weights never change", ha="center",
            fontsize=9, color=GREY, style="italic")
ax_rag.text(5.0, 2.4,
            "Same base model on every query.\nContext window is the only lever.",
            ha="center", fontsize=8.5, color=DARK)

# ═══════════════════════════════════════════════════════════════════
# PANEL 2 — Fine-Tuning-only
# ═══════════════════════════════════════════════════════════════════
ax_ft.set_title("2 · Fine-Tuning — Weights Updated Once",
                fontsize=12, fontweight="bold", color=DARK)
ax_ft.set_xlim(0, 10); ax_ft.set_ylim(0, 10); ax_ft.axis("off")

# training pairs
ax_ft.add_patch(FancyBboxPatch((0.2, 7.0), 2.4, 2.0,
                               boxstyle="round,pad=0.1",
                               facecolor=PURPLE, edgecolor="white", lw=2))
ax_ft.text(1.4, 8.0, "Training\npairs", ha="center", va="center",
           color="white", fontweight="bold", fontsize=10)
ax_ft.text(1.4, 7.2, "(prompt, target)", ha="center",
           color="white", fontsize=8, va="center")

# training / gradient step
box(ax_ft, 5.2, 8.0, "Gradient update\nW' = W + ΔW (LoRA)", ORANGE, w=3.4)
arrow(ax_ft, 2.65, 8.0, 3.5, 8.0)

# adapted LLM
box(ax_ft, 5.2, 5.5, "Adapted LLM\n(LoRA adapter\non frozen base)", RED, w=3.0, h=1.4)
arrow(ax_ft, 5.2, 7.55, 5.2, 6.2, color=ORANGE)

# user query → adapted LLM → answer (at inference)
ax_ft.text(0.5, 4.0, "User\nquery", ha="center", fontsize=9,
           color=DARK, fontweight="bold")
arrow(ax_ft, 1.4, 4.0, 3.65, 5.0, color=GREEN)

box(ax_ft, 8.5, 4.3, "Answer", GREEN, w=1.8, h=0.8)
arrow(ax_ft, 6.7, 5.5, 8.2, 4.6)

# changed annotation
ax_ft.text(5.2, 3.2, "✓ weights permanently changed", ha="center",
           fontsize=9, color=RED, style="italic", fontweight="bold")
ax_ft.text(5.2, 2.5,
           "Tone, format, reasoning style baked in.\nNo retrieval needed for behaviour.",
           ha="center", fontsize=8.5, color=DARK)

# ═══════════════════════════════════════════════════════════════════
# PANEL 3 — Combined
# ═══════════════════════════════════════════════════════════════════
ax_both.set_title("3 · Combined — Fine-Tuned Model + RAG",
                  fontsize=12, fontweight="bold", color=DARK)
ax_both.set_xlim(0, 10); ax_both.set_ylim(0, 10); ax_both.axis("off")

# corpus (smaller)
ax_both.add_patch(FancyBboxPatch((0.1, 7.0), 1.9, 2.2,
                                 boxstyle="round,pad=0.08",
                                 facecolor=BLUE, edgecolor="white", lw=2))
ax_both.text(1.0, 8.1, "Private\nCorpus", ha="center", va="center",
             color="white", fontweight="bold", fontsize=9)

box(ax_both, 4.0, 8.1, "Retriever", TEAL, w=2.0, h=0.8)
arrow(ax_both, 2.0, 8.1, 3.0, 8.1)

box(ax_both, 7.0, 8.1, "Chunks", ORANGE, w=1.8, h=0.8)
arrow(ax_both, 5.0, 8.1, 6.1, 8.1)

# fine-tuned LLM (bigger box with double border to show its special nature)
ax_both.add_patch(FancyBboxPatch((2.5, 3.5), 5.0, 2.5,
                                 boxstyle="round,pad=0.12",
                                 facecolor=RED, edgecolor=ORANGE, lw=3, zorder=3))
ax_both.text(5.0, 5.0, "Fine-tuned LLM", ha="center", va="center",
             color="white", fontweight="bold", fontsize=11, zorder=4)
ax_both.text(5.0, 4.3, "(LoRA adapter on frozen base)", ha="center",
             va="center", color="white", fontsize=8.5, zorder=4)
ax_both.text(5.0, 3.7, "knows HOW to respond", ha="center",
             va="center", color="#FFD", fontsize=8, style="italic", zorder=4)

arrow(ax_both, 7.0, 7.7, 6.5, 6.0)   # chunks → LLM

# user query
ax_both.text(0.5, 2.5, "User\nquery", ha="center", fontsize=9,
             color=DARK, fontweight="bold")
arrow(ax_both, 1.4, 2.5, 2.5, 4.2, color=GREEN)

# answer
box(ax_both, 8.5, 3.7, "Answer", GREEN, w=1.8, h=0.8)
arrow(ax_both, 7.5, 4.5, 8.1, 3.9)

# labels
ax_both.text(5.0, 1.8, "RAG supplies:  WHAT to say  (live facts, private data)",
             ha="center", fontsize=8.5, color=TEAL, fontweight="bold")
ax_both.text(5.0, 1.1, "Fine-tuning gives:  HOW to say it  (tone, format, style)",
             ha="center", fontsize=8.5, color=RED, fontweight="bold")
ax_both.text(5.0, 0.4,
             "Example — PizzaBot: RAG = today's menu prices;  Fine-tune = Mamma Rosa voice",
             ha="center", fontsize=8, color="#555", style="italic")

# ═══════════════════════════════════════════════════════════════════
# PANEL 4 — Decision matrix
# ═══════════════════════════════════════════════════════════════════
ax_dec.set_title("4 · Failure Mode → Right Tool",
                 fontsize=12, fontweight="bold", color=DARK)
ax_dec.set_xlim(0, 10); ax_dec.set_ylim(0, 10); ax_dec.axis("off")

rows = [
    ("Hallucinated facts / wrong prices",     "RAG",           TEAL),
    ("Stale knowledge / data cutoff",         "RAG",           TEAL),
    ("Live inventory / private documents",    "RAG",           TEAL),
    ("Generic tone despite system prompt",    "Fine-tune",     RED),
    ("Model ignores retrieved context",       "Fine-tune",     RED),
    ("Proprietary DSL / internal notation",   "Fine-tune",     RED),
    ("Too slow / too expensive at scale",     "Distillation",  ORANGE),
    ("Wrong format (JSON/XML schema)",        "Struct. output\nor fine-tune", PURPLE),
]

# header
ax_dec.add_patch(FancyBboxPatch((0.3, 8.8), 5.8, 0.7,
                                boxstyle="round,pad=0.05",
                                facecolor=DARK, edgecolor="white", lw=1.5))
ax_dec.text(3.2, 9.15, "Failure mode", ha="center", va="center",
            color="white", fontweight="bold", fontsize=9)
ax_dec.add_patch(FancyBboxPatch((6.2, 8.8), 3.4, 0.7,
                                boxstyle="round,pad=0.05",
                                facecolor=DARK, edgecolor="white", lw=1.5))
ax_dec.text(7.9, 9.15, "Tool", ha="center", va="center",
            color="white", fontweight="bold", fontsize=9)

for i, (problem, tool, tcol) in enumerate(rows):
    y = 8.1 - i * 0.95
    bg = "#F8F9FA" if i % 2 == 0 else "#EAECEE"
    ax_dec.add_patch(FancyBboxPatch((0.3, y - 0.35), 5.8, 0.7,
                                    boxstyle="round,pad=0.04",
                                    facecolor=bg, edgecolor="#CCC", lw=0.8))
    ax_dec.text(3.2, y, problem, ha="center", va="center",
                fontsize=8, color=DARK)
    ax_dec.add_patch(FancyBboxPatch((6.2, y - 0.35), 3.4, 0.7,
                                    boxstyle="round,pad=0.04",
                                    facecolor=tcol, edgecolor="white", lw=1.5))
    ax_dec.text(7.9, y, tool, ha="center", va="center",
                color="white", fontweight="bold", fontsize=8.5)

import pathlib
out = pathlib.Path(__file__).parent / "RAG-vs-FT.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
