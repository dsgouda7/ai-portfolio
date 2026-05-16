from pathlib import Path
"""Generate LLM Fundamentals.png — 4-panel xkcd-style concept sketch.

Panels:
  (1) BPE tokenisation of a sentence
  (2) Sampling: temperature + top-p on a probability distribution
  (3) Three training stages pipeline (pretrain -> SFT -> RLHF/DPO)
  (4) Context window + lost-in-the-middle recall curve
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Gentle xkcd: keep sketchy look but keep text legible (no stroke outlines).
plt.xkcd(scale=0.3, length=100, randomness=1)
plt.rcParams["path.effects"] = []

fig = plt.figure(figsize=(18, 11), facecolor="white")
fig.suptitle("LLM Fundamentals", fontsize=22, fontweight="bold", y=0.98)

gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.28,
                      left=0.06, right=0.97, top=0.91, bottom=0.06)
ax_tok = fig.add_subplot(gs[0, 0])
ax_sam = fig.add_subplot(gs[0, 1])
ax_trn = fig.add_subplot(gs[1, 0])
ax_ctx = fig.add_subplot(gs[1, 1])

BLUE, ORANGE, GREEN, RED, PURPLE, DARK, GREY = (
    "#2E86C1", "#E67E22", "#27AE60", "#E74C3C", "#8E44AD", "#2C3E50", "#BDC3C7")

# ═══════════════════════ PANEL 1 — Tokenisation (BPE) ═══════════════
ax_tok.set_title("1 · BPE Tokenisation", fontsize=13, fontweight="bold", color=DARK)
ax_tok.set_xlim(0, 10); ax_tok.set_ylim(0, 6); ax_tok.axis("off")

sentence = "tokenisation is hard"
ax_tok.text(5, 5.3, f'"{sentence}"', ha="center", fontsize=13,
            fontweight="bold", color=DARK)
ax_tok.annotate("", xy=(5, 4.0), xytext=(5, 4.8),
                arrowprops=dict(arrowstyle="-|>", color=DARK, lw=2))
tokens = [("to", BLUE), ("ken", ORANGE), ("isation", PURPLE),
          (" is", BLUE), (" hard", GREEN)]
x = 0.6
for t, c in tokens:
    w = max(0.9, len(t) * 0.35)
    ax_tok.add_patch(FancyBboxPatch((x, 2.8), w, 0.9,
                                    boxstyle="round,pad=0.05",
                                    facecolor=c, edgecolor="white", lw=2))
    ax_tok.text(x + w / 2, 3.25, t, ha="center", va="center",
                color="white", fontweight="bold", fontsize=10)
    x += w + 0.15
ax_tok.text(5, 2.0, "5 tokens ~= 4 words  (1 tok ~= 0.75 word)",
            ha="center", fontsize=10, color=DARK, style="italic")
ax_tok.text(5, 1.2,
            "Byte-Pair Encoding merges frequent adjacent pairs\n"
            "until vocabulary reaches 32k-100k tokens.",
            ha="center", fontsize=9.5, color="#555",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F8F9F9",
                      edgecolor=GREY, lw=0.8))

# ═══════════════════════ PANEL 2 — Sampling ═════════════════════════
ax_sam.set_title("2 · Sampling — Temperature & Top-p", fontsize=13,
                 fontweight="bold", color=DARK)
vocab = ["the", "a", "cat", "dog", "runs", "fast", "slow", "quietly"]
logits = np.array([3.2, 2.5, 1.8, 1.4, 0.9, 0.6, 0.2, -0.4])
x_pos = np.arange(len(vocab))

def softmax(z, T=1.0):
    e = np.exp(z / T); return e / e.sum()

p_hi = softmax(logits, T=0.3)   # sharp (low T)
p_lo = softmax(logits, T=1.5)   # flat (high T)
w = 0.38
ax_sam.bar(x_pos - w/2, p_hi, w, color=BLUE, edgecolor="white",
           label="T=0.3 (sharp)")
ax_sam.bar(x_pos + w/2, p_lo, w, color=ORANGE, edgecolor="white",
           label="T=1.5 (flat)")
# top-p=0.9 cutoff shading on p_hi
cum = np.cumsum(p_hi); cutoff = np.searchsorted(cum, 0.9) + 1
ax_sam.axvspan(-0.5, cutoff - 0.5, ymin=0, ymax=1, color=GREEN, alpha=0.12)
ax_sam.text(cutoff - 0.5, 0.55, "top-p=0.9\nnucleus",
            fontsize=9, color=GREEN, fontweight="bold", ha="left")
ax_sam.set_xticks(x_pos); ax_sam.set_xticklabels(vocab, rotation=25, fontsize=9)
ax_sam.set_ylabel("P(token)", fontsize=10, color=DARK)
ax_sam.set_ylim(0, 0.7)
ax_sam.legend(fontsize=9, loc="upper right", framealpha=0.9)
ax_sam.text(0.5, -0.34,
            "Low T -> deterministic | High T -> creative\n"
            "Top-p truncates tail; combine both in production.",
            transform=ax_sam.transAxes, ha="center", fontsize=9,
            color="#555", style="italic")

# ═══════════════════════ PANEL 3 — Training stages ══════════════════
ax_trn.set_title("3 · Three Training Stages", fontsize=13,
                 fontweight="bold", color=DARK)
ax_trn.set_xlim(0, 12); ax_trn.set_ylim(0, 6); ax_trn.axis("off")

stages = [
    (1.2,  "Pretraining",     "10T+ tokens\ncross-entropy\nnext-token",  BLUE),
    (5.2,  "SFT",             "(instruction,\nresponse) pairs\nfollows format", ORANGE),
    (9.2,  "RLHF / DPO",      "preference\npairs -> helpful\nharmless honest", GREEN),
]
for x0, name, body, color in stages:
    ax_trn.add_patch(FancyBboxPatch((x0, 1.8), 2.4, 2.6,
                                    boxstyle="round,pad=0.08",
                                    facecolor=color, edgecolor="white", lw=2))
    ax_trn.text(x0 + 1.2, 3.9, name, ha="center", va="center",
                color="white", fontweight="bold", fontsize=12)
    ax_trn.text(x0 + 1.2, 2.7, body, ha="center", va="center",
                color="white", fontsize=9)
for x0 in (3.7, 7.7):
    ax_trn.annotate("", xy=(x0 + 1.3, 3.1), xytext=(x0, 3.1),
                    arrowprops=dict(arrowstyle="-|>", color=DARK, lw=2))
ax_trn.text(1.2 + 1.2, 1.2, "raw LM", ha="center", fontsize=9, color="#555")
ax_trn.text(5.2 + 1.2, 1.2, "instruct model", ha="center", fontsize=9, color="#555")
ax_trn.text(9.2 + 1.2, 1.2, "chat model", ha="center", fontsize=9, color="#555")
ax_trn.text(6, 5.2,
            "Each stage adds behaviour without erasing the previous.",
            ha="center", fontsize=10, color=DARK, style="italic")
ax_trn.text(6, 0.3,
            "SFT teaches format+tone. RLHF/DPO aligns with preferences "
            "(and introduces sycophancy risk).",
            ha="center", fontsize=9, color="#555")

# ═══════════════════════ PANEL 4 — Context window ═══════════════════
ax_ctx.set_title("4 · Context Window & Lost-in-the-Middle",
                 fontsize=13, fontweight="bold", color=DARK)
pos = np.linspace(0, 1, 100)
recall = 0.45 + 0.45 * (np.exp(-((pos - 0.0) / 0.18) ** 2) +
                       np.exp(-((pos - 1.0) / 0.18) ** 2))
ax_ctx.plot(pos, recall, "-", color=BLUE, lw=3, label="recall accuracy")
ax_ctx.fill_between(pos, recall, 0.3, alpha=0.15, color=BLUE)
ax_ctx.axhline(0.5, color=RED, ls="--", lw=1.5, alpha=0.6)
ax_ctx.text(0.5, 0.48, "middle drop", color=RED, fontsize=10,
            ha="center", fontweight="bold")
ax_ctx.set_xlabel("fact position in context  (0=start, 1=end)",
                  fontsize=10, color=DARK)
ax_ctx.set_ylabel("recall", fontsize=10, color=DARK)
ax_ctx.set_xlim(0, 1); ax_ctx.set_ylim(0.3, 1.0)
ax_ctx.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax_ctx.set_xticklabels(["start", "", "middle", "", "end"])
ax_ctx.text(0.5, 0.95,
            "4k -> 200k -> 1M tokens\n"
            "longer != uniform recall",
            ha="center", fontsize=10, color=DARK,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF9E7",
                      edgecolor=ORANGE, lw=1))

out = str(Path(__file__).resolve().parent.parent / "img" / "LLM Fundamentals.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out}")
