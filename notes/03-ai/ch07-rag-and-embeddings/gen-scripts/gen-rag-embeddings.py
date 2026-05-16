from pathlib import Path
"""Generate RAG and Embeddings.png — 4-panel xkcd-style concept sketch.

Panels:
  (1) Embedding space — semantic clusters in 2D
  (2) Ingestion pipeline: docs -> chunks -> embed -> index
  (3) Query pipeline: query -> embed -> retrieve -> rerank -> LLM
  (4) Chunk size vs retrieval quality tradeoff
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.xkcd(scale=0.8, length=100, randomness=2)

fig = plt.figure(figsize=(18, 11), facecolor="white")
fig.suptitle("RAG & Embeddings", fontsize=22, fontweight="bold", y=0.98)

gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.28,
                      left=0.05, right=0.97, top=0.91, bottom=0.06)
ax_emb = fig.add_subplot(gs[0, 0])
ax_ing = fig.add_subplot(gs[0, 1])
ax_qry = fig.add_subplot(gs[1, 0])
ax_chk = fig.add_subplot(gs[1, 1])

BLUE, ORANGE, GREEN, RED, PURPLE, DARK, GREY = (
    "#2E86C1", "#E67E22", "#27AE60", "#E74C3C", "#8E44AD", "#2C3E50", "#BDC3C7")

# ═══════════════════════ PANEL 1 — Embedding space ══════════════════
ax_emb.set_title("1 · Embedding Space (2D projection)", fontsize=13,
                 fontweight="bold", color=DARK)
np.random.seed(7)
clusters = {
    "pizza": (2.5, 7.5, BLUE, ["margherita", "pepperoni", "crust", "dough"]),
    "drinks": (7.5, 7.5, ORANGE, ["coke", "sprite", "water", "juice"]),
    "allergen": (2.5, 2.5, RED, ["gluten", "nuts", "dairy", "egg"]),
    "delivery": (7.5, 2.5, GREEN, ["rider", "eta", "address", "tip"]),
}
for name, (cx, cy, c, items) in clusters.items():
    for w in items:
        x = cx + np.random.normal(0, 0.55)
        y = cy + np.random.normal(0, 0.55)
        ax_emb.plot(x, y, "o", color=c, markersize=14,
                    markeredgecolor="white", markeredgewidth=1.5)
        ax_emb.text(x, y - 0.45, w, fontsize=8, ha="center", color=DARK)
    ax_emb.text(cx, cy + 1.4, name, fontsize=11, fontweight="bold",
                color=c, ha="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor=c, lw=1))
ax_emb.set_xlim(0, 10); ax_emb.set_ylim(0, 10)
ax_emb.set_xticks([]); ax_emb.set_yticks([])
ax_emb.set_xlabel("dim 1", fontsize=9); ax_emb.set_ylabel("dim 2", fontsize=9)
ax_emb.text(5, 0.4,
            "Semantically similar text lands nearby. Cosine/dot = similarity.",
            ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 2 — Ingestion pipeline ═══════════════
ax_ing.set_title("2 · Ingestion-Time Pipeline", fontsize=13,
                 fontweight="bold", color=DARK)
ax_ing.set_xlim(0, 12); ax_ing.set_ylim(0, 6); ax_ing.axis("off")

steps = [
    (0.5,  "Docs",     BLUE,   "PDF/HTML\nmenu.md"),
    (3.0,  "Chunk",    ORANGE, "~300 tok\n+ overlap"),
    (5.8,  "Embed",    PURPLE, "encoder ->\nd-dim vec"),
    (8.6,  "Index",    GREEN,  "HNSW / IVF\n+ metadata"),
    (11.0, "Store",    DARK,   "vector DB"),
]
for i, (x, name, c, body) in enumerate(steps):
    ax_ing.add_patch(FancyBboxPatch((x, 2.2), 1.5, 2.2,
                                    boxstyle="round,pad=0.08",
                                    facecolor=c, edgecolor="white", lw=2))
    ax_ing.text(x + 0.75, 3.9, name, ha="center", va="center",
                color="white", fontweight="bold", fontsize=11)
    ax_ing.text(x + 0.75, 2.8, body, ha="center", va="center",
                color="white", fontsize=8)
    if i < len(steps) - 1:
        x_next = steps[i + 1][0]
        ax_ing.annotate("", xy=(x_next - 0.05, 3.3),
                        xytext=(x + 1.55, 3.3),
                        arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.8))
ax_ing.text(6, 0.8,
            "One-off batch job. Quality decisions made here "
            "(chunking, encoder) drive every query forever.",
            ha="center", fontsize=9.5, color="#555", style="italic")
ax_ing.text(6, 5.3, "offline", ha="center", fontsize=10,
            fontweight="bold", color=DARK,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FEF9E7",
                      edgecolor=ORANGE, lw=1))

# ═══════════════════════ PANEL 3 — Query pipeline ═══════════════════
ax_qry.set_title("3 · Query-Time Pipeline", fontsize=13,
                 fontweight="bold", color=DARK)
ax_qry.set_xlim(0, 12); ax_qry.set_ylim(0, 6); ax_qry.axis("off")

qsteps = [
    (0.3,  "Query",    BLUE,   '"gluten-\nfree?"'),
    (2.6,  "Embed",    PURPLE, "same encoder"),
    (5.0,  "Retrieve", GREEN,  "top-k ANN\n(k=20)"),
    (7.6,  "Rerank",   ORANGE, "cross-\nencoder -> top-5"),
    (10.2, "LLM",      DARK,   "generate with\ncontext"),
]
for i, (x, name, c, body) in enumerate(qsteps):
    ax_qry.add_patch(FancyBboxPatch((x, 2.2), 1.6, 2.2,
                                    boxstyle="round,pad=0.08",
                                    facecolor=c, edgecolor="white", lw=2))
    ax_qry.text(x + 0.8, 3.9, name, ha="center", va="center",
                color="white", fontweight="bold", fontsize=11)
    ax_qry.text(x + 0.8, 2.8, body, ha="center", va="center",
                color="white", fontsize=8)
    if i < len(qsteps) - 1:
        x_next = qsteps[i + 1][0]
        ax_qry.annotate("", xy=(x_next - 0.05, 3.3),
                        xytext=(x + 1.65, 3.3),
                        arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.8))
ax_qry.text(6, 0.8,
            "Runs on every user turn.\n"
            "Rerank is optional but usually worth it for top-tier precision.",
            ha="center", fontsize=9.5, color="#555", style="italic")
ax_qry.text(6, 5.3, "online", ha="center", fontsize=10,
            fontweight="bold", color=DARK,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#D5F5E3",
                      edgecolor=GREEN, lw=1))

# ═══════════════════════ PANEL 4 — Chunk size tradeoff ══════════════
ax_chk.set_title("4 · Chunk Size Tradeoff", fontsize=13,
                 fontweight="bold", color=DARK)
sizes = np.linspace(50, 2000, 100)
# precision: peaks small, drops as chunks get large (noisy)
precision = 0.9 * np.exp(-((sizes - 250) / 600) ** 2) + 0.2
# recall: low for tiny chunks (context fragmented), rises
recall = 0.3 + 0.55 * (1 - np.exp(-sizes / 500))
# overall
overall = 0.5 * precision + 0.5 * recall

ax_chk.plot(sizes, precision, "-", color=BLUE, lw=2.5, label="precision")
ax_chk.plot(sizes, recall, "-", color=ORANGE, lw=2.5, label="recall")
ax_chk.plot(sizes, overall, "--", color=GREEN, lw=2.5, label="overall")
# sweet spot
sweet = sizes[np.argmax(overall)]
ax_chk.axvline(sweet, color=RED, ls=":", lw=2)
ax_chk.text(sweet + 30, 0.4, f"sweet\nspot\n~{int(sweet)} tok",
            color=RED, fontsize=9, fontweight="bold")
ax_chk.set_xlabel("chunk size (tokens)", fontsize=10, color=DARK)
ax_chk.set_ylabel("quality", fontsize=10, color=DARK)
ax_chk.set_xlim(0, 2000); ax_chk.set_ylim(0.15, 1.0)
ax_chk.legend(fontsize=9, loc="lower right", framealpha=0.9)
ax_chk.text(0.5, -0.28,
            "Small chunks = precise but fragmented.\n"
            "Large chunks = full context but diluted relevance.\n"
            "Typical sweet spot: 200-500 tokens with 10-20% overlap.",
            transform=ax_chk.transAxes, ha="center",
            fontsize=9, color="#555", style="italic")

out = str(Path(__file__).resolve().parent.parent / "img" / "RAG and Embeddings.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out}")
