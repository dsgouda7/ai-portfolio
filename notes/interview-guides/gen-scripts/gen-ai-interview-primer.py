from pathlib import Path
"""Generate AI Interview Primer.png — 6-panel xkcd-style concept summary.

Panels (one per major interview section):
  (1) CoT & reasoning    (2) ReAct & agent loop
  (3) LangChain vs SK    (4) Embeddings + distance metrics
  (5) RAG pipeline       (6) Vector DB tradeoff triangle
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.xkcd(scale=0.8, length=100, randomness=2)

fig = plt.figure(figsize=(20, 13), facecolor="white")
fig.suptitle("AI Interview Primer — One-Page Summary",
             fontsize=22, fontweight="bold", y=0.985)

gs = fig.add_gridspec(3, 2, hspace=0.60, wspace=0.25,
                      left=0.05, right=0.97, top=0.93, bottom=0.05)
ax_cot = fig.add_subplot(gs[0, 0])
ax_react = fig.add_subplot(gs[0, 1])
ax_fw = fig.add_subplot(gs[1, 0])
ax_emb = fig.add_subplot(gs[1, 1])
ax_rag = fig.add_subplot(gs[2, 0])
ax_vdb = fig.add_subplot(gs[2, 1])

BLUE, ORANGE, GREEN, RED, PURPLE, DARK, GREY = (
    "#2E86C1", "#E67E22", "#27AE60", "#E74C3C", "#8E44AD", "#2C3E50", "#BDC3C7")

# ═══════════════════ PANEL 1 — CoT ═══════════════════
ax_cot.set_title("1 · CoT — Reasoning Tokens",
                 fontsize=12, fontweight="bold", color=DARK)
ax_cot.set_xlim(0, 10); ax_cot.set_ylim(0, 10); ax_cot.axis("off")
ax_cot.add_patch(FancyBboxPatch((0.3, 1), 9.4, 8,
                                boxstyle="round,pad=0.1",
                                facecolor="#F4ECF7", edgecolor=PURPLE, lw=2))
ax_cot.text(5, 8.3, "Q: 17*24 = ?", ha="center", fontsize=11,
            fontweight="bold", color=DARK)
ax_cot.text(5, 7.1, "Think step by step:", ha="center",
            fontsize=10, color=DARK, style="italic")
for i, line in enumerate([
    "17*20 = 340",
    "17*4 = 68",
    "340 + 68 = 408"]):
    ax_cot.text(5, 6.0 - i*0.9, line, ha="center", fontsize=10,
                color=DARK,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor=PURPLE, lw=1))
ax_cot.text(5, 2.5, "A: 408  check", ha="center", fontsize=12,
            color=GREEN, fontweight="bold")
ax_cot.text(5, 1.5, "Emerges ~100B params. Also: self-consistency, ToT.",
            ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════ PANEL 2 — ReAct loop ═══════════════════
ax_react.set_title("2 · ReAct — Thought / Act / Observe",
                   fontsize=12, fontweight="bold", color=DARK)
ax_react.set_xlim(0, 10); ax_react.set_ylim(0, 10); ax_react.axis("off")
nodes = [("Thought", 5, 8, BLUE),
         ("Action",  2, 3.5, ORANGE),
         ("Observation", 8, 3.5, GREEN)]
for name, x, y, c in nodes:
    ax_react.add_patch(FancyBboxPatch((x-1.5, y-0.8), 3, 1.6,
                                      boxstyle="round,pad=0.1",
                                      facecolor=c, edgecolor="white", lw=2))
    ax_react.text(x, y, name, ha="center", va="center",
                  color="white", fontweight="bold", fontsize=11)
for (x1,y1),(x2,y2) in [((5,7.2),(2.7,4.2)),
                         ((3.2,3.5),(6.8,3.5)),
                         ((7.3,4.2),(5,7.2))]:
    ax_react.annotate("", xy=(x2,y2), xytext=(x1,y1),
                      arrowprops=dict(arrowstyle="-|>", color=DARK, lw=2))
ax_react.text(5, 1.3,
              "Reasoning + Tool use interleaved.\n"
              "Loop until Final Answer.",
              ha="center", fontsize=9.5, color="#555", style="italic")

# ═══════════════════ PANEL 3 — LangChain vs SK ═══════════════════
ax_fw.set_title("3 · LangChain vs Semantic Kernel",
                fontsize=12, fontweight="bold", color=DARK)
ax_fw.set_xlim(0, 10); ax_fw.set_ylim(0, 10); ax_fw.axis("off")

rows = [
    (8.2, "Language",   "Python/TS",          ".NET / Python"),
    (7.0, "Abstraction","Chains / Agents",    "Kernel + Plugins"),
    (5.8, "Planning",   "Agent loop",         "Planner + Planner.Run"),
    (4.6, "Integration","many tools fast",    "enterprise DI-first"),
    (3.4, "Best for",   "prototypes, OSS",    "Microsoft stack, prod"),
]
ax_fw.text(1.5, 9.0, "dimension", fontweight="bold", fontsize=10, color=DARK)
ax_fw.text(5.0, 9.0, "LangChain", fontweight="bold", fontsize=10, color=PURPLE, ha="center")
ax_fw.text(8.0, 9.0, "Semantic K.", fontweight="bold", fontsize=10, color=BLUE, ha="center")
for y, dim, lc, sk in rows:
    ax_fw.text(1.5, y, dim, fontsize=10, color=DARK)
    ax_fw.text(5.0, y, lc, fontsize=10, color=DARK, ha="center")
    ax_fw.text(8.0, y, sk, fontsize=10, color=DARK, ha="center")
    ax_fw.axhline(y - 0.5, xmin=0.05, xmax=0.95, color=GREY, lw=0.5)
ax_fw.text(5, 1.5,
           "Both realise the ReAct pattern.\n"
           "Pick by ecosystem, not by capability.",
           ha="center", fontsize=9.5, color="#555", style="italic")

# ═══════════════════ PANEL 4 — Embeddings ═══════════════════
ax_emb.set_title("4 · Embeddings & Similarity",
                 fontsize=12, fontweight="bold", color=DARK)
# scatter clusters
np.random.seed(3)
for cx, cy, c, name in [(2.5, 7.5, BLUE,   "menu"),
                         (7.5, 7.5, ORANGE, "drinks"),
                         (5.0, 3.0, GREEN,  "allergen")]:
    pts = np.random.randn(10, 2) * 0.4 + [cx, cy]
    ax_emb.scatter(pts[:, 0], pts[:, 1], s=60, color=c,
                   edgecolor="white", linewidth=1, alpha=0.8)
    ax_emb.text(cx, cy + 0.9, name, ha="center",
                fontweight="bold", fontsize=10, color=c)
# distance arrows
ax_emb.annotate("", xy=(5.0, 3.6), xytext=(2.7, 7.3),
                arrowprops=dict(arrowstyle="<->", color=DARK, lw=1.5))
ax_emb.text(3.7, 5.4, "cos sim", fontsize=9, color=DARK,
            fontweight="bold", rotation=55,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor=GREY, lw=0.6))
ax_emb.set_xlim(0, 10); ax_emb.set_ylim(0, 10)
ax_emb.set_xticks([]); ax_emb.set_yticks([])
ax_emb.text(5, 0.5,
            "cosine (normalised) = dot product.  L2 after norm ~ 2(1-cos).",
            ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════ PANEL 5 — RAG pipeline ═══════════════════
ax_rag.set_title("5 · RAG Pipeline (query time)",
                 fontsize=12, fontweight="bold", color=DARK)
ax_rag.set_xlim(0, 12); ax_rag.set_ylim(0, 6); ax_rag.axis("off")

steps = [
    (0.3,  "Query",    BLUE,   "user"),
    (2.6,  "Embed",    PURPLE, "encoder"),
    (5.0,  "Retrieve", GREEN,  "ANN top-k"),
    (7.6,  "Rerank",   ORANGE, "cross-enc."),
    (10.2, "LLM",      DARK,   "generate"),
]
for i, (x, name, c, body) in enumerate(steps):
    ax_rag.add_patch(FancyBboxPatch((x, 2.2), 1.6, 2.2,
                                    boxstyle="round,pad=0.08",
                                    facecolor=c, edgecolor="white", lw=2))
    ax_rag.text(x + 0.8, 3.9, name, ha="center", va="center",
                color="white", fontweight="bold", fontsize=11)
    ax_rag.text(x + 0.8, 2.8, body, ha="center", va="center",
                color="white", fontsize=9)
    if i < len(steps) - 1:
        x_next = steps[i+1][0]
        ax_rag.annotate("", xy=(x_next - 0.05, 3.3),
                        xytext=(x + 1.65, 3.3),
                        arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.8))
ax_rag.text(6, 1.0,
            "Chunk (200-500 tok) + overlap. Hybrid search when keyword matters.\n"
            "Eval with RAGAS (faithfulness, relevancy, precision, recall).",
            ha="center", fontsize=9.5, color="#555", style="italic")

# ═══════════════════ PANEL 6 — Vector DB triangle ═══════════════════
ax_vdb.set_title("6 · Vector DB Tradeoff Triangle",
                 fontsize=12, fontweight="bold", color=DARK)
# triangle corners
pts = {"recall":   (5, 8.5),
       "latency":  (1.2, 2.5),
       "memory":   (8.8, 2.5)}
tri_x = [pts[k][0] for k in pts] + [pts["recall"][0]]
tri_y = [pts[k][1] for k in pts] + [pts["recall"][1]]
ax_vdb.plot(tri_x, tri_y, "-", color=DARK, lw=2)
for k, (x, y) in pts.items():
    ax_vdb.plot(x, y, "o", color=PURPLE, markersize=18,
                markeredgecolor="white", markeredgewidth=2)
    dy = 0.5 if y > 5 else -0.7
    ax_vdb.text(x, y + dy, k, ha="center", fontweight="bold",
                fontsize=11, color=DARK)

# Put indexes
idx_pts = [("HNSW",    5.0, 6.5, BLUE),
           ("IVF-PQ",  7.0, 3.6, ORANGE),
           ("Flat",    4.0, 4.8, RED),
           ("ScaNN",   5.6, 5.0, GREEN)]
for name, x, y, c in idx_pts:
    ax_vdb.plot(x, y, "s", color=c, markersize=14,
                markeredgecolor="white", markeredgewidth=1.5)
    ax_vdb.text(x + 0.2, y + 0.15, name, fontsize=9,
                color=c, fontweight="bold")
ax_vdb.set_xlim(0, 10); ax_vdb.set_ylim(0, 10)
ax_vdb.axis("off")
ax_vdb.text(5, 0.5,
            "Pick two corners — the third suffers.",
            ha="center", fontsize=9.5, color="#555", style="italic")

out = str(Path(__file__).resolve().parent.parent / "img" / "AI Interview Primer.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out}")
