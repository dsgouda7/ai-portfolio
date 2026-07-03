from pathlib import Path
"""Generate Vector DBs.png — 4-panel xkcd-style concept sketch.

Panels:
  (1) Exact vs ANN search — tradeoff
  (2) HNSW layered proximity graph
  (3) IVF clustering with probed cells
  (4) Recall vs latency frontier
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.xkcd(scale=0.8, length=100, randomness=2)

fig = plt.figure(figsize=(18, 11), facecolor="white")
fig.suptitle("Vector Databases — ANN Indexing", fontsize=22,
             fontweight="bold", y=0.98)

gs = fig.add_gridspec(2, 2, hspace=0.50, wspace=0.28,
                      left=0.05, right=0.97, top=0.91, bottom=0.06)
ax_cmp = fig.add_subplot(gs[0, 0])
ax_hnsw = fig.add_subplot(gs[0, 1])
ax_ivf = fig.add_subplot(gs[1, 0])
ax_fr = fig.add_subplot(gs[1, 1])

BLUE, ORANGE, GREEN, RED, PURPLE, DARK, GREY = (
    "#2E86C1", "#E67E22", "#27AE60", "#E74C3C", "#8E44AD", "#2C3E50", "#BDC3C7")

# ═══════════════════════ PANEL 1 — Exact vs ANN ═════════════════════
ax_cmp.set_title("1 · Exact vs Approximate Nearest Neighbour",
                 fontsize=13, fontweight="bold", color=DARK)
ax_cmp.set_xlim(0, 10); ax_cmp.set_ylim(0, 10); ax_cmp.axis("off")

ax_cmp.add_patch(FancyBboxPatch((0.3, 4.2), 4.4, 4.5,
                                boxstyle="round,pad=0.1",
                                facecolor="#FADBD8", edgecolor=RED, lw=2))
ax_cmp.text(2.5, 8.2, "Exact (brute force)", ha="center",
            fontweight="bold", fontsize=12, color=RED)
ax_cmp.text(2.5, 7.1, "O(N * d) per query", ha="center",
            fontsize=11, color=DARK)
ax_cmp.text(2.5, 6.0, "100% recall", ha="center", fontsize=10, color=GREEN)
ax_cmp.text(2.5, 5.0,
            "Fine for N < 100k.\nBreaks above ~1M.",
            ha="center", fontsize=9, color=DARK)

ax_cmp.add_patch(FancyBboxPatch((5.3, 4.2), 4.4, 4.5,
                                boxstyle="round,pad=0.1",
                                facecolor="#D5F5E3", edgecolor=GREEN, lw=2))
ax_cmp.text(7.5, 8.2, "ANN (index + search)", ha="center",
            fontweight="bold", fontsize=12, color=GREEN)
ax_cmp.text(7.5, 7.1, "O(log N) or sub-linear", ha="center",
            fontsize=11, color=DARK)
ax_cmp.text(7.5, 6.0, "90-99% recall", ha="center", fontsize=10, color=ORANGE)
ax_cmp.text(7.5, 5.0,
            "Trade a little recall for\n1000x speedup.",
            ha="center", fontsize=9, color=DARK)

ax_cmp.text(5, 2.8, "Tradeoff axes:", ha="center", fontsize=11,
            fontweight="bold", color=DARK)
ax_cmp.text(5, 1.6,
            "recall  <->  latency  <->  memory  <->  build time",
            ha="center", fontsize=11, color=PURPLE, fontweight="bold")
ax_cmp.text(5, 0.5,
            "Pick 3 — every index family is a point on this surface.",
            ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 2 — HNSW ═════════════════════════════
ax_hnsw.set_title("2 · HNSW — Hierarchical Navigable Small World",
                  fontsize=13, fontweight="bold", color=DARK)
ax_hnsw.set_xlim(0, 10); ax_hnsw.set_ylim(0, 10); ax_hnsw.axis("off")

layers = [
    (7.5, BLUE,   [(5, 7.5)]),
    (5.2, ORANGE, [(2.5, 5.2), (5, 5.2), (7.5, 5.2)]),
    (2.5, PURPLE, [(1, 2.5), (2.8, 2.5), (4.2, 2.5),
                   (5.6, 2.5), (7.0, 2.5), (8.5, 2.5)]),
]
for ly, color, pts in layers:
    ax_hnsw.axhline(ly - 0.55, color=GREY, lw=0.6, alpha=0.4)
    for i, (x, y) in enumerate(pts):
        ax_hnsw.plot(x, y, "o", color=color, markersize=18,
                     markeredgecolor="white", markeredgewidth=2)
    for i in range(len(pts) - 1):
        ax_hnsw.plot([pts[i][0], pts[i+1][0]],
                     [pts[i][1], pts[i+1][1]],
                     "-", color=color, lw=1.2, alpha=0.7)

# cross-layer links (entry points)
ax_hnsw.plot([5, 5], [7.5, 5.2], ":", color=DARK, lw=1.2)
ax_hnsw.plot([5, 5.6], [5.2, 2.5], ":", color=DARK, lw=1.2)

ax_hnsw.text(0.2, 7.5, "L2 (sparse)", fontsize=9, color=BLUE,
             fontweight="bold", va="center")
ax_hnsw.text(0.2, 5.2, "L1", fontsize=9, color=ORANGE,
             fontweight="bold", va="center")
ax_hnsw.text(0.2, 2.5, "L0 (all points)", fontsize=9, color=PURPLE,
             fontweight="bold", va="center")

# search path
ax_hnsw.annotate("query", xy=(9.2, 8.3),
                 fontsize=10, color=RED, fontweight="bold")
ax_hnsw.plot([9.2, 5], [8.0, 7.5], "-", color=RED, lw=1.5)
ax_hnsw.plot([5, 5], [7.5, 5.2], "-", color=RED, lw=1.5)
ax_hnsw.plot([5, 5.6], [5.2, 2.5], "-", color=RED, lw=1.5)
ax_hnsw.plot([5.6, 7.0], [2.5, 2.5], "-", color=RED, lw=1.5)

ax_hnsw.text(5, 0.8,
             "Greedy descent: top layer narrows region, bottom layer refines.\n"
             "Params: M (neighbours), ef_construction, ef_search.",
             ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 3 — IVF ══════════════════════════════
ax_ivf.set_title("3 · IVF — Inverted File Index (cluster + probe)",
                 fontsize=13, fontweight="bold", color=DARK)
np.random.seed(1)
centroids = np.array([[2, 2], [2, 7], [5, 4.5], [7.5, 2], [7.5, 7]])
colors = [BLUE, ORANGE, GREEN, PURPLE, RED]
for i, (cx, cy) in enumerate(centroids):
    pts = np.random.randn(15, 2) * 0.6 + [cx, cy]
    ax_ivf.scatter(pts[:, 0], pts[:, 1], s=50, color=colors[i], alpha=0.7,
                   edgecolors="white", linewidth=0.8)
    ax_ivf.plot(cx, cy, "X", color=colors[i], markersize=18,
                markeredgecolor="white", markeredgewidth=2)
# Query and probed clusters
qx, qy = 5.5, 4.8
ax_ivf.plot(qx, qy, "*", color=DARK, markersize=22, zorder=6)
ax_ivf.text(qx + 0.25, qy + 0.35, "query", fontsize=10, color=DARK,
            fontweight="bold")
# nprobe=2 -> highlight green + red clusters
for i in (2, 4):
    cx, cy = centroids[i]
    ax_ivf.add_patch(plt.Circle((cx, cy), 1.6, facecolor=colors[i],
                                alpha=0.15, edgecolor=colors[i],
                                lw=2, linestyle="--"))
ax_ivf.set_xlim(0, 10); ax_ivf.set_ylim(0, 10)
ax_ivf.set_xticks([]); ax_ivf.set_yticks([])
ax_ivf.text(5, 0.3,
            "k-means clusters. Probe top nprobe cells.\n"
            "Higher nprobe -> better recall, more cost.",
            ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 4 — Recall vs latency ════════════════
ax_fr.set_title("4 · Recall vs Latency Frontier", fontsize=13,
                fontweight="bold", color=DARK)
# Pareto-ish points
families = [
    ("Flat (exact)", 1.00, 150, RED),
    ("IVF-Flat",     0.96,  25, BLUE),
    ("IVF-PQ",       0.88,   8, PURPLE),
    ("HNSW",         0.98,  12, GREEN),
    ("ScaNN",        0.97,  10, ORANGE),
]
for name, r, lat, c in families:
    ax_fr.plot(lat, r, "o", color=c, markersize=18,
               markeredgecolor="white", markeredgewidth=2)
    ax_fr.annotate(name, (lat, r), textcoords="offset points",
                   xytext=(8, 8), fontsize=9.5, color=c,
                   fontweight="bold")
# frontier
xs = np.array([7, 10, 25, 150])
ys = np.array([0.88, 0.97, 0.96, 1.00])
order = np.argsort(xs)
ax_fr.plot(xs[order], ys[order], "--", color=GREY, lw=1.5, alpha=0.7)

ax_fr.set_xscale("log")
ax_fr.set_xlabel("latency (ms, p95) — log scale", fontsize=10, color=DARK)
ax_fr.set_ylabel("recall@10", fontsize=10, color=DARK)
ax_fr.set_ylim(0.82, 1.02)
ax_fr.text(0.5, -0.28,
           "Modern vector DBs (Pinecone, Weaviate, Milvus, pgvector)\n"
           "bundle these indexes behind a single API.",
           transform=ax_fr.transAxes, ha="center",
           fontsize=9, color="#555", style="italic")

out = str(Path(__file__).resolve().parent.parent / "img" / "Vector DBs.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out}")
