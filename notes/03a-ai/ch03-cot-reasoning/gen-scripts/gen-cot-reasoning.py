from pathlib import Path
"""Generate CoT Reasoning.png — 4-panel xkcd-style concept sketch.

Panels:
  (1) Direct answer vs Chain-of-Thought
  (2) Thought -> Action -> Observation loop (ReAct-style agent)
  (3) Reasoning structures: Linear CoT, Self-Consistency, Tree-of-Thoughts
  (4) Reasoning budget vs accuracy curve
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.xkcd(scale=0.8, length=100, randomness=2)

fig = plt.figure(figsize=(18, 11), facecolor="white")
fig.suptitle("Chain-of-Thought Reasoning", fontsize=22, fontweight="bold", y=0.98)

gs = fig.add_gridspec(2, 2, hspace=0.50, wspace=0.28,
                      left=0.05, right=0.97, top=0.91, bottom=0.06)
ax_cmp = fig.add_subplot(gs[0, 0])
ax_loop = fig.add_subplot(gs[0, 1])
ax_struct = fig.add_subplot(gs[1, 0])
ax_budget = fig.add_subplot(gs[1, 1])

BLUE, ORANGE, GREEN, RED, PURPLE, DARK, GREY = (
    "#2E86C1", "#E67E22", "#27AE60", "#E74C3C", "#8E44AD", "#2C3E50", "#BDC3C7")

# ═══════════════════════ PANEL 1 — Direct vs CoT ════════════════════
ax_cmp.set_title("1 · Direct Answer vs Chain-of-Thought",
                 fontsize=13, fontweight="bold", color=DARK)
ax_cmp.set_xlim(0, 10); ax_cmp.set_ylim(0, 10); ax_cmp.axis("off")

ax_cmp.add_patch(FancyBboxPatch((0.3, 6.5), 4.4, 3,
                                boxstyle="round,pad=0.1",
                                facecolor="#FADBD8", edgecolor=RED, lw=2))
ax_cmp.text(2.5, 9.1, "Direct", ha="center", fontweight="bold",
            fontsize=12, color=RED)
ax_cmp.text(2.5, 8.0, 'Q: "If 3 pizzas feed 8,\nhow many for 20?"',
            ha="center", fontsize=9, color=DARK)
ax_cmp.text(2.5, 7.0, "A: 7 pizzas  X", ha="center",
            fontsize=11, color=RED, fontweight="bold")

ax_cmp.add_patch(FancyBboxPatch((5.3, 6.5), 4.4, 3,
                                boxstyle="round,pad=0.1",
                                facecolor="#D5F5E3", edgecolor=GREEN, lw=2))
ax_cmp.text(7.5, 9.1, "Chain-of-Thought", ha="center", fontweight="bold",
            fontsize=12, color=GREEN)
ax_cmp.text(7.5, 8.0, 'Think step by step.',
            ha="center", fontsize=9, color=DARK, style="italic")
ax_cmp.text(7.5, 7.2, "1) 3/8 pizza per person\n2) 20 * 3/8 = 7.5\n3) round up -> 8",
            ha="center", fontsize=8.5, color=DARK)
ax_cmp.text(7.5, 6.75, "A: 8  check", ha="center", fontsize=10,
            fontweight="bold", color=GREEN)

# Bottom: key insight
ax_cmp.text(5, 4.8, "Same model. Same prompt scaffold. Different output.",
            ha="center", fontsize=11, color=DARK, fontweight="bold")
ax_cmp.text(5, 3.7,
            "CoT emits *reasoning tokens* before the final answer.\n"
            "Each token conditions the next — the chain becomes the plan.",
            ha="center", fontsize=10, color="#555", style="italic")
ax_cmp.text(5, 1.8,
            "Emerges at ~100B parameters.\n"
            "Hidden in modern reasoning models (o1, o3, R1).",
            ha="center", fontsize=9, color=PURPLE,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F4ECF7",
                      edgecolor=PURPLE, lw=1))

# ═══════════════════════ PANEL 2 — Thought/Action/Observation loop ══
ax_loop.set_title("2 · Thought -> Action -> Observation Loop",
                  fontsize=13, fontweight="bold", color=DARK)
ax_loop.set_xlim(0, 10); ax_loop.set_ylim(0, 10); ax_loop.axis("off")

# Three boxes on a triangle
nodes = [
    ("Thought",     5.0, 8.0, BLUE),
    ("Action",      2.0, 3.5, ORANGE),
    ("Observation", 8.0, 3.5, GREEN),
]
for name, x, y, c in nodes:
    ax_loop.add_patch(FancyBboxPatch((x - 1.5, y - 0.8), 3.0, 1.6,
                                     boxstyle="round,pad=0.1",
                                     facecolor=c, edgecolor="white", lw=2))
    ax_loop.text(x, y, name, ha="center", va="center",
                 color="white", fontweight="bold", fontsize=12)

for (x1, y1), (x2, y2), lbl in [
        ((5.0, 7.2), (2.7, 4.2), "call tool"),
        ((3.2, 3.5), (6.8, 3.5), "tool result"),
        ((7.3, 4.2), (5.0, 7.2), "incorporate"),
]:
    ax_loop.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="-|>", color=DARK, lw=2))
    ax_loop.text((x1+x2)/2, (y1+y2)/2, lbl, ha="center", fontsize=9,
                 color=DARK, style="italic",
                 bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                           edgecolor=GREY, lw=0.6))

ax_loop.text(5, 0.9,
             "Repeat until Thought emits Final Answer.\n"
             "Each iteration grows the transcript the model conditions on.",
             ha="center", fontsize=10, color="#555", style="italic")

# ═══════════════════════ PANEL 3 — Reasoning structures ═════════════
ax_struct.set_title("3 · Reasoning Structures", fontsize=13,
                    fontweight="bold", color=DARK)
ax_struct.set_xlim(0, 12); ax_struct.set_ylim(0, 8); ax_struct.axis("off")

# Linear CoT
ax_struct.text(2, 7.3, "Linear CoT", ha="center", fontweight="bold",
               fontsize=11, color=BLUE)
for i, y in enumerate([6.0, 4.8, 3.6, 2.4]):
    ax_struct.add_patch(plt.Circle((2, y), 0.35, facecolor=BLUE,
                                   edgecolor="white", lw=2))
    if i < 3:
        ax_struct.annotate("", xy=(2, y - 0.8), xytext=(2, y - 0.4),
                           arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.5))

# Self-consistency (fan-out)
ax_struct.text(6, 7.3, "Self-Consistency", ha="center", fontweight="bold",
               fontsize=11, color=ORANGE)
ax_struct.add_patch(plt.Circle((6, 6.0), 0.35, facecolor=ORANGE,
                               edgecolor="white", lw=2))
for dx in (-1.6, -0.6, 0.6, 1.6):
    for y in (4.8, 3.6):
        ax_struct.add_patch(plt.Circle((6 + dx, y), 0.28, facecolor=ORANGE,
                                       edgecolor="white", lw=1.5, alpha=0.8))
    ax_struct.annotate("", xy=(6 + dx, 4.8 + 0.28),
                       xytext=(6, 5.7),
                       arrowprops=dict(arrowstyle="-", color=GREY, lw=1))
ax_struct.add_patch(plt.Circle((6, 2.2), 0.38, facecolor=GREEN,
                               edgecolor="white", lw=2))
ax_struct.text(6, 2.2, "vote", ha="center", va="center",
               color="white", fontsize=8, fontweight="bold")

# Tree-of-Thoughts
ax_struct.text(10, 7.3, "Tree-of-Thoughts", ha="center", fontweight="bold",
               fontsize=11, color=PURPLE)
tree = [(10, 6.0), (9, 4.8), (11, 4.8),
        (8.3, 3.6), (9.7, 3.6), (10.3, 3.6), (11.7, 3.6)]
for (x, y) in tree:
    ax_struct.add_patch(plt.Circle((x, y), 0.32, facecolor=PURPLE,
                                   edgecolor="white", lw=2))
edges = [((10,6.0),(9,4.8)),((10,6.0),(11,4.8)),
         ((9,4.8),(8.3,3.6)),((9,4.8),(9.7,3.6)),
         ((11,4.8),(10.3,3.6)),((11,4.8),(11.7,3.6))]
for (x1,y1),(x2,y2) in edges:
    ax_struct.plot([x1,x2],[y1,y2], "-", color=GREY, lw=1.2)
ax_struct.text(9.7, 2.3, "score + prune", ha="center", fontsize=9, color=DARK,
               style="italic")

ax_struct.text(6, 0.9,
               "More exploration -> higher accuracy, higher cost.\n"
               "Pick structure to match task complexity.",
               ha="center", fontsize=10, color="#555", style="italic")

# ═══════════════════════ PANEL 4 — Budget vs accuracy ═══════════════
ax_budget.set_title("4 · Reasoning Budget vs Accuracy",
                    fontsize=13, fontweight="bold", color=DARK)
tokens = np.linspace(50, 5000, 100)
acc = 0.55 + 0.4 * (1 - np.exp(-tokens / 1200))
ax_budget.plot(tokens, acc, "-", color=BLUE, lw=3)
ax_budget.fill_between(tokens, acc, 0.5, alpha=0.15, color=BLUE)

marks = [(150, "direct"), (800, "CoT"), (2500, "self-consistency"),
         (4500, "ToT / o1")]
for tk, lbl in marks:
    a = 0.55 + 0.4 * (1 - np.exp(-tk / 1200))
    ax_budget.plot(tk, a, "o", color=ORANGE, markersize=10,
                   markeredgecolor="white", markeredgewidth=2, zorder=5)
    ax_budget.annotate(lbl, (tk, a), textcoords="offset points",
                       xytext=(5, -15), fontsize=9, color=DARK,
                       fontweight="bold")
ax_budget.set_xlabel("reasoning tokens / call", fontsize=10, color=DARK)
ax_budget.set_ylabel("accuracy", fontsize=10, color=DARK)
ax_budget.set_xlim(0, 5200); ax_budget.set_ylim(0.5, 1.0)
ax_budget.text(0.5, -0.28,
               "Diminishing returns — calibrate budget to task.\n"
               "Most tasks do NOT need o1-level spend.",
               transform=ax_budget.transAxes, ha="center",
               fontsize=9.5, color="#555", style="italic")

out = str(Path(__file__).resolve().parent.parent / "img" / "CoT Reasoning.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out}")
