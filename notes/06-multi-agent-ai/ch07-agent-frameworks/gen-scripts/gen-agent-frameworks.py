from pathlib import Path
"""Generate Agent Frameworks.png — 4-panel xkcd-style concept sketch.

Panels:
  (1) AutoGen — conversation-first group chat
  (2) LangGraph — explicit graph with state
  (3) Semantic Kernel AgentGroupChat — enterprise planner + plugins
  (4) Framework comparison matrix
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.xkcd(scale=0.8, length=100, randomness=2)

fig = plt.figure(figsize=(18, 11), facecolor="white")
fig.suptitle("Agent Frameworks — AutoGen / LangGraph / Semantic Kernel",
             fontsize=20, fontweight="bold", y=0.98)

gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.28,
                      left=0.05, right=0.97, top=0.91, bottom=0.06)
ax_ag = fig.add_subplot(gs[0, 0])
ax_lg = fig.add_subplot(gs[0, 1])
ax_sk = fig.add_subplot(gs[1, 0])
ax_cmp = fig.add_subplot(gs[1, 1])

BLUE, ORANGE, GREEN, RED, PURPLE, DARK, GREY = (
    "#2E86C1", "#E67E22", "#27AE60", "#E74C3C", "#8E44AD", "#2C3E50", "#BDC3C7")

# ═══════════════════════ PANEL 1 — AutoGen ══════════════════════════
ax_ag.set_title("1 · AutoGen — Conversation-First",
                fontsize=13, fontweight="bold", color=DARK)
ax_ag.set_xlim(0, 10); ax_ag.set_ylim(0, 10); ax_ag.axis("off")

# Group chat manager in the middle
ax_ag.add_patch(FancyBboxPatch((3.8, 4.2), 2.4, 1.6,
                               boxstyle="round,pad=0.1",
                               facecolor=PURPLE, edgecolor="white", lw=2))
ax_ag.text(5, 5.0, "GroupChat\nManager", ha="center", va="center",
           color="white", fontweight="bold", fontsize=10)

peers = [
    (1.5, 8.2, "Planner", BLUE),
    (8.5, 8.2, "Coder",   GREEN),
    (1.5, 1.5, "Critic",  ORANGE),
    (8.5, 1.5, "User",    RED),
]
for x, y, name, c in peers:
    ax_ag.add_patch(plt.Circle((x, y), 0.6, facecolor=c, lw=2,
                               edgecolor="white"))
    ax_ag.text(x, y, name, ha="center", va="center",
               color="white", fontweight="bold", fontsize=9)
    ax_ag.annotate("", xy=(5 + (x - 5) * 0.25, 5 + (y - 5) * 0.25),
                   xytext=(x + (5 - x) * 0.15, y + (5 - y) * 0.15),
                   arrowprops=dict(arrowstyle="<->", color=c, lw=1.3))

ax_ag.text(5, 0.3,
           "Emergent order. Manager picks next speaker via LLM.\n"
           "+ flexible  - harder to predict / test",
           ha="center", fontsize=9.5, color="#555", style="italic")

# ═══════════════════════ PANEL 2 — LangGraph ═══════════════════════
ax_lg.set_title("2 · LangGraph — Graph-First",
                fontsize=13, fontweight="bold", color=DARK)
ax_lg.set_xlim(0, 10); ax_lg.set_ylim(0, 10); ax_lg.axis("off")

nodes = [
    (5, 9.0, "START", DARK),
    (2.5, 7.0, "classify", BLUE),
    (7.5, 7.0, "retrieve", ORANGE),
    (5, 5.0, "generate", GREEN),
    (5, 3.0, "validate", PURPLE),
    (5, 1.0, "END", DARK),
]
for x, y, name, c in nodes:
    ax_lg.add_patch(FancyBboxPatch((x - 1.1, y - 0.4), 2.2, 0.8,
                                   boxstyle="round,pad=0.05",
                                   facecolor=c, edgecolor="white", lw=2))
    ax_lg.text(x, y, name, ha="center", va="center",
               color="white", fontweight="bold", fontsize=9)
edges = [
    (0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5)
]
for a, b in edges:
    xa, ya = nodes[a][0], nodes[a][1] - 0.4
    xb, yb = nodes[b][0], nodes[b][1] + 0.4
    ax_lg.annotate("", xy=(xb, yb), xytext=(xa, ya),
                   arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.3))
# retry edge
ax_lg.annotate("", xy=(5 - 1.15, 5), xytext=(5 - 1.15, 3),
               arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.3,
                               connectionstyle="arc3,rad=-0.4"))
ax_lg.text(2.9, 4.0, "retry", fontsize=8.5, color=RED,
           fontweight="bold")

ax_lg.text(5, 0.2,
           "Explicit state graph. Reproducible, testable, resumable.",
           ha="center", fontsize=9.5, color="#555", style="italic")

# ═══════════════════════ PANEL 3 — Semantic Kernel ═════════════════
ax_sk.set_title("3 · Semantic Kernel — Enterprise Orchestration",
                fontsize=13, fontweight="bold", color=DARK)
ax_sk.set_xlim(0, 10); ax_sk.set_ylim(0, 10); ax_sk.axis("off")

# Kernel at top
ax_sk.add_patch(FancyBboxPatch((3.4, 7.8), 3.2, 1.4,
                               boxstyle="round,pad=0.05",
                               facecolor=BLUE, edgecolor="white", lw=2))
ax_sk.text(5, 8.5, "Kernel + Planner", ha="center", va="center",
           color="white", fontweight="bold", fontsize=11)

# Plugins
plugins = [(1.5, 5.0, "Menu\nPlugin",    ORANGE),
           (4.0, 5.0, "Billing\nPlugin", GREEN),
           (6.5, 5.0, "Notify\nPlugin",  PURPLE),
           (9.0, 5.0, "Memory\nPlugin",  DARK)]
for x, y, name, c in plugins:
    ax_sk.add_patch(FancyBboxPatch((x - 1.0, y - 0.9), 2.0, 1.8,
                                   boxstyle="round,pad=0.05",
                                   facecolor=c, edgecolor="white", lw=2))
    ax_sk.text(x, y, name, ha="center", va="center",
               color="white", fontweight="bold", fontsize=9)
    ax_sk.annotate("", xy=(x, y + 0.95), xytext=(5, 7.75),
                   arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.3))

# Connectors
ax_sk.add_patch(FancyBboxPatch((0.5, 1.5), 9.0, 1.4,
                               boxstyle="round,pad=0.05",
                               facecolor=GREY, edgecolor="white", lw=2))
ax_sk.text(5, 2.2, "AI / Memory Connectors  (Azure OpenAI, Qdrant, Redis, ...)",
           ha="center", va="center", color="white",
           fontweight="bold", fontsize=10)

ax_sk.text(5, 0.3,
           "Plugins = typed functions. Planner composes them at runtime.\n"
           "Best for .NET / enterprise integration scenarios.",
           ha="center", fontsize=9.5, color="#555", style="italic")

# ═══════════════════════ PANEL 4 — Comparison ══════════════════════
ax_cmp.set_title("4 · Framework Comparison",
                 fontsize=13, fontweight="bold", color=DARK)
ax_cmp.set_xlim(0, 10); ax_cmp.set_ylim(0, 10); ax_cmp.axis("off")

headers = ["dimension", "AutoGen", "LangGraph", "Semantic K."]
rows = [
    ("control",        "emergent",     "explicit",     "planner"),
    ("language",       "Python",       "Py / TS",      ".NET / Py"),
    ("state",          "chat history", "typed state",  "kernel ctx"),
    ("best for",       "brainstorm",   "prod pipelines","enterprise"),
    ("testability",    "harder",       "first-class",  "good"),
    ("observability",  "chat log",     "graph trace",  "telemetry"),
]

col_x = [0.4, 3.0, 5.6, 8.2]
ax_cmp.text(col_x[0], 9.2, headers[0], fontweight="bold",
            fontsize=10, color=DARK)
for i, h in enumerate(headers[1:], 1):
    ax_cmp.text(col_x[i], 9.2, h, fontweight="bold", fontsize=10,
                color=[ORANGE, BLUE, GREEN][i - 1], ha="left")
ax_cmp.axhline(8.9, xmin=0.03, xmax=0.97, color=DARK, lw=1)

for j, r in enumerate(rows):
    y = 8.0 - j * 1.1
    for i, cell in enumerate(r):
        ax_cmp.text(col_x[i], y, cell, fontsize=9.5, color=DARK)
    ax_cmp.axhline(y - 0.5, xmin=0.03, xmax=0.97, color=GREY,
                   lw=0.4, alpha=0.6)

ax_cmp.text(5, 0.5,
            "Pick by team language + how much control you need over the plan.",
            ha="center", fontsize=9.5, color="#555", style="italic")

out = str(Path(__file__).resolve().parent.parent / "img" / "Agent Frameworks.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out}")
