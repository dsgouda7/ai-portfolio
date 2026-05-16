from pathlib import Path
"""Generate Shared Memory.png — 4-panel xkcd-style concept sketch.

Panels:
  (1) Blackboard architecture — agents read/write a shared store
  (2) Memory scopes: per-task / per-entity / per-user
  (3) In-memory vs external store (local vs distributed)
  (4) Long-term memory: episodic + semantic retrieval
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.xkcd(scale=0.8, length=100, randomness=2)

fig = plt.figure(figsize=(18, 11), facecolor="white")
fig.suptitle("Shared Memory & Blackboard Architectures",
             fontsize=22, fontweight="bold", y=0.98)

gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.28,
                      left=0.05, right=0.97, top=0.91, bottom=0.06)
ax_bb = fig.add_subplot(gs[0, 0])
ax_scope = fig.add_subplot(gs[0, 1])
ax_store = fig.add_subplot(gs[1, 0])
ax_long = fig.add_subplot(gs[1, 1])

BLUE, ORANGE, GREEN, RED, PURPLE, DARK, GREY = (
    "#2E86C1", "#E67E22", "#27AE60", "#E74C3C", "#8E44AD", "#2C3E50", "#BDC3C7")

# ═══════════════════════ PANEL 1 — Blackboard ═══════════════════════
ax_bb.set_title("1 · Blackboard Pattern",
                fontsize=13, fontweight="bold", color=DARK)
ax_bb.set_xlim(0, 10); ax_bb.set_ylim(0, 10); ax_bb.axis("off")

# Blackboard in the centre
ax_bb.add_patch(FancyBboxPatch((3.0, 3.5), 4.0, 3.0,
                               boxstyle="round,pad=0.1",
                               facecolor=DARK, edgecolor="white", lw=2))
ax_bb.text(5.0, 6.1, "Blackboard (shared state)",
           ha="center", color="white", fontweight="bold", fontsize=11)
keys = ["order: {...}", "allergens: [...]", "price: £24", "status: paid"]
for i, k in enumerate(keys):
    ax_bb.text(5.0, 5.5 - i*0.5, k, ha="center", va="center",
               color="white", fontsize=9, family="monospace")

# Agents around it
agents = [
    (1.5, 8.5, "Planner",   BLUE),
    (8.5, 8.5, "Menu",      ORANGE),
    (1.5, 1.5, "Pricing",   GREEN),
    (8.5, 1.5, "Notifier",  PURPLE),
]
for x, y, name, c in agents:
    ax_bb.add_patch(plt.Circle((x, y), 0.6, facecolor=c, lw=2,
                               edgecolor="white"))
    ax_bb.text(x, y, name, ha="center", va="center", color="white",
               fontweight="bold", fontsize=9)
    ax_bb.annotate("", xy=(5 + (x - 5) * 0.25, 5 + (y - 5) * 0.25),
                   xytext=(x + (5 - x) * 0.15, y + (5 - y) * 0.15),
                   arrowprops=dict(arrowstyle="<->", color=c, lw=1.3))

ax_bb.text(5, 0.3,
           "Agents don't need to know about each other — they observe the board.",
           ha="center", fontsize=9.5, color="#555", style="italic")

# ═══════════════════════ PANEL 2 — Memory scope ════════════════════
ax_scope.set_title("2 · Memory Scopes",
                   fontsize=13, fontweight="bold", color=DARK)
ax_scope.set_xlim(0, 10); ax_scope.set_ylim(0, 10); ax_scope.axis("off")

cards = [
    (8.0, "per-task",   BLUE,
     "scratchpad for one\nworkflow; TTL short"),
    (5.5, "per-entity", ORANGE,
     "per-order / per-cart /\nper-session; TTL med"),
    (3.0, "per-user",   PURPLE,
     "preferences, history;\npersistent"),
    (0.5, "global",     DARK,
     "menu, policy docs;\nread-mostly"),
]
for y, name, c, body in cards:
    ax_scope.add_patch(FancyBboxPatch((0.5, y), 9.0, 2.0,
                                      boxstyle="round,pad=0.05",
                                      facecolor=c, edgecolor="white", lw=2))
    ax_scope.text(2.2, y + 1.0, name, ha="center", va="center",
                  color="white", fontweight="bold", fontsize=11)
    ax_scope.text(6.5, y + 1.0, body, ha="center", va="center",
                  color="white", fontsize=9.5)

# ═══════════════════════ PANEL 3 — In-memory vs external store ═════
ax_store.set_title("3 · In-Memory vs External Store",
                   fontsize=13, fontweight="bold", color=DARK)
ax_store.set_xlim(0, 10); ax_store.set_ylim(0, 10); ax_store.axis("off")

ax_store.add_patch(FancyBboxPatch((0.3, 1.5), 4.4, 7.5,
                                  boxstyle="round,pad=0.1",
                                  facecolor="#FADBD8", edgecolor=RED, lw=2))
ax_store.text(2.5, 8.5, "In-process dict", ha="center",
              fontweight="bold", fontsize=11, color=RED)
ax_store.text(2.5, 7.4, "state = {}", ha="center", fontsize=10,
              color=DARK, family="monospace",
              bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                        edgecolor=RED, lw=0.8))
for i, b in enumerate([
    "works in notebook",
    "lost on restart",
    "not shared across\npods / processes",
    "no persistence",
    "breaks in production"]):
    ax_store.text(2.5, 6.3 - i*0.9, "- " + b, ha="center",
                  fontsize=9.5, color=DARK)

ax_store.add_patch(FancyBboxPatch((5.3, 1.5), 4.4, 7.5,
                                  boxstyle="round,pad=0.1",
                                  facecolor="#D5F5E3", edgecolor=GREEN, lw=2))
ax_store.text(7.5, 8.5, "External store", ha="center",
              fontweight="bold", fontsize=11, color=GREEN)
ax_store.text(7.5, 7.4, "Redis / DB / blob", ha="center", fontsize=10,
              color=DARK, family="monospace",
              bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                        edgecolor=GREEN, lw=0.8))
for i, b in enumerate([
    "survives restarts",
    "shared by all agents",
    "TTL + eviction",
    "replicated",
    "production-ready"]):
    ax_store.text(7.5, 6.3 - i*0.9, "+ " + b, ha="center",
                  fontsize=9.5, color=DARK)

ax_store.text(5, 0.5,
              "Prototype in-memory. Ship with an external store.",
              ha="center", fontsize=9.5, color="#555", style="italic")

# ═══════════════════════ PANEL 4 — Long-term memory ═════════════════
ax_long.set_title("4 · Long-Term Memory — Episodic + Semantic",
                  fontsize=13, fontweight="bold", color=DARK)
ax_long.set_xlim(0, 10); ax_long.set_ylim(0, 10); ax_long.axis("off")

# Write path
ax_long.add_patch(FancyBboxPatch((0.3, 6.5), 2.0, 1.4,
                                 boxstyle="round,pad=0.05",
                                 facecolor=BLUE, edgecolor="white", lw=2))
ax_long.text(1.3, 7.2, "interaction", ha="center", va="center",
             color="white", fontweight="bold", fontsize=10)
ax_long.annotate("", xy=(3.0, 7.2), xytext=(2.3, 7.2),
                 arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.5))
ax_long.add_patch(FancyBboxPatch((3.0, 6.5), 2.0, 1.4,
                                 boxstyle="round,pad=0.05",
                                 facecolor=PURPLE, edgecolor="white", lw=2))
ax_long.text(4.0, 7.2, "summarise +\nembed", ha="center", va="center",
             color="white", fontweight="bold", fontsize=9)
ax_long.annotate("", xy=(5.7, 7.2), xytext=(5.0, 7.2),
                 arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.5))
ax_long.add_patch(FancyBboxPatch((5.7, 6.5), 3.8, 1.4,
                                 boxstyle="round,pad=0.05",
                                 facecolor=GREEN, edgecolor="white", lw=2))
ax_long.text(7.6, 7.2, "vector store + kv metadata",
             ha="center", va="center",
             color="white", fontweight="bold", fontsize=10)

# Read path
ax_long.text(5, 5.3, "— at next interaction —",
             ha="center", fontsize=9.5, color="#555", style="italic")

ax_long.add_patch(FancyBboxPatch((0.3, 3.0), 2.0, 1.4,
                                 boxstyle="round,pad=0.05",
                                 facecolor=BLUE, edgecolor="white", lw=2))
ax_long.text(1.3, 3.7, "new query", ha="center", va="center",
             color="white", fontweight="bold", fontsize=10)
ax_long.annotate("", xy=(3.0, 3.7), xytext=(2.3, 3.7),
                 arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.5))
ax_long.add_patch(FancyBboxPatch((3.0, 3.0), 2.0, 1.4,
                                 boxstyle="round,pad=0.05",
                                 facecolor=ORANGE, edgecolor="white", lw=2))
ax_long.text(4.0, 3.7, "retrieve\nmemories", ha="center", va="center",
             color="white", fontweight="bold", fontsize=9)
ax_long.annotate("", xy=(5.7, 3.7), xytext=(5.0, 3.7),
                 arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.5))
ax_long.add_patch(FancyBboxPatch((5.7, 3.0), 3.8, 1.4,
                                 boxstyle="round,pad=0.05",
                                 facecolor=DARK, edgecolor="white", lw=2))
ax_long.text(7.6, 3.7, "inject into agent prompt",
             ha="center", va="center",
             color="white", fontweight="bold", fontsize=10)

ax_long.text(5, 1.2,
             "Episodic = \"what happened\". Semantic = \"distilled facts\".\n"
             "Summarise aggressively to keep context small.",
             ha="center", fontsize=9.5, color="#555", style="italic")

out = str(Path(__file__).resolve().parent.parent / "img" / "Shared Memory.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out}")
