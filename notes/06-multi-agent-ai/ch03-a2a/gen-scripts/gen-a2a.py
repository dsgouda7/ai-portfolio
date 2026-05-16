from pathlib import Path
"""Generate A2A.png — 4-panel xkcd-style concept sketch.

Panels:
  (1) Tool vs Agent — capability boundary
  (2) Agent Card structure
  (3) Task lifecycle state machine
  (4) MCP vs A2A layering
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.xkcd(scale=0.8, length=100, randomness=2)

fig = plt.figure(figsize=(18, 11), facecolor="white")
fig.suptitle("Agent-to-Agent Protocol (A2A)",
             fontsize=22, fontweight="bold", y=0.98)

gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.28,
                      left=0.05, right=0.97, top=0.91, bottom=0.06)
ax_vs = fig.add_subplot(gs[0, 0])
ax_card = fig.add_subplot(gs[0, 1])
ax_life = fig.add_subplot(gs[1, 0])
ax_stack = fig.add_subplot(gs[1, 1])

BLUE, ORANGE, GREEN, RED, PURPLE, DARK, GREY = (
    "#2E86C1", "#E67E22", "#27AE60", "#E74C3C", "#8E44AD", "#2C3E50", "#BDC3C7")

# ═══════════════════════ PANEL 1 — Tool vs Agent ════════════════════
ax_vs.set_title("1 · Tool vs Agent — Different Kinds of Callable",
                fontsize=13, fontweight="bold", color=DARK)
ax_vs.set_xlim(0, 10); ax_vs.set_ylim(0, 10); ax_vs.axis("off")

ax_vs.add_patch(FancyBboxPatch((0.3, 1.5), 4.4, 7.5,
                               boxstyle="round,pad=0.1",
                               facecolor="#EBF5FB", edgecolor=BLUE, lw=2))
ax_vs.text(2.5, 8.4, "Tool", ha="center", fontweight="bold",
           fontsize=12, color=BLUE)
ax_vs.text(2.5, 7.3, "deterministic\nfunction",
           ha="center", fontsize=10, color=DARK)
for i, b in enumerate([
    "pure I/O schema",
    "no internal LLM",
    "no multi-step plan",
    "short-lived call",
    "MCP native"]):
    ax_vs.text(2.5, 6.1 - i*0.7, "- " + b, ha="center",
               fontsize=9.5, color=DARK)

ax_vs.add_patch(FancyBboxPatch((5.3, 1.5), 4.4, 7.5,
                               boxstyle="round,pad=0.1",
                               facecolor="#F4ECF7", edgecolor=PURPLE, lw=2))
ax_vs.text(7.5, 8.4, "Agent", ha="center", fontweight="bold",
           fontsize=12, color=PURPLE)
ax_vs.text(7.5, 7.3, "autonomous\nworker",
           ha="center", fontsize=10, color=DARK)
for i, b in enumerate([
    "own LLM + memory",
    "multi-step plans",
    "calls other agents/tools",
    "long-lived tasks",
    "A2A native"]):
    ax_vs.text(7.5, 6.1 - i*0.7, "- " + b, ha="center",
               fontsize=9.5, color=DARK)

ax_vs.text(5, 0.6,
           "A2A exists because agents aren't just bigger tools.",
           ha="center", fontsize=9.5, color="#555", style="italic")

# ═══════════════════════ PANEL 2 — Agent Card ══════════════════════
ax_card.set_title("2 · Agent Card (public capability manifest)",
                  fontsize=13, fontweight="bold", color=DARK)
ax_card.set_xlim(0, 10); ax_card.set_ylim(0, 10); ax_card.axis("off")

ax_card.add_patch(FancyBboxPatch((0.5, 0.8), 9.0, 8.5,
                                 boxstyle="round,pad=0.1",
                                 facecolor="#FEF9E7", edgecolor=ORANGE, lw=2))
fields = [
    (8.5, "name",          "BillingAgent"),
    (7.5, "description",   "Prices orders & applies discounts"),
    (6.5, "skills",         "[price_order, apply_discount]"),
    (5.5, "input_schema",  "{ order: Order, coupon?: str }"),
    (4.5, "output_schema", "{ total: float, breakdown: [] }"),
    (3.5, "auth",          "bearer, mtls, oauth2"),
    (2.5, "streaming",     "true"),
    (1.5, "url / endpoint","https://billing.internal/a2a"),
]
for y, k, v in fields:
    ax_card.text(1.0, y, k, fontsize=10, fontweight="bold",
                 color=DARK, family="monospace")
    ax_card.text(3.5, y, ": " + v, fontsize=10, color=DARK,
                 family="monospace")

ax_card.text(5, 0.2,
             "Discoverable like an OpenAPI doc.",
             ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 3 — Task lifecycle ═══════════════════
ax_life.set_title("3 · Task Lifecycle",
                  fontsize=13, fontweight="bold", color=DARK)
ax_life.set_xlim(0, 10); ax_life.set_ylim(0, 10); ax_life.axis("off")

states = [
    (1.5, 8.0, "submitted",  BLUE),
    (4.5, 8.0, "working",    ORANGE),
    (7.5, 8.0, "completed",  GREEN),
    (4.5, 5.0, "input-required", PURPLE),
    (7.5, 5.0, "canceled",   GREY),
    (4.5, 2.0, "failed",     RED),
]
for x, y, name, c in states:
    ax_life.add_patch(FancyBboxPatch((x - 1.3, y - 0.5), 2.6, 1.0,
                                     boxstyle="round,pad=0.05",
                                     facecolor=c, edgecolor="white", lw=2))
    ax_life.text(x, y, name, ha="center", va="center",
                 color="white", fontweight="bold", fontsize=10)

edges = [
    ((1.5, 8.0), (4.5, 8.0)),
    ((4.5, 8.0), (7.5, 8.0)),
    ((4.5, 8.0), (4.5, 5.0)),
    ((4.5, 5.0), (4.5, 8.0)),
    ((4.5, 8.0), (7.5, 5.0)),
    ((4.5, 8.0), (4.5, 2.0)),
]
for (x1, y1), (x2, y2) in edges:
    ax_life.annotate("", xy=(x2, y2 + 0.5 if y2 < y1 else y2 - 0.5),
                     xytext=(x1, y1 - 0.5 if y1 > y2 else y1 + 0.5),
                     arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.5))

ax_life.text(5, 0.6,
             "Tasks are addressable & resumable — critical for long-running work.\n"
             "Clients poll or subscribe for status.",
             ha="center", fontsize=9.5, color="#555", style="italic")

# ═══════════════════════ PANEL 4 — MCP vs A2A stack ═════════════════
ax_stack.set_title("4 · MCP vs A2A — Complementary Layers",
                   fontsize=13, fontweight="bold", color=DARK)
ax_stack.set_xlim(0, 10); ax_stack.set_ylim(0, 10); ax_stack.axis("off")

tiers = [
    (8.0, "User / Host app",           GREY,   "chat UI, IDE"),
    (6.5, "Agent (via A2A)",           PURPLE, "BillingAgent / PlannerAgent"),
    (5.0, "Tools / Resources (MCP)",   ORANGE, "create_order, read_menu"),
    (3.5, "LLM + runtime",             BLUE,   "model weights, inference"),
    (2.0, "Infra",                     DARK,   "auth, queues, storage"),
]
for y, name, c, body in tiers:
    ax_stack.add_patch(FancyBboxPatch((0.6, y - 0.55), 8.8, 1.1,
                                      boxstyle="round,pad=0.05",
                                      facecolor=c, edgecolor="white", lw=2))
    ax_stack.text(2.3, y, name, ha="center", va="center",
                  color="white", fontweight="bold", fontsize=10)
    ax_stack.text(6.5, y, body, ha="center", va="center",
                  color="white", fontsize=9.5)

ax_stack.text(5, 0.6,
              "MCP = tools & data.  A2A = peer agents.  Use both.",
              ha="center", fontsize=9.5, color=DARK, fontweight="bold",
              style="italic")

out = str(Path(__file__).resolve().parent.parent / "img" / "A2A.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out}")
