from pathlib import Path
"""Generate ReAct and Semantic Kernel.png — 4-panel xkcd-style concept sketch.

Panels:
  (1) ReAct loop with concrete Thought/Action/Observation example
  (2) LangChain vs Semantic Kernel architecture side-by-side
  (3) Planning mode vs execution mode
  (4) Multi-agent supervisor pattern
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.xkcd(scale=0.8, length=100, randomness=2)

fig = plt.figure(figsize=(18, 11), facecolor="white")
fig.suptitle("ReAct, LangChain & Semantic Kernel", fontsize=22,
             fontweight="bold", y=0.98)

gs = fig.add_gridspec(2, 2, hspace=0.50, wspace=0.28,
                      left=0.05, right=0.97, top=0.91, bottom=0.06)
ax_react = fig.add_subplot(gs[0, 0])
ax_fw = fig.add_subplot(gs[0, 1])
ax_mode = fig.add_subplot(gs[1, 0])
ax_multi = fig.add_subplot(gs[1, 1])

BLUE, ORANGE, GREEN, RED, PURPLE, DARK, GREY = (
    "#2E86C1", "#E67E22", "#27AE60", "#E74C3C", "#8E44AD", "#2C3E50", "#BDC3C7")

# ═══════════════════════ PANEL 1 — ReAct loop ═══════════════════════
ax_react.set_title("1 · ReAct Loop — Reason / Act / Observe",
                   fontsize=13, fontweight="bold", color=DARK)
ax_react.set_xlim(0, 10); ax_react.set_ylim(0, 10); ax_react.axis("off")

rows = [
    (8.7, "User",         'Gluten-free pizzas under £15?',  BLUE),
    (7.4, "Thought",      'Need menu + filter by allergen + price.', PURPLE),
    (6.1, "Action",       'search_menu(filter="gluten-free")',       ORANGE),
    (4.8, "Observation",  '[Margherita £13, Veggie £14, Quattro £18]', GREEN),
    (3.5, "Thought",      'Filter price<=15 -> two items.',           PURPLE),
    (2.2, "Action",       'Final Answer',                              ORANGE),
    (0.9, "Assistant",    'Margherita (£13), Veggie (£14)',           BLUE),
]
for y, role, txt, c in rows:
    ax_react.add_patch(FancyBboxPatch((0.3, y - 0.45), 1.8, 0.9,
                                      boxstyle="round,pad=0.05",
                                      facecolor=c, edgecolor="white", lw=1.5))
    ax_react.text(1.2, y, role, ha="center", va="center",
                  color="white", fontweight="bold", fontsize=9)
    ax_react.text(2.3, y, txt, va="center", fontsize=9, color=DARK)

# right-side arrow loop
ax_react.annotate("", xy=(9.5, 3.5), xytext=(9.5, 7.4),
                  arrowprops=dict(arrowstyle="-|>", color=DARK, lw=2,
                                  connectionstyle="arc3,rad=-0.3"))
ax_react.text(9.8, 5.4, "repeat", fontsize=10, color=DARK,
              fontweight="bold", rotation=-90)

# ═══════════════════════ PANEL 2 — Framework comparison ═════════════
ax_fw.set_title("2 · LangChain vs Semantic Kernel",
                fontsize=13, fontweight="bold", color=DARK)
ax_fw.set_xlim(0, 10); ax_fw.set_ylim(0, 10); ax_fw.axis("off")

# LangChain stack (left)
lc_blocks = [("Chains / LCEL", PURPLE),
             ("Agents", ORANGE),
             ("Tools", GREEN),
             ("Memory", BLUE),
             ("LLM wrapper", DARK)]
ax_fw.text(2.5, 9.2, "LangChain (Python/TS)", ha="center",
           fontweight="bold", fontsize=11, color=PURPLE)
for i, (name, c) in enumerate(lc_blocks):
    y = 8.0 - i * 1.25
    ax_fw.add_patch(FancyBboxPatch((0.8, y - 0.5), 3.4, 1.0,
                                   boxstyle="round,pad=0.05",
                                   facecolor=c, edgecolor="white", lw=1.5))
    ax_fw.text(2.5, y, name, ha="center", va="center",
               color="white", fontweight="bold", fontsize=10)

# SK stack (right)
sk_blocks = [("Planners", PURPLE),
             ("Kernel", ORANGE),
             ("Plugins (Skills)", GREEN),
             ("Memory / Connectors", BLUE),
             ("LLM connector", DARK)]
ax_fw.text(7.5, 9.2, "Semantic Kernel (.NET/Py)", ha="center",
           fontweight="bold", fontsize=11, color=BLUE)
for i, (name, c) in enumerate(sk_blocks):
    y = 8.0 - i * 1.25
    ax_fw.add_patch(FancyBboxPatch((5.8, y - 0.5), 3.4, 1.0,
                                   boxstyle="round,pad=0.05",
                                   facecolor=c, edgecolor="white", lw=1.5))
    ax_fw.text(7.5, y, name, ha="center", va="center",
               color="white", fontweight="bold", fontsize=10)

ax_fw.text(5, 0.6,
           "LangChain: open-source, pythonic, fast-moving.\n"
           "SK: enterprise .NET, DI-friendly, plugins = typed functions.",
           ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 3 — Planning vs Execution ════════════
ax_mode.set_title("3 · Planning vs Execution Modes",
                  fontsize=13, fontweight="bold", color=DARK)
ax_mode.set_xlim(0, 10); ax_mode.set_ylim(0, 10); ax_mode.axis("off")

# Left: planning
ax_mode.add_patch(FancyBboxPatch((0.3, 1.5), 4.4, 7,
                                 boxstyle="round,pad=0.1",
                                 facecolor="#EBF5FB", edgecolor=BLUE, lw=2))
ax_mode.text(2.5, 8.0, "Planning (plan-and-execute)",
             ha="center", fontweight="bold", fontsize=11, color=BLUE)
ax_mode.text(2.5, 7.2, "LLM emits full plan first.",
             ha="center", fontsize=9, color=DARK, style="italic")
for i, step in enumerate(["1. get_menu", "2. filter_allergens",
                          "3. compute_price", "4. respond"]):
    ax_mode.text(2.5, 6.3 - i * 0.9, step, ha="center",
                 fontsize=10, color=DARK,
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                           edgecolor=BLUE, lw=1))
ax_mode.text(2.5, 2.0,
             "+ cacheable + auditable\n- stale if env changes",
             ha="center", fontsize=9, color="#555")

# Right: ReAct execution
ax_mode.add_patch(FancyBboxPatch((5.3, 1.5), 4.4, 7,
                                 boxstyle="round,pad=0.1",
                                 facecolor="#FEF9E7", edgecolor=ORANGE, lw=2))
ax_mode.text(7.5, 8.0, "Execution (ReAct)",
             ha="center", fontweight="bold", fontsize=11, color=ORANGE)
ax_mode.text(7.5, 7.2, "Next step chosen after each obs.",
             ha="center", fontsize=9, color=DARK, style="italic")
for i, step in enumerate(["Thought_1", "Action_1",
                          "Obs_1", "Thought_2", "..."]):
    ax_mode.text(7.5, 6.3 - i * 0.8, step, ha="center",
                 fontsize=10, color=DARK,
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                           edgecolor=ORANGE, lw=1))
ax_mode.text(7.5, 2.0,
             "+ adaptive + robust\n- more tokens, higher latency",
             ha="center", fontsize=9, color="#555")

# ═══════════════════════ PANEL 4 — Multi-agent supervisor ═══════════
ax_multi.set_title("4 · Multi-Agent Supervisor Pattern",
                   fontsize=13, fontweight="bold", color=DARK)
ax_multi.set_xlim(0, 10); ax_multi.set_ylim(0, 10); ax_multi.axis("off")

# Supervisor at top
ax_multi.add_patch(FancyBboxPatch((3.8, 7.5), 2.4, 1.6,
                                  boxstyle="round,pad=0.1",
                                  facecolor=PURPLE, edgecolor="white", lw=2))
ax_multi.text(5, 8.3, "Supervisor\n(router)", ha="center", va="center",
              color="white", fontweight="bold", fontsize=11)

# Workers
workers = [
    (1.2, 4.0, "Menu Agent",     BLUE,   "search_menu\nlookup_prices"),
    (4.2, 4.0, "Allergen Agent", GREEN,  "check_allergens\nsafety rules"),
    (7.2, 4.0, "Delivery Agent", ORANGE, "eta\nassign_rider"),
]
for x, y, name, c, tools in workers:
    ax_multi.add_patch(FancyBboxPatch((x, y - 0.6), 2.2, 1.4,
                                      boxstyle="round,pad=0.08",
                                      facecolor=c, edgecolor="white", lw=2))
    ax_multi.text(x + 1.1, y + 0.15, name, ha="center", va="center",
                  color="white", fontweight="bold", fontsize=10)
    ax_multi.text(x + 1.1, y - 0.35, tools, ha="center", va="center",
                  color="white", fontsize=8)
    # arrow from supervisor
    ax_multi.annotate("", xy=(x + 1.1, y + 0.8), xytext=(5, 7.4),
                      arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.5))
    # arrow back
    ax_multi.annotate("", xy=(5, 7.5), xytext=(x + 1.1 + 0.2, y + 0.8),
                      arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1,
                                      linestyle=":"))

ax_multi.text(5, 2.0,
              "LangGraph / SK GroupChat / AutoGen all realise this pattern.\n"
              "Supervisor routes; workers specialise; shared state = graph edges.",
              ha="center", fontsize=9.5, color="#555", style="italic")

out = str(Path(__file__).resolve().parent.parent / "img" / "ReAct and Semantic Kernel.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out}")
