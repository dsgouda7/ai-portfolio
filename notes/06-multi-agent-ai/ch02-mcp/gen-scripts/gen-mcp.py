from pathlib import Path
"""Generate MCP.png — 4-panel xkcd-style concept sketch.

Panels:
  (1) N x M integration problem -> N + M via MCP
  (2) Client/Server handshake (JSON-RPC)
  (3) Three primitives: Resources, Tools, Prompts
  (4) Transports: stdio / SSE / streamable HTTP
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.xkcd(scale=0.8, length=100, randomness=2)

fig = plt.figure(figsize=(18, 11), facecolor="white")
fig.suptitle("Model Context Protocol (MCP)",
             fontsize=22, fontweight="bold", y=0.98)

gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.28,
                      left=0.05, right=0.97, top=0.91, bottom=0.06)
ax_nm = fig.add_subplot(gs[0, 0])
ax_hs = fig.add_subplot(gs[0, 1])
ax_prim = fig.add_subplot(gs[1, 0])
ax_tr = fig.add_subplot(gs[1, 1])

BLUE, ORANGE, GREEN, RED, PURPLE, DARK, GREY = (
    "#2E86C1", "#E67E22", "#27AE60", "#E74C3C", "#8E44AD", "#2C3E50", "#BDC3C7")

# ═══════════════════════ PANEL 1 — N x M -> N + M ═══════════════════
ax_nm.set_title("1 · N x M -> N + M via MCP",
                fontsize=13, fontweight="bold", color=DARK)
ax_nm.set_xlim(0, 10); ax_nm.set_ylim(0, 10); ax_nm.axis("off")

# BEFORE
ax_nm.text(2.5, 9.3, "Before (N x M)", ha="center",
           fontweight="bold", fontsize=11, color=RED)
clients = [(0.7, 7.5), (0.7, 6.2), (0.7, 4.9), (0.7, 3.6)]
servers = [(4.3, 7.5), (4.3, 6.2), (4.3, 4.9), (4.3, 3.6)]
for x, y in clients:
    ax_nm.add_patch(plt.Circle((x, y), 0.3, facecolor=BLUE,
                               edgecolor="white", lw=1.5))
for x, y in servers:
    ax_nm.add_patch(plt.Circle((x, y), 0.3, facecolor=ORANGE,
                               edgecolor="white", lw=1.5))
for cx, cy in clients:
    for sx, sy in servers:
        ax_nm.plot([cx, sx], [cy, sy], "-", color=RED, lw=0.4, alpha=0.5)
ax_nm.text(2.5, 2.8, f"N*M = {len(clients)*len(servers)} links",
           ha="center", fontsize=10, color=RED, fontweight="bold")

# AFTER
ax_nm.text(7.5, 9.3, "With MCP (N + M)", ha="center",
           fontweight="bold", fontsize=11, color=GREEN)
clients2 = [(5.8, 7.5), (5.8, 6.2), (5.8, 4.9), (5.8, 3.6)]
servers2 = [(9.2, 7.5), (9.2, 6.2), (9.2, 4.9), (9.2, 3.6)]
for x, y in clients2:
    ax_nm.add_patch(plt.Circle((x, y), 0.3, facecolor=BLUE,
                               edgecolor="white", lw=1.5))
for x, y in servers2:
    ax_nm.add_patch(plt.Circle((x, y), 0.3, facecolor=ORANGE,
                               edgecolor="white", lw=1.5))
# MCP hub
ax_nm.add_patch(FancyBboxPatch((6.9, 5.2), 1.2, 1.6,
                               boxstyle="round,pad=0.1",
                               facecolor=PURPLE, edgecolor="white", lw=2))
ax_nm.text(7.5, 6.0, "MCP", ha="center", va="center",
           color="white", fontweight="bold", fontsize=12)
for cx, cy in clients2:
    ax_nm.plot([cx, 6.9], [cy, 6.0], "-", color=GREEN, lw=1)
for sx, sy in servers2:
    ax_nm.plot([8.1, sx], [6.0, sy], "-", color=GREEN, lw=1)
ax_nm.text(7.5, 2.8, f"N+M = {len(clients2)+len(servers2)} links",
           ha="center", fontsize=10, color=GREEN, fontweight="bold")

ax_nm.text(5, 1.5,
           "One protocol -> any client talks to any server.",
           ha="center", fontsize=10, color=DARK, style="italic")

# ═══════════════════════ PANEL 2 — Handshake ════════════════════════
ax_hs.set_title("2 · Client / Server Handshake (JSON-RPC)",
                fontsize=13, fontweight="bold", color=DARK)
ax_hs.set_xlim(0, 10); ax_hs.set_ylim(0, 10); ax_hs.axis("off")

# Two lifelines
ax_hs.plot([2.5, 2.5], [0.5, 9], "-", color=BLUE, lw=2)
ax_hs.plot([7.5, 7.5], [0.5, 9], "-", color=ORANGE, lw=2)
ax_hs.text(2.5, 9.4, "Client", ha="center", fontweight="bold",
           fontsize=11, color=BLUE)
ax_hs.text(7.5, 9.4, "Server", ha="center", fontweight="bold",
           fontsize=11, color=ORANGE)

msgs = [
    (8.2, "initialize",        "->", DARK),
    (7.3, "initializeResult",  "<-", GREEN),
    (6.2, "tools/list",        "->", DARK),
    (5.3, "tools/list result", "<-", GREEN),
    (4.2, "tools/call create_order", "->", DARK),
    (3.3, "result {order_id:42}",    "<-", GREEN),
    (2.2, "resources/read",    "->", DARK),
    (1.3, "resource content",  "<-", GREEN),
]
for y, txt, dir, c in msgs:
    if dir == "->":
        ax_hs.annotate("", xy=(7.3, y), xytext=(2.7, y),
                       arrowprops=dict(arrowstyle="-|>", color=c, lw=1.5))
        ax_hs.text(5.0, y + 0.18, txt, ha="center", fontsize=8.5,
                   color=DARK, fontweight="bold")
    else:
        ax_hs.annotate("", xy=(2.7, y), xytext=(7.3, y),
                       arrowprops=dict(arrowstyle="-|>", color=c, lw=1.5))
        ax_hs.text(5.0, y + 0.18, txt, ha="center", fontsize=8.5,
                   color=c, fontweight="bold")

ax_hs.text(5, 0.2,
           "All messages are JSON-RPC 2.0 over the chosen transport.",
           ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 3 — Three primitives ═════════════════
ax_prim.set_title("3 · The Three Primitives",
                  fontsize=13, fontweight="bold", color=DARK)
ax_prim.set_xlim(0, 10); ax_prim.set_ylim(0, 10); ax_prim.axis("off")

cards = [
    (2.0, "Resources", BLUE,
     "read-only\ndata / docs\n(menu, orders...)",  "GET a URI"),
    (5.0, "Tools", ORANGE,
     "side-effect\nfunctions\n(create_order)",      "POST args -> result"),
    (8.0, "Prompts", PURPLE,
     "reusable\nprompt templates\n(\"summarise\")", "parametrise + fetch"),
]
for x, name, c, body, verb in cards:
    ax_prim.add_patch(FancyBboxPatch((x - 1.4, 2.5), 2.8, 5.5,
                                     boxstyle="round,pad=0.1",
                                     facecolor=c, edgecolor="white", lw=2))
    ax_prim.text(x, 7.3, name, ha="center", fontweight="bold",
                 color="white", fontsize=12)
    ax_prim.text(x, 5.5, body, ha="center", fontsize=10,
                 color="white")
    ax_prim.text(x, 3.3, verb, ha="center", fontsize=9,
                 color="white", style="italic")

ax_prim.text(5, 1.2,
             "Resources = GET.  Tools = POST.  Prompts = templates.\n"
             "Client decides which ones the LLM sees.",
             ha="center", fontsize=9.5, color="#555", style="italic")

# ═══════════════════════ PANEL 4 — Transports ══════════════════════
ax_tr.set_title("4 · Transport Options",
                fontsize=13, fontweight="bold", color=DARK)
ax_tr.set_xlim(0, 10); ax_tr.set_ylim(0, 10); ax_tr.axis("off")

rows = [
    (8.2, "stdio",            GREEN,
     "local child process  /  fastest  /  same-machine only"),
    (6.4, "HTTP + SSE",       BLUE,
     "streaming events  /  firewall-friendly  /  long-lived"),
    (4.6, "Streamable HTTP",  PURPLE,
     "modern replacement  /  bidirectional  /  chunked"),
    (2.8, "(future) WebSocket", GREY,
     "full duplex  /  low overhead  /  less tooling"),
]
for y, name, c, body in rows:
    ax_tr.add_patch(FancyBboxPatch((0.6, y - 0.7), 8.8, 1.4,
                                   boxstyle="round,pad=0.05",
                                   facecolor=c, edgecolor="white", lw=2))
    ax_tr.text(2.2, y, name, ha="center", va="center",
               color="white", fontweight="bold", fontsize=10.5)
    ax_tr.text(6.2, y, body, ha="center", va="center",
               color="white", fontsize=9)
ax_tr.text(5, 1.6,
           "Same JSON-RPC payload on every transport — pick by deployment.",
           ha="center", fontsize=9, color="#555", style="italic")

out = str(Path(__file__).resolve().parent.parent / "img" / "MCP.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out}")
