from pathlib import Path
"""Generate Event-Driven Agents.png — 4-panel xkcd-style concept sketch.

Panels:
  (1) Sync vs Async messaging
  (2) Pub/Sub with correlation + DLQ
  (3) Delivery guarantees: at-most / at-least / exactly-once
  (4) Fan-out / Fan-in aggregator pattern
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.xkcd(scale=0.8, length=100, randomness=2)

fig = plt.figure(figsize=(18, 11), facecolor="white")
fig.suptitle("Event-Driven Agent Messaging",
             fontsize=22, fontweight="bold", y=0.98)

gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.28,
                      left=0.05, right=0.97, top=0.91, bottom=0.06)
ax_sync = fig.add_subplot(gs[0, 0])
ax_ps = fig.add_subplot(gs[0, 1])
ax_del = fig.add_subplot(gs[1, 0])
ax_fan = fig.add_subplot(gs[1, 1])

BLUE, ORANGE, GREEN, RED, PURPLE, DARK, GREY = (
    "#2E86C1", "#E67E22", "#27AE60", "#E74C3C", "#8E44AD", "#2C3E50", "#BDC3C7")

# ═══════════════════════ PANEL 1 — Sync vs Async ═══════════════════
ax_sync.set_title("1 · Sync vs Async Between Agents",
                  fontsize=13, fontweight="bold", color=DARK)
ax_sync.set_xlim(0, 10); ax_sync.set_ylim(0, 10); ax_sync.axis("off")

# Sync
ax_sync.text(2.5, 9.3, "Synchronous (blocks)", ha="center",
             fontweight="bold", fontsize=11, color=RED)
ax_sync.add_patch(plt.Circle((1.2, 7.0), 0.4, facecolor=BLUE, lw=1.5,
                             edgecolor="white"))
ax_sync.text(1.2, 7.0, "A", ha="center", va="center", color="white",
             fontweight="bold", fontsize=11)
ax_sync.add_patch(plt.Circle((4.0, 7.0), 0.4, facecolor=ORANGE, lw=1.5,
                             edgecolor="white"))
ax_sync.text(4.0, 7.0, "B", ha="center", va="center", color="white",
             fontweight="bold", fontsize=11)
ax_sync.annotate("", xy=(3.6, 7.0), xytext=(1.6, 7.0),
                 arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.5))
ax_sync.text(2.6, 7.4, "request", ha="center", fontsize=9, color=DARK)
ax_sync.annotate("", xy=(1.6, 6.3), xytext=(3.6, 6.3),
                 arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.5))
ax_sync.text(2.6, 5.9, "wait... reply", ha="center", fontsize=9, color=RED)
ax_sync.text(2.5, 5.0,
             "- coupled\n- head-of-line\n  blocking\n- fragile under load",
             ha="center", fontsize=9.5, color=DARK)

# Async
ax_sync.text(7.5, 9.3, "Asynchronous (pub/sub)", ha="center",
             fontweight="bold", fontsize=11, color=GREEN)
ax_sync.add_patch(plt.Circle((5.8, 7.0), 0.4, facecolor=BLUE, lw=1.5,
                             edgecolor="white"))
ax_sync.text(5.8, 7.0, "A", ha="center", va="center", color="white",
             fontweight="bold", fontsize=11)
ax_sync.add_patch(FancyBboxPatch((7.0, 6.4), 1.3, 1.2,
                                 boxstyle="round,pad=0.05",
                                 facecolor=PURPLE, edgecolor="white", lw=1.5))
ax_sync.text(7.65, 7.0, "bus", ha="center", va="center",
             color="white", fontweight="bold", fontsize=10)
ax_sync.add_patch(plt.Circle((9.1, 7.0), 0.4, facecolor=ORANGE, lw=1.5,
                             edgecolor="white"))
ax_sync.text(9.1, 7.0, "B", ha="center", va="center", color="white",
             fontweight="bold", fontsize=11)
ax_sync.annotate("", xy=(6.95, 7.0), xytext=(6.2, 7.0),
                 arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.5))
ax_sync.annotate("", xy=(8.7, 7.0), xytext=(8.35, 7.0),
                 arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.5))
ax_sync.text(7.5, 5.0,
             "+ decoupled\n+ back-pressure\n+ retries / DLQ\n+ scale independently",
             ha="center", fontsize=9.5, color=DARK)

ax_sync.text(5, 1.3,
             "Use async the moment a workflow has > 2 hops or slow tasks.",
             ha="center", fontsize=9.5, color="#555", style="italic")

# ═══════════════════════ PANEL 2 — Pub/Sub with correlation ════════
ax_ps.set_title("2 · Pub/Sub with Correlation + DLQ",
                fontsize=13, fontweight="bold", color=DARK)
ax_ps.set_xlim(0, 10); ax_ps.set_ylim(0, 10); ax_ps.axis("off")

# producer
ax_ps.add_patch(FancyBboxPatch((0.3, 5.0), 1.8, 1.4,
                               boxstyle="round,pad=0.05",
                               facecolor=BLUE, edgecolor="white", lw=2))
ax_ps.text(1.2, 5.7, "Producer", ha="center", va="center",
           color="white", fontweight="bold", fontsize=10)

# Topic
ax_ps.add_patch(FancyBboxPatch((3.0, 4.2), 2.6, 3.0,
                               boxstyle="round,pad=0.1",
                               facecolor=PURPLE, edgecolor="white", lw=2))
ax_ps.text(4.3, 6.7, "Topic: orders", ha="center",
           color="white", fontweight="bold", fontsize=10)
for i in range(4):
    ax_ps.add_patch(FancyBboxPatch((3.2 + i*0.55, 5.0), 0.45, 0.5,
                                   boxstyle="round,pad=0.02",
                                   facecolor="white", edgecolor=PURPLE, lw=1))
ax_ps.text(4.3, 4.6, "msgs", ha="center", fontsize=8.5, color="white")

# Subscribers
subs = [(8.0, 7.5, "Pricing"),
        (8.0, 5.7, "Inventory"),
        (8.0, 3.9, "Notify")]
for x, y, name in subs:
    ax_ps.add_patch(FancyBboxPatch((x - 0.9, y - 0.5), 1.8, 1.0,
                                   boxstyle="round,pad=0.05",
                                   facecolor=ORANGE, edgecolor="white", lw=1.5))
    ax_ps.text(x, y, name, ha="center", va="center",
               color="white", fontweight="bold", fontsize=9)
    ax_ps.annotate("", xy=(x - 0.9, y), xytext=(5.7, 5.7),
                   arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.3))

# DLQ
ax_ps.add_patch(FancyBboxPatch((4.0, 1.3), 2.6, 1.1,
                               boxstyle="round,pad=0.05",
                               facecolor=RED, edgecolor="white", lw=2))
ax_ps.text(5.3, 1.85, "DLQ (poison msgs)", ha="center", va="center",
           color="white", fontweight="bold", fontsize=10)
ax_ps.annotate("", xy=(5.3, 2.4), xytext=(7.1, 5.2),
               arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.3,
                               linestyle=":"))

ax_ps.annotate("", xy=(3.0, 5.7), xytext=(2.1, 5.7),
               arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.5))

ax_ps.text(5, 0.4,
           "Every msg carries correlation_id. N retries -> DLQ.",
           ha="center", fontsize=9.5, color="#555", style="italic")

# ═══════════════════════ PANEL 3 — Delivery guarantees ═════════════
ax_del.set_title("3 · Delivery Guarantees",
                 fontsize=13, fontweight="bold", color=DARK)
ax_del.set_xlim(0, 10); ax_del.set_ylim(0, 10); ax_del.axis("off")

rows = [
    (8.2, "at-most-once",  RED,
     "fire & forget - possible loss",      "metrics, logs"),
    (5.8, "at-least-once", ORANGE,
     "may duplicate - needs idempotency",  "most agent systems"),
    (3.4, "exactly-once",  GREEN,
     "dedup via keys + transactional txn", "payments, state changes"),
]
for y, name, c, when, ex in rows:
    ax_del.add_patch(FancyBboxPatch((0.5, y - 1.0), 9.0, 2.0,
                                    boxstyle="round,pad=0.05",
                                    facecolor=c, edgecolor="white", lw=2))
    ax_del.text(2.0, y, name, ha="center", va="center",
                color="white", fontweight="bold", fontsize=11)
    ax_del.text(5.5, y + 0.3, when, ha="center", va="center",
                color="white", fontsize=10)
    ax_del.text(5.5, y - 0.4, "use: " + ex, ha="center", va="center",
                color="white", fontsize=9, style="italic")

ax_del.text(5, 1.3,
            "Idempotency_key + dedup window fakes exactly-once cheaply.",
            ha="center", fontsize=9.5, color="#555", style="italic")

# ═══════════════════════ PANEL 4 — Fan out / in ════════════════════
ax_fan.set_title("4 · Fan-Out / Fan-In (parallel + aggregate)",
                 fontsize=13, fontweight="bold", color=DARK)
ax_fan.set_xlim(0, 10); ax_fan.set_ylim(0, 10); ax_fan.axis("off")

# Coordinator
ax_fan.add_patch(FancyBboxPatch((4.0, 8.0), 2.0, 1.2,
                                boxstyle="round,pad=0.05",
                                facecolor=BLUE, edgecolor="white", lw=2))
ax_fan.text(5.0, 8.6, "Coordinator", ha="center", va="center",
            color="white", fontweight="bold", fontsize=10)

# Workers
workers = [(1.5, 5.0), (3.8, 5.0), (6.2, 5.0), (8.5, 5.0)]
for i, (x, y) in enumerate(workers):
    ax_fan.add_patch(FancyBboxPatch((x - 0.8, y - 0.5), 1.6, 1.0,
                                    boxstyle="round,pad=0.05",
                                    facecolor=ORANGE, edgecolor="white", lw=1.5))
    ax_fan.text(x, y, f"W{i+1}", ha="center", va="center",
                color="white", fontweight="bold", fontsize=10)
    ax_fan.annotate("", xy=(x, y + 0.55), xytext=(5, 7.95),
                    arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.3))

# Aggregator
ax_fan.add_patch(FancyBboxPatch((3.8, 1.8), 2.4, 1.2,
                                boxstyle="round,pad=0.05",
                                facecolor=PURPLE, edgecolor="white", lw=2))
ax_fan.text(5.0, 2.4, "Aggregator", ha="center", va="center",
            color="white", fontweight="bold", fontsize=10)
for x, y in workers:
    ax_fan.annotate("", xy=(5.0, 3.0), xytext=(x, y - 0.55),
                    arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.2,
                                    linestyle=":"))

ax_fan.text(5, 0.6,
            "Waits for all correlation_ids or quorum.\n"
            "Timeout -> partial result / retry.",
            ha="center", fontsize=9.5, color="#555", style="italic")

out = str(Path(__file__).resolve().parent.parent / "img" / "Event-Driven Agents.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out}")
