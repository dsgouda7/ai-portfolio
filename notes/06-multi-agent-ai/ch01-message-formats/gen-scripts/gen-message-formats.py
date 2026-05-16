from pathlib import Path
"""Generate Message Formats.png — 4-panel xkcd-style concept sketch.

Panels:
  (1) Anatomy of an agent message (role, id, correlation, payload, meta)
  (2) Three payload styles: full state / delta / blackboard reference
  (3) Context-window growth across turns
  (4) Serialization tradeoffs: JSON / MessagePack / Protobuf
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.xkcd(scale=0.8, length=100, randomness=2)

fig = plt.figure(figsize=(18, 11), facecolor="white")
fig.suptitle("Message Formats & Shared Context",
             fontsize=22, fontweight="bold", y=0.98)

gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.28,
                      left=0.05, right=0.97, top=0.91, bottom=0.06)
ax_anat = fig.add_subplot(gs[0, 0])
ax_pay = fig.add_subplot(gs[0, 1])
ax_ctx = fig.add_subplot(gs[1, 0])
ax_ser = fig.add_subplot(gs[1, 1])

BLUE, ORANGE, GREEN, RED, PURPLE, DARK, GREY = (
    "#2E86C1", "#E67E22", "#27AE60", "#E74C3C", "#8E44AD", "#2C3E50", "#BDC3C7")

# ═══════════════════════ PANEL 1 — Message anatomy ══════════════════
ax_anat.set_title("1 · Anatomy of an Agent Message",
                  fontsize=13, fontweight="bold", color=DARK)
ax_anat.set_xlim(0, 10); ax_anat.set_ylim(0, 10); ax_anat.axis("off")

fields = [
    (8.6, "role",           PURPLE, '"planner" | "worker"'),
    (7.4, "id",             BLUE,   "msg_9f3c..."),
    (6.2, "correlation_id", BLUE,   "task_42"),
    (5.0, "sender / receiver", GREEN, "PlannerAgent -> BillingAgent"),
    (3.8, "payload",        ORANGE, "{\"order\": {...}}"),
    (2.6, "schema_version", GREY,   "v2"),
    (1.4, "meta",           DARK,   "ts, auth, signature"),
]
for y, name, c, body in fields:
    ax_anat.add_patch(FancyBboxPatch((0.3, y - 0.45), 2.6, 0.9,
                                     boxstyle="round,pad=0.05",
                                     facecolor=c, edgecolor="white", lw=1.5))
    ax_anat.text(1.6, y, name, ha="center", va="center",
                 color="white", fontweight="bold", fontsize=9.5)
    ax_anat.text(3.1, y, body, va="center", fontsize=9,
                 color=DARK, family="monospace")
ax_anat.text(5, 0.4,
             "Missing any of these turns a multi-agent system into guesswork.",
             ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 2 — Payload styles ═══════════════════
ax_pay.set_title("2 · Three Payload Styles",
                 fontsize=13, fontweight="bold", color=DARK)
ax_pay.set_xlim(0, 10); ax_pay.set_ylim(0, 10); ax_pay.axis("off")

cards = [
    (8.0, "Full state",    BLUE,
     "send ENTIRE shared\nstate every hop.\n+simple  -fat"),
    (5.0, "Delta",         ORANGE,
     "send only what\nchanged.\n+cheap  -order-sensitive"),
    (2.0, "Blackboard ref", GREEN,
     "send a key\ninto shared store.\n+small msgs  -needs store"),
]
for y, name, c, body in cards:
    ax_pay.add_patch(FancyBboxPatch((0.5, y - 1.1), 9.0, 2.2,
                                    boxstyle="round,pad=0.08",
                                    facecolor=c, edgecolor="white", lw=2))
    ax_pay.text(2.3, y, name, ha="center", va="center",
                color="white", fontweight="bold", fontsize=11)
    ax_pay.text(6.5, y, body, ha="center", va="center",
                color="white", fontsize=9.5)

ax_pay.text(5, 0.3,
            "Pick ONE per system and document it. Mixing styles = hidden bugs.",
            ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 3 — Context growth ═══════════════════
ax_ctx.set_title("3 · Shared-Context Growth Across Turns",
                 fontsize=13, fontweight="bold", color=DARK)
turns = np.arange(1, 21)
# full-state: linear + slight inflation
full = 200 + 350 * turns
# delta: mostly flat small
delta = 200 + 40 * turns + 5 * turns
# blackboard: tiny, just references
bb = 180 + 10 * turns

ax_ctx.plot(turns, full,  "-o", color=BLUE,   lw=2.5, markersize=5,
            label="full state")
ax_ctx.plot(turns, delta, "-s", color=ORANGE, lw=2.5, markersize=5,
            label="delta")
ax_ctx.plot(turns, bb,    "-^", color=GREEN,  lw=2.5, markersize=5,
            label="blackboard ref")
ax_ctx.axhline(8000, color=RED, ls="--", lw=1.5, alpha=0.7)
ax_ctx.text(18, 8300, "context cap", color=RED, fontsize=9,
            fontweight="bold", ha="right")

ax_ctx.set_xlabel("turns", fontsize=10, color=DARK)
ax_ctx.set_ylabel("tokens in transit", fontsize=10, color=DARK)
ax_ctx.legend(fontsize=9, loc="upper left", framealpha=0.9)
ax_ctx.set_xlim(1, 20); ax_ctx.set_ylim(0, 10000)
ax_ctx.text(0.5, -0.25,
            "Full-state saturates fast; delta and blackboard scale.",
            transform=ax_ctx.transAxes, ha="center",
            fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 4 — Serialization ═══════════════════
ax_ser.set_title("4 · Wire Format Tradeoffs",
                 fontsize=13, fontweight="bold", color=DARK)
formats = ["JSON", "MessagePack", "Protobuf", "Avro"]
size  = [100, 55, 40, 38]          # relative bytes
speed = [50,  70, 95, 90]          # relative encode/decode speed
human = [100, 25, 5,  10]          # human-readability

x = np.arange(len(formats))
w = 0.26
ax_ser.bar(x - w, size,  w, color=BLUE,   edgecolor="white", label="size (rel)")
ax_ser.bar(x,     speed, w, color=GREEN,  edgecolor="white", label="speed (rel)")
ax_ser.bar(x + w, human, w, color=ORANGE, edgecolor="white", label="human-readable")
ax_ser.set_xticks(x); ax_ser.set_xticklabels(formats, fontsize=10)
ax_ser.set_ylim(0, 120)
ax_ser.legend(fontsize=9, loc="upper right", framealpha=0.9)
ax_ser.text(0.5, -0.28,
            "Start with JSON. Switch to MessagePack/Protobuf only\n"
            "when payload cost or latency is measurably hurting you.",
            transform=ax_ser.transAxes, ha="center",
            fontsize=9, color="#555", style="italic")

out = str(Path(__file__).resolve().parent.parent / "img" / "Message Formats.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out}")
