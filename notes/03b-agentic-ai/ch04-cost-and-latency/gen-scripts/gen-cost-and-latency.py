from pathlib import Path
"""Generate Cost and Latency.png — 4-panel xkcd-style concept sketch.

Panels:
  (1) Token-cost stack (input vs output pricing)
  (2) Latency components: TTFT / TPOT / tool calls
  (3) Cost-vs-accuracy Pareto across model tiers
  (4) Optimisation patterns: cache, routing, batching
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.xkcd(scale=0.8, length=100, randomness=2)

fig = plt.figure(figsize=(18, 11), facecolor="white")
fig.suptitle("Cost & Latency", fontsize=22, fontweight="bold", y=0.98)

gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.30,
                      left=0.06, right=0.97, top=0.91, bottom=0.06)
ax_cost = fig.add_subplot(gs[0, 0])
ax_lat = fig.add_subplot(gs[0, 1])
ax_par = fig.add_subplot(gs[1, 0])
ax_opt = fig.add_subplot(gs[1, 1])

BLUE, ORANGE, GREEN, RED, PURPLE, DARK, GREY = (
    "#2E86C1", "#E67E22", "#27AE60", "#E74C3C", "#8E44AD", "#2C3E50", "#BDC3C7")

# ═══════════════════════ PANEL 1 — Token cost stack ═════════════════
ax_cost.set_title("1 · Where Tokens Come From (per request)",
                  fontsize=13, fontweight="bold", color=DARK)
parts = ["system\nprompt", "few-shot", "retrieval\n(k=5)",
         "user msg", "reasoning", "answer"]
# input-priced (system..user..reasoning partly) vs output-priced
input_tok  = [250, 800, 2500, 120, 0,   0]
output_tok = [0,   0,   0,    0,   600, 300]
x = np.arange(len(parts))
ax_cost.bar(x, input_tok, color=BLUE, edgecolor="white",
            label="input ($3/Mtok)")
ax_cost.bar(x, output_tok, bottom=input_tok, color=ORANGE,
            edgecolor="white", label="output ($15/Mtok)")
for xi, iv, ov in zip(x, input_tok, output_tok):
    total = iv + ov
    if total:
        ax_cost.text(xi, total + 50, str(total), ha="center",
                     fontsize=8.5, color=DARK)
ax_cost.set_xticks(x); ax_cost.set_xticklabels(parts, fontsize=9)
ax_cost.set_ylabel("tokens", fontsize=10, color=DARK)
ax_cost.legend(fontsize=9, framealpha=0.9)
total_in = sum(input_tok); total_out = sum(output_tok)
cost = total_in/1e6*3 + total_out/1e6*15
ax_cost.text(0.5, -0.28,
             f"Total: {total_in+total_out} tok -> ~${cost:.4f} per turn\n"
             f"RAG retrieval is usually the biggest input line.",
             transform=ax_cost.transAxes, ha="center",
             fontsize=9.5, color="#555", style="italic")

# ═══════════════════════ PANEL 2 — Latency components ═══════════════
ax_lat.set_title("2 · Latency Components",
                 fontsize=13, fontweight="bold", color=DARK)
segs = [
    ("network RTT",  80,  GREY),
    ("retrieval",    150, BLUE),
    ("TTFT",         350, PURPLE),
    ("generate 300 tok (TPOT 25 ms)", 300 * 25, ORANGE),
    ("tool call",    250, GREEN),
]
t0 = 0
for name, dur, c in segs:
    ax_lat.barh(0, dur, left=t0, color=c, edgecolor="white", height=0.7)
    ax_lat.text(t0 + dur/2, 0, f"{name}\n{dur} ms",
                ha="center", va="center", fontsize=9,
                color="white", fontweight="bold")
    t0 += dur

ax_lat.set_xlim(0, t0 + 200)
ax_lat.set_ylim(-1, 1)
ax_lat.set_yticks([])
ax_lat.set_xlabel("ms (p50)", fontsize=10, color=DARK)
ax_lat.text(0.5, -0.3,
            f"Total ~ {t0} ms.  Generation dominates (tokens x TPOT).\n"
            "TTFT matters for perceived latency; stream to mask it.",
            transform=ax_lat.transAxes, ha="center",
            fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 3 — Cost vs accuracy Pareto ══════════
ax_par.set_title("3 · Cost vs Accuracy — Model Tiers",
                 fontsize=13, fontweight="bold", color=DARK)
models = [
    ("nano",   0.72, 0.15, BLUE),
    ("mini",   0.82, 0.60, GREEN),
    ("mid",    0.88, 2.0,  ORANGE),
    ("flagship", 0.93, 10.0, PURPLE),
    ("reasoning", 0.96, 40.0, RED),
]
for name, acc, cost, c in models:
    ax_par.plot(cost, acc, "o", color=c, markersize=18,
                markeredgecolor="white", markeredgewidth=2)
    ax_par.annotate(name, (cost, acc), textcoords="offset points",
                    xytext=(8, 8), fontsize=9.5, color=c,
                    fontweight="bold")
# frontier curve
xs = np.array([m[2] for m in models])
ys = np.array([m[1] for m in models])
order = np.argsort(xs)
ax_par.plot(xs[order], ys[order], "--", color=GREY, lw=1.5, alpha=0.7)
ax_par.set_xscale("log")
ax_par.set_xlabel("$ / Mtok (log)", fontsize=10, color=DARK)
ax_par.set_ylabel("accuracy", fontsize=10, color=DARK)
ax_par.set_ylim(0.65, 1.0)
ax_par.text(0.5, -0.28,
            "Route cheap queries to small models.\n"
            "Reserve flagship/reasoning tier for hard tasks.",
            transform=ax_par.transAxes, ha="center",
            fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 4 — Optimisation patterns ════════════
ax_opt.set_title("4 · Cost Optimisation Patterns",
                 fontsize=13, fontweight="bold", color=DARK)
ax_opt.set_xlim(0, 10); ax_opt.set_ylim(0, 10); ax_opt.axis("off")

patterns = [
    ("Semantic cache",     "skip identical/near-dup questions",     GREEN,  9.0),
    ("Router / cascade",   "small -> large only if confidence low", BLUE,   7.5),
    ("Prompt compression", "summarise history, drop boilerplate",   PURPLE, 6.0),
    ("Batching",           "group async calls",                     ORANGE, 4.5),
    ("Stream + stop early","stop tokens, length caps",              DARK,   3.0),
    ("Fine-tune small",    "QLoRA small model for one task",        RED,    1.5),
]
for name, body, c, y in patterns:
    ax_opt.add_patch(FancyBboxPatch((0.3, y - 0.45), 3.2, 0.9,
                                    boxstyle="round,pad=0.05",
                                    facecolor=c, edgecolor="white", lw=1.5))
    ax_opt.text(1.9, y, name, ha="center", va="center",
                color="white", fontweight="bold", fontsize=10)
    ax_opt.text(3.8, y, body, va="center", fontsize=10, color=DARK)

ax_opt.text(5, 0.3,
            "Stack multiple — each saves 20-60%.",
            ha="center", fontsize=9, color="#555", style="italic")

out = str(Path(__file__).resolve().parent.parent / "img" / "Cost and Latency.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out}")
