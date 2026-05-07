from pathlib import Path
"""Generate Evaluating AI Systems.png — 4-panel xkcd-style concept sketch.

Panels:
  (1) RAGAS 4 metrics radar
  (2) Reasoning-trace evaluation checklist
  (3) Component-level eval pipeline (retriever, reranker, generator)
  (4) Hallucination detection gate
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.xkcd(scale=0.8, length=100, randomness=2)

fig = plt.figure(figsize=(18, 11), facecolor="white")
fig.suptitle("Evaluating AI Systems", fontsize=22, fontweight="bold", y=0.98)

gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.30,
                      left=0.05, right=0.97, top=0.91, bottom=0.06)
ax_rad = fig.add_subplot(gs[0, 0], projection="polar")
ax_trace = fig.add_subplot(gs[0, 1])
ax_comp = fig.add_subplot(gs[1, 0])
ax_hal = fig.add_subplot(gs[1, 1])

BLUE, ORANGE, GREEN, RED, PURPLE, DARK, GREY = (
    "#2E86C1", "#E67E22", "#27AE60", "#E74C3C", "#8E44AD", "#2C3E50", "#BDC3C7")

# ═══════════════════════ PANEL 1 — RAGAS radar ══════════════════════
ax_rad.set_title("1 · RAGAS Metrics", fontsize=13,
                 fontweight="bold", color=DARK, pad=20)
labels = ["faithfulness", "answer\nrelevance", "context\nprecision",
          "context\nrecall"]
baseline = [0.55, 0.60, 0.50, 0.45]
tuned =    [0.88, 0.90, 0.82, 0.78]
angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
baseline += baseline[:1]; tuned += tuned[:1]; angles += angles[:1]
ax_rad.plot(angles, baseline, "-", color=RED, lw=2, label="baseline")
ax_rad.fill(angles, baseline, color=RED, alpha=0.15)
ax_rad.plot(angles, tuned, "-", color=GREEN, lw=2, label="tuned")
ax_rad.fill(angles, tuned, color=GREEN, alpha=0.20)
ax_rad.set_xticks(angles[:-1])
ax_rad.set_xticklabels(labels, fontsize=9.5, color=DARK)
ax_rad.set_ylim(0, 1.0)
ax_rad.set_yticks([0.25, 0.5, 0.75, 1.0])
ax_rad.set_yticklabels(["0.25","0.5","0.75","1"], fontsize=8)
ax_rad.legend(fontsize=9, loc="upper right", bbox_to_anchor=(1.25, 1.1))

# ═══════════════════════ PANEL 2 — Reasoning trace ══════════════════
ax_trace.set_title("2 · Reasoning-Trace Evaluation", fontsize=13,
                   fontweight="bold", color=DARK)
ax_trace.set_xlim(0, 10); ax_trace.set_ylim(0, 10); ax_trace.axis("off")

checks = [
    (9.0, "Tool choice correct",        GREEN),
    (7.8, "Tool args well-formed",      GREEN),
    (6.6, "Observation used in next step", ORANGE),
    (5.4, "No redundant calls",         ORANGE),
    (4.2, "Terminates (<= max steps)",  GREEN),
    (3.0, "Final answer grounded",      GREEN),
    (1.8, "Latency budget respected",   BLUE),
]
for y, name, c in checks:
    ax_trace.add_patch(FancyBboxPatch((0.4, y - 0.45), 0.9, 0.9,
                                      boxstyle="round,pad=0.05",
                                      facecolor=c, edgecolor="white", lw=1.5))
    ax_trace.text(0.85, y, "OK", ha="center", va="center",
                  color="white", fontweight="bold", fontsize=10)
    ax_trace.text(1.6, y, name, va="center", fontsize=10, color=DARK)

ax_trace.text(5, 0.6,
              "Grade each trace on structure AND outcome.\n"
              "Automate with an LLM-as-judge + spot-check.",
              ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 3 — Component pipeline ═══════════════
ax_comp.set_title("3 · Component-Level Evaluation",
                  fontsize=13, fontweight="bold", color=DARK)
ax_comp.set_xlim(0, 12); ax_comp.set_ylim(0, 6); ax_comp.axis("off")

comps = [
    (0.5, "Retriever", BLUE,   "recall@k\nMRR"),
    (3.2, "Reranker",  PURPLE, "nDCG@10"),
    (5.9, "Generator", ORANGE, "faithfulness\nrelevance"),
    (8.6, "Safety",    RED,    "toxicity\njailbreak"),
    (11.0, "End-to-end", GREEN, "user SAT\ntask success"),
]
for i, (x, name, c, body) in enumerate(comps):
    ax_comp.add_patch(FancyBboxPatch((x, 2.0), 1.8, 2.6,
                                     boxstyle="round,pad=0.08",
                                     facecolor=c, edgecolor="white", lw=2))
    ax_comp.text(x + 0.9, 4.1, name, ha="center", va="center",
                 color="white", fontweight="bold", fontsize=10)
    ax_comp.text(x + 0.9, 2.9, body, ha="center", va="center",
                 color="white", fontsize=8.5)
    if i < len(comps) - 1:
        ax_comp.annotate("", xy=(comps[i+1][0] - 0.05, 3.3),
                         xytext=(x + 1.85, 3.3),
                         arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.8))
ax_comp.text(6, 0.6,
             "Evaluate each stage in isolation, then end-to-end.\n"
             "Top-level metric hides which component failed.",
             ha="center", fontsize=9.5, color="#555", style="italic")

# ═══════════════════════ PANEL 4 — Hallucination gate ═══════════════
ax_hal.set_title("4 · Hallucination Detection Gate", fontsize=13,
                 fontweight="bold", color=DARK)
ax_hal.set_xlim(0, 10); ax_hal.set_ylim(0, 10); ax_hal.axis("off")

# flow: answer -> claims -> NLI check -> pass/fail
stages = [
    (0.3, 7.0, "LLM\nanswer",     BLUE),
    (3.0, 7.0, "extract\nclaims", PURPLE),
    (5.8, 7.0, "NLI vs\ncontext", ORANGE),
    (8.6, 7.0, "score",           GREEN),
]
for x, y, name, c in stages:
    ax_hal.add_patch(FancyBboxPatch((x, y - 1.0), 1.3, 2.0,
                                    boxstyle="round,pad=0.05",
                                    facecolor=c, edgecolor="white", lw=2))
    ax_hal.text(x + 0.65, y, name, ha="center", va="center",
                color="white", fontweight="bold", fontsize=9)
for a, b in [(0, 1), (1, 2), (2, 3)]:
    x1 = stages[a][0] + 1.35
    x2 = stages[b][0] - 0.05
    ax_hal.annotate("", xy=(x2, 7.0), xytext=(x1, 7.0),
                    arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.8))

# Decision branches
ax_hal.add_patch(FancyBboxPatch((6.5, 2.5), 3.0, 1.4,
                                boxstyle="round,pad=0.05",
                                facecolor=GREEN, edgecolor="white", lw=2))
ax_hal.text(8.0, 3.2, ">=0.9  -> ship", ha="center", va="center",
            color="white", fontweight="bold", fontsize=10)
ax_hal.add_patch(FancyBboxPatch((3.2, 2.5), 3.0, 1.4,
                                boxstyle="round,pad=0.05",
                                facecolor=ORANGE, edgecolor="white", lw=2))
ax_hal.text(4.7, 3.2, "0.6-0.9 -> retry/hedge", ha="center", va="center",
            color="white", fontweight="bold", fontsize=10)
ax_hal.add_patch(FancyBboxPatch((0.3, 2.5), 2.6, 1.4,
                                boxstyle="round,pad=0.05",
                                facecolor=RED, edgecolor="white", lw=2))
ax_hal.text(1.6, 3.2, "<0.6 -> block", ha="center", va="center",
            color="white", fontweight="bold", fontsize=10)

ax_hal.annotate("", xy=(1.6, 3.9), xytext=(9.3, 5.9),
                arrowprops=dict(arrowstyle="-", color=GREY, lw=1,
                                linestyle=":"))
ax_hal.annotate("", xy=(4.7, 3.9), xytext=(9.3, 5.9),
                arrowprops=dict(arrowstyle="-", color=GREY, lw=1,
                                linestyle=":"))
ax_hal.annotate("", xy=(8.0, 3.9), xytext=(9.3, 5.9),
                arrowprops=dict(arrowstyle="-", color=GREY, lw=1,
                                linestyle=":"))

ax_hal.text(5, 1.0,
            "Claim-level NLI against retrieved context =\n"
            "cheapest high-signal hallucination filter.",
            ha="center", fontsize=9, color="#555", style="italic")

out = str(Path(__file__).resolve().parent.parent / "img" / "Evaluating AI Systems.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out}")
