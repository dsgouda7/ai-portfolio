from pathlib import Path
"""Generate Ensemble Voting.png — 3+ models vote on answer.

Shows each model's output and confidence, then aggregates via voting/weighted average.
4-panel xkcd-style visualization of ensemble model consensus.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Wedge
import numpy as np

plt.xkcd(scale=0.8, length=100, randomness=2)

fig = plt.figure(figsize=(18, 11), facecolor="white")
fig.suptitle("Ensemble Voting — Multi-Model Consensus", 
             fontsize=22, fontweight="bold", y=0.98)

gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.30,
                      left=0.06, right=0.97, top=0.91, bottom=0.06)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

BLUE, ORANGE, GREEN, RED, PURPLE, DARK, GREY = (
    "#2E86C1", "#E67E22", "#27AE60", "#E74C3C", "#8E44AD", "#2C3E50", "#BDC3C7")

# ═══════════════════════ PANEL 1 — Problem setup ═══════════════════════
ax1.set_title("1 · The Challenge — Ambiguous Medical Query",
              fontsize=13, fontweight="bold", color=DARK)
ax1.set_xlim(0, 10); ax1.set_ylim(0, 10); ax1.axis("off")

# Query
ax1.add_patch(FancyBboxPatch((0.5, 7.0), 9, 2.5,
                             boxstyle="round,pad=0.1",
                             facecolor=BLUE, edgecolor="white", lw=2, alpha=0.2))
ax1.text(5, 8.8, '🏥 Query:', ha="center", fontsize=11, 
         color=DARK, fontweight="bold")
ax1.text(5, 8.2, '"Patient has fever + cough + fatigue for 5 days."',
         ha="center", fontsize=10, color=DARK)
ax1.text(5, 7.6, '"Should we test for flu, COVID, or pneumonia?"',
         ha="center", fontsize=10, color=DARK)

# Single model risk
ax1.add_patch(FancyBboxPatch((1.5, 4.0), 7, 2.5,
                             boxstyle="round,pad=0.1",
                             facecolor=RED, edgecolor="white", lw=2, alpha=0.2))
ax1.text(5, 5.9, "❌ Single Model Risk:",
         ha="center", fontsize=10, color=RED, fontweight="bold")
ax1.text(5, 5.3, "• May hallucinate rare diagnoses",
         ha="center", fontsize=9, color=DARK)
ax1.text(5, 4.8, "• Overconfident on edge cases",
         ha="center", fontsize=9, color=DARK)
ax1.text(5, 4.3, "• No second opinion",
         ha="center", fontsize=9, color=DARK)

ax1.text(5, 2.0, "High-stakes decisions need\nmultiple independent opinions.",
         ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 2 — Individual model outputs ════════════
ax2.set_title("2 · Each Model Votes (Independent Reasoning)",
              fontsize=13, fontweight="bold", color=DARK)
ax2.set_xlim(0, 10); ax2.set_ylim(0, 10); ax2.axis("off")

models = [
    ("GPT-4", "COVID-19", 0.75, BLUE, 8.5),
    ("Claude 3", "COVID-19", 0.68, PURPLE, 6.5),
    ("Gemini", "Influenza", 0.62, ORANGE, 4.5),
    ("Llama 3", "COVID-19", 0.71, GREEN, 2.5),
]

for model, diagnosis, confidence, color, y in models:
    # Model name + diagnosis
    ax2.add_patch(FancyBboxPatch((0.5, y - 0.4), 4.5, 0.8,
                                 boxstyle="round,pad=0.05",
                                 facecolor=color, edgecolor="white", lw=2, alpha=0.3))
    ax2.text(2.75, y, f"{model}: {diagnosis}",
             ha="center", va="center", fontsize=9, color=DARK, fontweight="bold")
    
    # Confidence bar
    bar_width = 4 * confidence
    ax2.add_patch(FancyBboxPatch((5.5, y - 0.25), bar_width, 0.5,
                                 boxstyle="round,pad=0.02",
                                 facecolor=color, edgecolor="white", lw=1))
    ax2.text(9.5, y, f"{confidence:.0%}",
             ha="right", va="center", fontsize=8, color=DARK)

ax2.text(5, 0.8, "Each model reasons independently\nwith its own training & architecture.",
         ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 3 — Voting aggregation ══════════════════
ax3.set_title("3 · Aggregation Strategies",
              fontsize=13, fontweight="bold", color=DARK)
ax3.set_xlim(0, 10); ax3.set_ylim(0, 10); ax3.axis("off")

# Simple majority vote
ax3.add_patch(FancyBboxPatch((0.5, 7.0), 4.5, 2.5,
                             boxstyle="round,pad=0.1",
                             facecolor=GREEN, edgecolor="white", lw=2, alpha=0.2))
ax3.text(2.75, 9.0, "🗳️ Majority Vote",
         ha="center", fontsize=10, color=GREEN, fontweight="bold")
ax3.text(2.75, 8.3, "COVID: 3 votes",
         ha="center", fontsize=9, color=DARK, fontweight="bold")
ax3.text(2.75, 7.8, "Flu: 1 vote",
         ha="center", fontsize=9, color=DARK)
ax3.text(2.75, 7.4, "✅ Winner: COVID",
         ha="center", fontsize=9, color=GREEN, fontweight="bold")

# Weighted by confidence
ax3.add_patch(FancyBboxPatch((5.5, 7.0), 4.0, 2.5,
                             boxstyle="round,pad=0.1",
                             facecolor=PURPLE, edgecolor="white", lw=2, alpha=0.2))
ax3.text(7.5, 9.0, "⚖️ Weighted Avg",
         ha="center", fontsize=10, color=PURPLE, fontweight="bold")
ax3.text(7.5, 8.3, "COVID: 0.71",
         ha="center", fontsize=9, color=DARK, fontweight="bold")
ax3.text(7.5, 7.8, "Flu: 0.62",
         ha="center", fontsize=9, color=DARK)
ax3.text(7.5, 7.4, "✅ Winner: COVID",
         ha="center", fontsize=9, color=PURPLE, fontweight="bold")

# Pie chart showing consensus
center_x, center_y = 5, 3.5
radius = 1.5
angles = [270, 90]  # COVID 75%, Flu 25%
colors_pie = [BLUE, ORANGE]
labels = ["COVID\n75%", "Flu\n25%"]

for i, (start, end, color, label) in enumerate(zip(angles[:-1], angles[1:], colors_pie, labels)):
    wedge = Wedge((center_x, center_y), radius, start, end,
                  facecolor=color, edgecolor="white", lw=2)
    ax3.add_patch(wedge)
    angle_mid = (start + end) / 2
    text_x = center_x + 0.7 * radius * np.cos(np.radians(angle_mid))
    text_y = center_y + 0.7 * radius * np.sin(np.radians(angle_mid))
    ax3.text(text_x, text_y, label, ha="center", va="center",
             fontsize=8, color="white", fontweight="bold")

ax3.text(5, 0.8, "Consensus emerges from\nindependent model predictions.",
         ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 4 — Metrics & when to use ═══════════════
ax4.set_title("4 · Ensemble Impact & When to Use",
              fontsize=13, fontweight="bold", color=DARK)
ax4.set_xlim(0, 10); ax4.set_ylim(0, 10); ax4.axis("off")

# Metrics comparison
y = 8.5
ax4.text(2.5, y, "Metric", ha="center", fontsize=9,
         color=DARK, fontweight="bold")
ax4.text(5, y, "Single Model", ha="center", fontsize=9,
         color=DARK, fontweight="bold")
ax4.text(7.5, y, "Ensemble (n=4)", ha="center", fontsize=9,
         color=DARK, fontweight="bold")

metrics = [
    ("Accuracy", "87%", "93%", GREEN),
    ("Hallucination rate", "5%", "1.2%", GREEN),
    ("Cost per query", "$0.02", "$0.08", RED),
    ("Latency", "800ms", "850ms*", ORANGE),
]

y = 7.8
for metric, single, ensemble, color in metrics:
    ax4.text(2.5, y, metric, ha="center", fontsize=8, color=DARK)
    ax4.text(5, y, single, ha="center", fontsize=8, color=DARK)
    ax4.text(7.5, y, ensemble, ha="center", fontsize=8,
             color=color, fontweight="bold")
    y -= 0.6

ax4.text(7.5, y + 0.1, "*parallel calls",
         ha="center", fontsize=7, color="#555", style="italic")

# When to use
ax4.add_patch(FancyBboxPatch((1, 4.5), 8, 3.0,
                             boxstyle="round,pad=0.1",
                             facecolor=BLUE, edgecolor="white", lw=2, alpha=0.1))
ax4.text(5, 7.0, "✅ Use Ensemble When:",
         ha="center", fontsize=10, color=BLUE, fontweight="bold")

use_cases = [
    "• High-stakes decisions (medical, legal, financial)",
    "• Ambiguous queries with multiple valid interpretations",
    "• Need to reduce hallucination risk below 2%",
    "• Can afford 3-5× cost for improved accuracy",
]

y = 6.3
for case in use_cases:
    ax4.text(5, y, case, ha="center", fontsize=8, color=DARK)
    y -= 0.5

ax4.text(5, 3.5, "⚠️ Avoid When:",
         ha="center", fontsize=10, color=RED, fontweight="bold")
ax4.text(5, 2.9, "• Simple queries (e.g., 'What's 2+2?')",
         ha="center", fontsize=8, color=DARK)
ax4.text(5, 2.4, "• Tight latency/cost budgets (<100ms, <$0.01/query)",
         ha="center", fontsize=8, color=DARK)

ax4.text(5, 1.0, "Ensemble = Trading cost for reliability.\nDisagreement signals uncertainty.",
         ha="center", fontsize=9, color="#555", style="italic")

out = str(Path(__file__).resolve().parent.parent / "img" / "Ensemble Voting.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out}")
