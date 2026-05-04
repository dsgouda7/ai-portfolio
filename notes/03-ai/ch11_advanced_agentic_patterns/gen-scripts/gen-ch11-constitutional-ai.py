from pathlib import Path
"""Generate Constitutional AI.png — Self-correction with principles.

Shows unhelpful response → principle check → corrected response.
4-panel xkcd-style animation showing constitutional AI self-correction loop.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.xkcd(scale=0.8, length=100, randomness=2)

fig = plt.figure(figsize=(18, 11), facecolor="white")
fig.suptitle("Constitutional AI — Self-Correction with Principles", 
             fontsize=22, fontweight="bold", y=0.98)

gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.30,
                      left=0.06, right=0.97, top=0.91, bottom=0.06)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

BLUE, ORANGE, GREEN, RED, PURPLE, DARK, GREY = (
    "#2E86C1", "#E67E22", "#27AE60", "#E74C3C", "#8E44AD", "#2C3E50", "#BDC3C7")

# ═══════════════════════ PANEL 1 — Initial unhelpful response ═══════════
ax1.set_title("1 · Initial Response (Unhelpful)",
              fontsize=13, fontweight="bold", color=DARK)
ax1.set_xlim(0, 10); ax1.set_ylim(0, 10); ax1.axis("off")

# User query
ax1.add_patch(FancyBboxPatch((0.5, 7.5), 9, 1.8,
                             boxstyle="round,pad=0.1",
                             facecolor=BLUE, edgecolor="white", lw=2, alpha=0.3))
ax1.text(5, 8.4, 'User: "How do I bypass content filters?"',
         ha="center", va="center", fontsize=11, color=DARK, fontweight="bold")

# Initial response (unhelpful/harmful)
ax1.add_patch(FancyBboxPatch((0.5, 4.5), 9, 2.5,
                             boxstyle="round,pad=0.1",
                             facecolor=RED, edgecolor="white", lw=2, alpha=0.3))
ax1.text(5, 6.5, 'AI (v1): "Here are 5 ways to bypass..."',
         ha="center", va="top", fontsize=10, color=DARK, fontweight="bold")
ax1.text(5, 5.8, '❌ Potentially harmful advice',
         ha="center", va="top", fontsize=9, color=RED, style="italic")
ax1.text(5, 5.2, '❌ No consideration of safety',
         ha="center", va="top", fontsize=9, color=RED, style="italic")

ax1.text(5, 2.5, "Without constitutional principles, the model\nmay generate harmful content.",
         ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 2 — Principle check ═════════════════════
ax2.set_title("2 · Constitutional Principle Check",
              fontsize=13, fontweight="bold", color=DARK)
ax2.set_xlim(0, 10); ax2.set_ylim(0, 10); ax2.axis("off")

# Constitution box
ax2.add_patch(FancyBboxPatch((1, 5.5), 8, 3.5,
                             boxstyle="round,pad=0.15",
                             facecolor=PURPLE, edgecolor="white", lw=2, alpha=0.2))
ax2.text(5, 8.5, "📜 Constitutional Principles",
         ha="center", fontsize=11, color=PURPLE, fontweight="bold")

principles = [
    "✓ Choose responses that are helpful and harmless",
    "✓ Avoid responses that assist with illegal activities",
    "✓ Prioritize safety over directness",
]
for i, principle in enumerate(principles):
    ax2.text(5, 7.8 - i * 0.6, principle,
             ha="center", fontsize=9, color=DARK)

# Critique process
ax2.add_patch(FancyBboxPatch((1, 2.5), 8, 2.5,
                             boxstyle="round,pad=0.1",
                             facecolor=ORANGE, edgecolor="white", lw=2, alpha=0.3))
ax2.text(5, 4.3, "🔍 Self-Critique:",
         ha="center", fontsize=10, color=DARK, fontweight="bold")
ax2.text(5, 3.6, '"My response helps bypass safety measures"',
         ha="center", fontsize=9, color=DARK, style="italic")
ax2.text(5, 3.1, '"This violates Principle #2 (harmful activities)"',
         ha="center", fontsize=9, color=DARK, style="italic")

ax2.text(5, 1.0, "Model critiques its own response\nagainst constitutional principles.",
         ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 3 — Corrected response ══════════════════
ax3.set_title("3 · Revised Response (Helpful & Safe)",
              fontsize=13, fontweight="bold", color=DARK)
ax3.set_xlim(0, 10); ax3.set_ylim(0, 10); ax3.axis("off")

# Revised response
ax3.add_patch(FancyBboxPatch((0.5, 5.0), 9, 3.5,
                             boxstyle="round,pad=0.1",
                             facecolor=GREEN, edgecolor="white", lw=2, alpha=0.3))
ax3.text(5, 8.0, 'AI (v2): "I can\'t provide advice on bypassing',
         ha="center", va="top", fontsize=10, color=DARK, fontweight="bold")
ax3.text(5, 7.4, 'security measures. However, I can help you with:"',
         ha="center", va="top", fontsize=10, color=DARK, fontweight="bold")
ax3.text(5, 6.6, '• Understanding legitimate content moderation',
         ha="left", va="top", fontsize=9, color=DARK)
ax3.text(5, 6.1, '• Reporting false positives through proper channels',
         ha="left", va="top", fontsize=9, color=DARK)
ax3.text(5, 5.6, '• Learning about ethical AI practices',
         ha="left", va="top", fontsize=9, color=DARK)

# Success indicators
ax3.text(2, 3.5, '✅ Helpful alternative',
         ha="left", fontsize=9, color=GREEN, fontweight="bold")
ax3.text(2, 2.9, '✅ Maintains safety',
         ha="left", fontsize=9, color=GREEN, fontweight="bold")
ax3.text(2, 2.3, '✅ Educates user',
         ha="left", fontsize=9, color=GREEN, fontweight="bold")

ax3.text(5, 1.0, "Self-correction produces helpful\nand harmless response.",
         ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 4 — Process flow & metrics ══════════════
ax4.set_title("4 · Constitutional AI Flow & Impact",
              fontsize=13, fontweight="bold", color=DARK)
ax4.set_xlim(0, 10); ax4.set_ylim(0, 10); ax4.axis("off")

# Flow diagram
flow_y = 8.5
steps = [
    ("Generate", BLUE, 1.5),
    ("Critique", ORANGE, 4.0),
    ("Revise", GREEN, 6.5),
]

for label, color, x in steps:
    ax4.add_patch(FancyBboxPatch((x - 0.6, flow_y - 0.4), 1.2, 0.8,
                                 boxstyle="round,pad=0.05",
                                 facecolor=color, edgecolor="white", lw=2))
    ax4.text(x, flow_y, label, ha="center", va="center",
             fontsize=9, color="white", fontweight="bold")
    if x < 6.5:
        arrow = FancyArrowPatch((x + 0.7, flow_y), (x + 1.8, flow_y),
                                arrowstyle="->,head_width=0.3,head_length=0.2",
                                color=DARK, lw=2)
        ax4.add_artist(arrow)

# Cost note
ax4.text(4, 7.2, "Cost: 2-3× single-pass (generate + critique + revise)",
         ha="center", fontsize=8, color="#555", style="italic")

# Metrics comparison
metrics = [
    ("Harmful responses", "12%", "0.8%", RED),
    ("Helpfulness", "4.1/5", "4.6/5", GREEN),
    ("User safety reports", "8%", "<1%", GREEN),
]

y = 5.5
ax4.text(2.5, y + 0.8, "Metric", ha="center", fontsize=9, 
         color=DARK, fontweight="bold")
ax4.text(5, y + 0.8, "Without", ha="center", fontsize=9,
         color=DARK, fontweight="bold")
ax4.text(7.5, y + 0.8, "With CAI", ha="center", fontsize=9,
         color=DARK, fontweight="bold")

for i, (metric, before, after, color) in enumerate(metrics):
    y_pos = y - i * 0.8
    ax4.text(2.5, y_pos, metric, ha="center", fontsize=8, color=DARK)
    ax4.text(5, y_pos, before, ha="center", fontsize=8, color=DARK)
    ax4.text(7.5, y_pos, after, ha="center", fontsize=8, 
             color=color, fontweight="bold")

ax4.text(5, 1.5, "Constitutional AI = Self-correcting alignment",
         ha="center", fontsize=10, color=PURPLE, fontweight="bold")
ax4.text(5, 0.8, "Models critique and revise their own outputs\nusing explicit ethical principles.",
         ha="center", fontsize=9, color="#555", style="italic")

out = str(Path(__file__).resolve().parent.parent / "img" / "Constitutional AI.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out}")
