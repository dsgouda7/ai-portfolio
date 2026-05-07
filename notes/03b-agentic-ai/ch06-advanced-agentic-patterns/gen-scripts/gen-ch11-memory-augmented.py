from pathlib import Path
"""Generate Memory Augmented.png — Episodic memory retrieval.

Shows context retrieval from memory store and highlights relevant past interactions.
4-panel xkcd-style visualization of memory-augmented agent architecture.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

plt.xkcd(scale=0.8, length=100, randomness=2)

fig = plt.figure(figsize=(18, 11), facecolor="white")
fig.suptitle("Memory-Augmented Agents — Episodic Recall", 
             fontsize=22, fontweight="bold", y=0.98)

gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.30,
                      left=0.06, right=0.97, top=0.91, bottom=0.06)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

BLUE, ORANGE, GREEN, RED, PURPLE, DARK, GREY = (
    "#2E86C1", "#E67E22", "#27AE60", "#E74C3C", "#8E44AD", "#2C3E50", "#BDC3C7")

# ═══════════════════════ PANEL 1 — Without memory (context loss) ═══════
ax1.set_title("1 · Without Memory — Context Loss",
              fontsize=13, fontweight="bold", color=DARK)
ax1.set_xlim(0, 10); ax1.set_ylim(0, 10); ax1.axis("off")

# Conversation flow
conversations = [
    ("Week 1: 'I'm allergic to nuts'", BLUE, 8.5),
    ("Week 2: 'Recommend snack options'", BLUE, 7.0),
]

for conv, color, y in conversations:
    ax1.add_patch(FancyBboxPatch((0.5, y - 0.4), 6, 0.8,
                                 boxstyle="round,pad=0.05",
                                 facecolor=color, edgecolor="white", lw=2, alpha=0.2))
    ax1.text(3.5, y, conv, ha="center", va="center",
             fontsize=9, color=DARK)

# Agent response (without memory)
ax1.add_patch(FancyBboxPatch((1, 4.5), 8, 2.0,
                             boxstyle="round,pad=0.1",
                             facecolor=RED, edgecolor="white", lw=2, alpha=0.2))
ax1.text(5, 6.0, "❌ Agent Response:",
         ha="center", fontsize=10, color=RED, fontweight="bold")
ax1.text(5, 5.4, '"Try our trail mix with almonds and cashews!"',
         ha="center", fontsize=9, color=DARK, style="italic")
ax1.text(5, 4.8, "⚠️ Recommends nuts despite prior allergy disclosure",
         ha="center", fontsize=8, color=RED)

# Problem
ax1.add_patch(FancyBboxPatch((1.5, 2.5), 7, 1.5,
                             boxstyle="round,pad=0.1",
                             facecolor=ORANGE, edgecolor="white", lw=2, alpha=0.2))
ax1.text(5, 3.6, "⚠️ Problem: No long-term memory",
         ha="center", fontsize=10, color=ORANGE, fontweight="bold")
ax1.text(5, 3.0, "Past context outside token window is lost.",
         ha="center", fontsize=8, color=DARK)

ax1.text(5, 1.0, "Standard LLMs forget conversations\nbeyond their context window.",
         ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 2 — Memory storage ══════════════════════
ax2.set_title("2 · Episodic Memory Store",
              fontsize=13, fontweight="bold", color=DARK)
ax2.set_xlim(0, 10); ax2.set_ylim(0, 10); ax2.axis("off")

# Memory database
ax2.add_patch(FancyBboxPatch((1, 6.5), 8, 3.0,
                             boxstyle="round,pad=0.1",
                             facecolor=PURPLE, edgecolor="white", lw=2, alpha=0.2))
ax2.text(5, 9.0, "🗄️ Memory Store (Vector DB)",
         ha="center", fontsize=11, color=PURPLE, fontweight="bold")

# Memory entries
memory_entries = [
    ("2024-04-01: User allergic to nuts", 8.2),
    ("2024-04-05: Prefers vegan options", 7.6),
    ("2024-04-12: Interested in protein snacks", 7.0),
]

for entry, y in memory_entries:
    ax2.add_patch(Rectangle((1.5, y - 0.2), 7, 0.4,
                            facecolor=GREY, edgecolor="white", lw=1, alpha=0.2))
    ax2.text(5, y, entry, ha="center", va="center",
             fontsize=8, color=DARK)

# Storage process
ax2.add_patch(FancyBboxPatch((1.5, 4.0), 7, 2.0,
                             boxstyle="round,pad=0.1",
                             facecolor=BLUE, edgecolor="white", lw=2, alpha=0.2))
ax2.text(5, 5.5, "💾 Storage Process:",
         ha="center", fontsize=10, color=BLUE, fontweight="bold")
ax2.text(5, 4.9, "1. Embed conversation turns",
         ha="center", fontsize=8, color=DARK)
ax2.text(5, 4.5, "2. Store with metadata (timestamp, user_id, topic)",
         ha="center", fontsize=8, color=DARK)
ax2.text(5, 4.1, "3. Index for semantic search",
         ha="center", fontsize=8, color=DARK)

ax2.text(5, 2.5, "📊 Typical storage:", ha="center", fontsize=9,
         color=DARK, fontweight="bold")
ax2.text(5, 2.0, "• 100K memories ≈ 50MB (embeddings only)",
         ha="center", fontsize=8, color=DARK)
ax2.text(5, 1.5, "• Retrieval: <50ms for top-k=10",
         ha="center", fontsize=8, color=DARK)

ax2.text(5, 0.5, "Long-term storage enables\npersonalization across sessions.",
         ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 3 — Memory retrieval ════════════════════
ax3.set_title("3 · Retrieval & Context Injection",
              fontsize=13, fontweight="bold", color=DARK)
ax3.set_xlim(0, 10); ax3.set_ylim(0, 10); ax3.axis("off")

# Query
ax3.add_patch(FancyBboxPatch((0.5, 8.5), 6, 1.0,
                             boxstyle="round,pad=0.05",
                             facecolor=BLUE, edgecolor="white", lw=2, alpha=0.2))
ax3.text(3.5, 9.0, "Query: 'Recommend snack options'",
         ha="center", va="center", fontsize=9, color=DARK)

# Arrow to retrieval
arrow1 = FancyArrowPatch((3.5, 8.3), (3.5, 7.5),
                         arrowstyle="->,head_width=0.3,head_length=0.2",
                         color=DARK, lw=2)
ax3.add_artist(arrow1)

# Retrieval process
ax3.add_patch(FancyBboxPatch((0.5, 6.0), 6, 1.5,
                             boxstyle="round,pad=0.1",
                             facecolor=PURPLE, edgecolor="white", lw=2, alpha=0.2))
ax3.text(3.5, 7.1, "🔍 Semantic Search:",
         ha="center", fontsize=10, color=PURPLE, fontweight="bold")
ax3.text(3.5, 6.5, "Retrieve top-3 relevant memories",
         ha="center", fontsize=8, color=DARK)

# Retrieved memories (highlighted)
ax3.add_patch(FancyBboxPatch((0.5, 4.0), 6, 1.8,
                             boxstyle="round,pad=0.1",
                             facecolor=GREEN, edgecolor="white", lw=2, alpha=0.2))
ax3.text(3.5, 5.5, "✅ Retrieved Context:",
         ha="center", fontsize=10, color=GREEN, fontweight="bold")
ax3.text(3.5, 4.9, "1. ⭐ Allergic to nuts (similarity: 0.89)",
         ha="center", fontsize=8, color=DARK, fontweight="bold")
ax3.text(3.5, 4.5, "2. Prefers vegan (similarity: 0.72)",
         ha="center", fontsize=8, color=DARK)
ax3.text(3.5, 4.1, "3. Protein snacks (similarity: 0.68)",
         ha="center", fontsize=8, color=DARK)

# Augmented response
arrow2 = FancyArrowPatch((3.5, 3.8), (3.5, 3.0),
                         arrowstyle="->,head_width=0.3,head_length=0.2",
                         color=DARK, lw=2)
ax3.add_artist(arrow2)

ax3.add_patch(FancyBboxPatch((0.5, 1.0), 6, 2.0,
                             boxstyle="round,pad=0.1",
                             facecolor=GREEN, edgecolor="white", lw=2, alpha=0.3))
ax3.text(3.5, 2.6, "✅ Memory-Aware Response:",
         ha="center", fontsize=10, color=GREEN, fontweight="bold")
ax3.text(3.5, 2.0, '"Since you\'re allergic to nuts and prefer',
         ha="center", fontsize=8, color=DARK, style="italic")
ax3.text(3.5, 1.6, 'vegan options, try: hummus cups, protein',
         ha="center", fontsize=8, color=DARK, style="italic")
ax3.text(3.5, 1.2, 'bars (pea-based), or roasted chickpeas."',
         ha="center", fontsize=8, color=DARK, style="italic")

# Flow on right side
ax3.text(8.5, 9.0, "Flow:", ha="center", fontsize=9,
         color=DARK, fontweight="bold")
ax3.text(8.5, 8.4, "1️⃣ Query", ha="center", fontsize=8, color=BLUE)
ax3.text(8.5, 7.5, "⬇", ha="center", fontsize=10, color=DARK)
ax3.text(8.5, 6.9, "2️⃣ Embed", ha="center", fontsize=8, color=PURPLE)
ax3.text(8.5, 6.0, "⬇", ha="center", fontsize=10, color=DARK)
ax3.text(8.5, 5.4, "3️⃣ Search", ha="center", fontsize=8, color=PURPLE)
ax3.text(8.5, 4.5, "⬇", ha="center", fontsize=10, color=DARK)
ax3.text(8.5, 3.9, "4️⃣ Inject", ha="center", fontsize=8, color=GREEN)
ax3.text(8.5, 3.0, "⬇", ha="center", fontsize=10, color=DARK)
ax3.text(8.5, 2.4, "5️⃣ Generate", ha="center", fontsize=8, color=GREEN)

ax3.text(5, 0.2, "Relevant memories augment prompt context.",
         ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 4 — Impact & architecture ═══════════════
ax4.set_title("4 · Memory-Augmented Architecture & Impact",
              fontsize=13, fontweight="bold", color=DARK)
ax4.set_xlim(0, 10); ax4.set_ylim(0, 10); ax4.axis("off")

# Architecture diagram
components = [
    ("LLM", BLUE, 2, 7.5, 1.5, 1.0),
    ("Memory\nStore", PURPLE, 6, 7.5, 1.8, 1.0),
]

for label, color, x, y, w, h in components:
    ax4.add_patch(FancyBboxPatch((x, y), w, h,
                                 boxstyle="round,pad=0.1",
                                 facecolor=color, edgecolor="white", lw=2, alpha=0.3))
    ax4.text(x + w/2, y + h/2, label, ha="center", va="center",
             fontsize=9, color="white", fontweight="bold")

# Bidirectional arrows
arrow_read = FancyArrowPatch((3.6, 8.2), (5.9, 8.2),
                             arrowstyle="->,head_width=0.25,head_length=0.15",
                             color=GREEN, lw=2)
ax4.add_artist(arrow_read)
ax4.text(4.75, 8.5, "Read", ha="center", fontsize=7, color=GREEN)

arrow_write = FancyArrowPatch((5.9, 7.8), (3.6, 7.8),
                              arrowstyle="->,head_width=0.25,head_length=0.15",
                              color=ORANGE, lw=2)
ax4.add_artist(arrow_write)
ax4.text(4.75, 7.5, "Write", ha="center", fontsize=7, color=ORANGE)

# Metrics comparison
y = 6.0
ax4.text(2, y, "Metric", ha="center", fontsize=9,
         color=DARK, fontweight="bold")
ax4.text(5, y, "No Memory", ha="center", fontsize=9,
         color=DARK, fontweight="bold")
ax4.text(8, y, "With Memory", ha="center", fontsize=9,
         color=DARK, fontweight="bold")

metrics = [
    ("Personalization", "0%", "85%", GREEN),
    ("Context retention", "8K tokens", "Unlimited*", GREEN),
    ("User satisfaction", "3.9/5", "4.6/5", GREEN),
    ("Latency overhead", "0ms", "+40ms", RED),
]

y = 5.4
for metric, without, with_mem, color in metrics:
    ax4.text(2, y, metric, ha="center", fontsize=8, color=DARK)
    ax4.text(5, y, without, ha="center", fontsize=8, color=DARK)
    ax4.text(8, y, with_mem, ha="center", fontsize=8,
             color=color, fontweight="bold")
    y -= 0.5

ax4.text(8, y + 0.1, "*storage limit",
         ha="center", fontsize=7, color="#555", style="italic")

# Use cases
ax4.add_patch(FancyBboxPatch((0.5, 1.5), 9, 2.0,
                             boxstyle="round,pad=0.1",
                             facecolor=BLUE, edgecolor="white", lw=2, alpha=0.1))
ax4.text(5, 3.1, "✅ Use Cases:",
         ha="center", fontsize=10, color=BLUE, fontweight="bold")

use_cases = [
    "• Personal assistants (remember preferences, habits, history)",
    "• Customer support (recall past issues, solutions)",
    "• Healthcare (track symptoms, medications, allergies)",
]

y = 2.5
for case in use_cases:
    ax4.text(5, y, case, ha="center", fontsize=8, color=DARK)
    y -= 0.4

ax4.text(5, 0.5, "Memory = Personalization at scale.\nRAG for conversations, not just docs.",
         ha="center", fontsize=9, color="#555", style="italic")

out = str(Path(__file__).resolve().parent.parent / "img" / "Memory Augmented.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out}")
