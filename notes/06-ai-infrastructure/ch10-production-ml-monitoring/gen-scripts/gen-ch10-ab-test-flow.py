"""
gen_ch10_ab_test_flow.py
Generates: ../img/ch10_ab_test_flow.png

Shows the complete A/B testing workflow for ML models:
Traffic Split → Metrics Collection → Statistical Decision → Rollout/Rollback

Visual flow diagram with decision points and outcomes.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "img", "ch10_ab_test_flow.png")

# Ensure output directory exists
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# Create figure with dark background
fig, ax = plt.subplots(figsize=(16, 9), facecolor="#1a1a2e")
ax.set_facecolor("#1a1a2e")
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")

# Title
ax.text(8, 8.2, "A/B Testing Flow for ML Models", 
        ha="center", va="center", fontsize=26, fontweight="bold", color="white")
ax.text(8, 7.6, "From traffic split to production rollout decision",
        ha="center", va="center", fontsize=12, color="#94a3b8", style="italic")

# Define colors
TRAFFIC_COLOR = "#3b82f6"    # Blue
METRICS_COLOR = "#8b5cf6"    # Purple
DECISION_COLOR = "#f59e0b"   # Amber
SUCCESS_COLOR = "#10b981"    # Green
FAILURE_COLOR = "#ef4444"    # Red
ARROW_COLOR = "#64748b"      # Gray

# ============ STAGE 1: TRAFFIC SPLIT ============
x1 = 2
y1 = 5
ax.add_patch(FancyBboxPatch((x1 - 1, y1 - 0.6), 2, 1.4,
                            boxstyle="round,pad=0.1",
                            facecolor=TRAFFIC_COLOR, edgecolor="white", linewidth=2))
ax.text(x1, y1 + 0.3, "1. Traffic Split", ha="center", va="center",
        fontsize=14, fontweight="bold", color="white")
ax.text(x1, y1 - 0.1, "90% → Model A", ha="center", va="center",
        fontsize=10, color="white", family="monospace")
ax.text(x1, y1 - 0.35, "10% → Model B", ha="center", va="center",
        fontsize=10, color="#60a5fa", family="monospace")

# Arrow to metrics
arrow1 = FancyArrowPatch((x1 + 1, y1), (4.5, y1),
                        arrowstyle="->,head_width=0.4,head_length=0.3",
                        color=ARROW_COLOR, linewidth=2.5)
ax.add_patch(arrow1)

# ============ STAGE 2: METRICS COLLECTION ============
x2 = 6.5
y2 = 5
ax.add_patch(FancyBboxPatch((x2 - 1.5, y2 - 0.8), 3, 1.8,
                            boxstyle="round,pad=0.1",
                            facecolor=METRICS_COLOR, edgecolor="white", linewidth=2))
ax.text(x2, y2 + 0.5, "2. Collect Metrics", ha="center", va="center",
        fontsize=14, fontweight="bold", color="white")

# Metrics list
metrics = [
    ("Latency", "p99 < 200ms"),
    ("Accuracy", "0.87 vs 0.89"),
    ("Click Rate", "3.2% vs 3.8%"),
]
for i, (name, value) in enumerate(metrics):
    y_offset = 0.1 - i * 0.3
    ax.text(x2 - 0.8, y2 + y_offset, f"{name}:", ha="left", va="center",
            fontsize=9, color="#c4b5fd", fontweight="bold")
    ax.text(x2 + 0.85, y2 + y_offset, value, ha="right", va="center",
            fontsize=9, color="white", family="monospace")

# Arrow to decision
arrow2 = FancyArrowPatch((x2 + 1.5, y2), (9, y2),
                        arrowstyle="->,head_width=0.4,head_length=0.3",
                        color=ARROW_COLOR, linewidth=2.5)
ax.add_patch(arrow2)

# ============ STAGE 3: STATISTICAL DECISION ============
x3 = 11
y3 = 5
# Diamond shape for decision
diamond_points = np.array([
    [x3, y3 + 0.9],      # top
    [x3 + 1.1, y3],      # right
    [x3, y3 - 0.9],      # bottom
    [x3 - 1.1, y3],      # left
])
diamond = Polygon(diamond_points, facecolor=DECISION_COLOR, 
                 edgecolor="white", linewidth=2, alpha=0.9)
ax.add_patch(diamond)

ax.text(x3, y3 + 0.3, "3. Decision", ha="center", va="center",
        fontsize=13, fontweight="bold", color="white")
ax.text(x3, y3 - 0.1, "p < 0.05?", ha="center", va="center",
        fontsize=11, color="white", style="italic")
ax.text(x3, y3 - 0.4, "B > A?", ha="center", va="center",
        fontsize=11, color="white", style="italic")

# ============ STAGE 4A: ROLLOUT (SUCCESS) ============
x4a = 13.5
y4a = 6.5
ax.add_patch(FancyBboxPatch((x4a - 1, y4a - 0.5), 2, 1.2,
                            boxstyle="round,pad=0.1",
                            facecolor=SUCCESS_COLOR, edgecolor="white", linewidth=2))
ax.text(x4a, y4a + 0.25, "✓ Rollout", ha="center", va="center",
        fontsize=14, fontweight="bold", color="white")
ax.text(x4a, y4a - 0.15, "Model B → 100%", ha="center", va="center",
        fontsize=10, color="white", family="monospace")

# Arrow to rollout
arrow3a = FancyArrowPatch((x3 + 0.8, y3 + 0.6), (x4a - 1, y4a),
                         arrowstyle="->,head_width=0.4,head_length=0.3",
                         color=SUCCESS_COLOR, linewidth=2.5)
ax.add_patch(arrow3a)
ax.text(x3 + 1.2, y3 + 1.2, "YES", ha="center", va="center",
        fontsize=11, fontweight="bold", color=SUCCESS_COLOR)

# ============ STAGE 4B: ROLLBACK (FAILURE) ============
x4b = 13.5
y4b = 3.5
ax.add_patch(FancyBboxPatch((x4b - 1, y4b - 0.5), 2, 1.2,
                            boxstyle="round,pad=0.1",
                            facecolor=FAILURE_COLOR, edgecolor="white", linewidth=2))
ax.text(x4b, y4b + 0.25, "✗ Rollback", ha="center", va="center",
        fontsize=14, fontweight="bold", color="white")
ax.text(x4b, y4b - 0.15, "Keep Model A", ha="center", va="center",
        fontsize=10, color="white", family="monospace")

# Arrow to rollback
arrow3b = FancyArrowPatch((x3 + 0.8, y3 - 0.6), (x4b - 1, y4b),
                         arrowstyle="->,head_width=0.4,head_length=0.3",
                         color=FAILURE_COLOR, linewidth=2.5)
ax.add_patch(arrow3b)
ax.text(x3 + 1.2, y3 - 1.2, "NO", ha="center", va="center",
        fontsize=11, fontweight="bold", color=FAILURE_COLOR)

# ============ BOTTOM NOTES ============
notes_y = 1.5
ax.text(8, notes_y, "Key Principles", ha="center", va="top",
        fontsize=14, fontweight="bold", color="white")

notes = [
    "• Start small: 5-10% traffic to new model minimizes risk",
    "• Wait for statistical significance (typically 1-2 weeks)",
    "• Monitor business metrics, not just ML metrics (CTR, revenue, retention)",
    "• Automate rollback if latency spikes or error rate increases",
]
for i, note in enumerate(notes):
    ax.text(8, notes_y - 0.5 - i * 0.3, note, ha="center", va="center",
            fontsize=10, color="#94a3b8")

# Save figure
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"✓ Saved: {OUT_PATH}")
