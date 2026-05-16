"""
gen_ch10_gradual_rollout.py
Generates: ../img/ch10_gradual_rollout.png

Shows the timeline of a gradual model rollout:
Timeline visualization showing traffic migration: 10% → 25% → 50% → 100%
with monitoring checkpoints and decision gates at each stage.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "img", "ch10_gradual_rollout.png")

# Ensure output directory exists
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# Create figure with dark background
fig, ax = plt.subplots(figsize=(16, 9), facecolor="#1a1a2e")
ax.set_facecolor("#1a1a2e")
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")

# Title
ax.text(8, 8.2, "Gradual Model Rollout Timeline", 
        ha="center", va="center", fontsize=26, fontweight="bold", color="white")
ax.text(8, 7.6, "Progressive traffic migration with monitoring gates at each stage",
        ha="center", va="center", fontsize=12, color="#94a3b8", style="italic")

# Define colors
STAGE_COLOR = "#3b82f6"      # Blue
MONITOR_COLOR = "#8b5cf6"    # Purple
SUCCESS_COLOR = "#10b981"    # Green
TIMELINE_COLOR = "#64748b"   # Gray

# ============ TIMELINE BASE ============
timeline_y = 4.5
ax.plot([1.5, 14.5], [timeline_y, timeline_y], color=TIMELINE_COLOR, 
        linewidth=3, zorder=1)

# ============ STAGE DEFINITIONS ============
stages = [
    {
        "x": 2.5,
        "traffic": "10%",
        "day": "Day 1",
        "duration": "Monitor 3 days",
        "checks": ["Latency OK?", "Error rate OK?", "Drift OK?"]
    },
    {
        "x": 6,
        "traffic": "25%",
        "day": "Day 4",
        "duration": "Monitor 3 days",
        "checks": ["Business metrics?", "User feedback?", "System load?"]
    },
    {
        "x": 9.5,
        "traffic": "50%",
        "day": "Day 7",
        "duration": "Monitor 4 days",
        "checks": ["Revenue impact?", "Churn rate?", "A/B test sig?"]
    },
    {
        "x": 13.5,
        "traffic": "100%",
        "day": "Day 11",
        "duration": "Full rollout",
        "checks": ["Continue monitoring", "Establish baseline", "Next iteration"]
    },
]

# ============ DRAW EACH STAGE ============
for i, stage in enumerate(stages):
    x = stage["x"]
    
    # Timeline marker (circle)
    marker = Circle((x, timeline_y), 0.25, facecolor=STAGE_COLOR, 
                   edgecolor="white", linewidth=2, zorder=2)
    ax.add_patch(marker)
    
    # Stage box above timeline
    box_y = timeline_y + 1.5
    ax.add_patch(FancyBboxPatch((x - 0.9, box_y - 0.4), 1.8, 1.3,
                                boxstyle="round,pad=0.1",
                                facecolor=STAGE_COLOR, edgecolor="white", 
                                linewidth=2, alpha=0.9))
    
    ax.text(x, box_y + 0.5, stage["traffic"], ha="center", va="center",
            fontsize=20, fontweight="bold", color="white")
    ax.text(x, box_y + 0.15, "Traffic", ha="center", va="center",
            fontsize=10, color="#dbeafe")
    ax.text(x, box_y - 0.2, stage["day"], ha="center", va="center",
            fontsize=10, color="white", fontweight="bold", family="monospace")
    
    # Monitoring checklist below timeline
    checklist_y = timeline_y - 1.2
    ax.text(x, checklist_y, stage["duration"], ha="center", va="center",
            fontsize=10, fontweight="bold", color=MONITOR_COLOR)
    
    for j, check in enumerate(stage["checks"]):
        check_y = checklist_y - 0.4 - j * 0.3
        ax.text(x - 0.15, check_y, "✓", ha="center", va="center",
                fontsize=10, color=SUCCESS_COLOR, fontweight="bold")
        ax.text(x + 0.05, check_y, check, ha="left", va="center",
                fontsize=8, color="#94a3b8")
    
    # Arrow to next stage (except last)
    if i < len(stages) - 1:
        next_x = stages[i + 1]["x"]
        arrow = FancyArrowPatch((x + 0.3, timeline_y), (next_x - 0.3, timeline_y),
                               arrowstyle="->,head_width=0.3,head_length=0.2",
                               color=TIMELINE_COLOR, linewidth=2, zorder=1)
        ax.add_patch(arrow)
        
        # "Pass" or "Go" label on arrow
        mid_x = (x + next_x) / 2
        ax.text(mid_x, timeline_y + 0.3, "GO", ha="center", va="center",
                fontsize=9, fontweight="bold", color=SUCCESS_COLOR,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=SUCCESS_COLOR, 
                         alpha=0.2, edgecolor=SUCCESS_COLOR))

# ============ ROLLBACK WARNING ============
warning_y = 0.8
ax.add_patch(FancyBboxPatch((3, warning_y - 0.3), 10, 0.7,
                            boxstyle="round,pad=0.1",
                            facecolor="#ef4444", edgecolor="white", 
                            linewidth=2, alpha=0.2))

ax.text(8, warning_y + 0.15, "⚠ ROLLBACK TRIGGERS", ha="center", va="center",
        fontsize=11, fontweight="bold", color="#ef4444")
ax.text(8, warning_y - 0.1, "At ANY stage: Latency spike (>300ms) | Error rate jump (>2%) | Accuracy drop (>3%) → Immediate rollback to previous stage",
        ha="center", va="center", fontsize=9, color="#fca5a5")

# ============ TOP TIMELINE LABELS ============
ax.text(2.5, timeline_y + 3.2, "Initial Test", ha="center", va="center",
        fontsize=11, fontweight="bold", color="#60a5fa", style="italic")
ax.text(7.75, timeline_y + 3.2, "Validation Phase", ha="center", va="center",
        fontsize=11, fontweight="bold", color="#a78bfa", style="italic")
ax.text(13.5, timeline_y + 3.2, "Full Production", ha="center", va="center",
        fontsize=11, fontweight="bold", color="#34d399", style="italic")

# Bracket for validation phase
bracket_y = timeline_y + 2.9
ax.plot([5.5, 5.5], [bracket_y, bracket_y + 0.15], color="#a78bfa", linewidth=2)
ax.plot([5.5, 10], [bracket_y + 0.15, bracket_y + 0.15], color="#a78bfa", linewidth=2)
ax.plot([10, 10], [bracket_y + 0.15, bracket_y], color="#a78bfa", linewidth=2)

# Save figure
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"✓ Saved: {OUT_PATH}")
