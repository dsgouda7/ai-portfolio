from pathlib import Path
"""Generate Plan and Execute.png — Decompose → Plan → Execute → Replan cycle.

Shows plan failure and replanning with adaptive recovery.
4-panel xkcd-style visualization of dynamic planning workflow.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

plt.xkcd(scale=0.8, length=100, randomness=2)

fig = plt.figure(figsize=(18, 11), facecolor="white")
fig.suptitle("Plan-and-Execute — Adaptive Task Decomposition", 
             fontsize=22, fontweight="bold", y=0.98)

gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.30,
                      left=0.06, right=0.97, top=0.91, bottom=0.06)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

BLUE, ORANGE, GREEN, RED, PURPLE, DARK, GREY = (
    "#2E86C1", "#E67E22", "#27AE60", "#E74C3C", "#8E44AD", "#2C3E50", "#BDC3C7")

# ═══════════════════════ PANEL 1 — Complex task & initial plan ═════════
ax1.set_title("1 · Complex Task → Initial Plan",
              fontsize=13, fontweight="bold", color=DARK)
ax1.set_xlim(0, 10); ax1.set_ylim(0, 10); ax1.axis("off")

# Task
ax1.add_patch(FancyBboxPatch((0.5, 8.0), 9, 1.5,
                             boxstyle="round,pad=0.1",
                             facecolor=BLUE, edgecolor="white", lw=2, alpha=0.2))
ax1.text(5, 8.75, '📋 Task: "Organize company retreat for 50 people"',
         ha="center", va="center", fontsize=10, color=DARK, fontweight="bold")

# Initial plan (before execution)
plan_steps = [
    ("1. Find venue (capacity 50+)", BLUE, 6.5),
    ("2. Book catering", GREEN, 5.5),
    ("3. Send invitations", GREEN, 4.5),
    ("4. Arrange transportation", GREEN, 3.5),
]

ax1.text(5, 7.2, "🗺️ Initial Plan:", ha="center", fontsize=10,
         color=DARK, fontweight="bold")

for step, color, y in plan_steps:
    ax1.add_patch(FancyBboxPatch((1, y - 0.3), 8, 0.6,
                                 boxstyle="round,pad=0.05",
                                 facecolor=color, edgecolor="white", lw=1, alpha=0.3))
    ax1.text(5, y, step, ha="center", va="center",
             fontsize=9, color=DARK)

ax1.text(5, 2.0, "Planner decomposes complex task\ninto sequential subtasks.",
         ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 2 — Execution & failure ═════════════════
ax2.set_title("2 · Execute Steps → Encounter Failure",
              fontsize=13, fontweight="bold", color=DARK)
ax2.set_xlim(0, 10); ax2.set_ylim(0, 10); ax2.axis("off")

# Execution timeline
exec_steps = [
    ("Step 1: Find venue", "✅ Found 3 options", GREEN, 8.5),
    ("Step 2: Book catering", "❌ FAILED", RED, 7.0),
]

for step, result, color, y in exec_steps:
    ax2.add_patch(FancyBboxPatch((0.5, y - 0.4), 4, 0.8,
                                 boxstyle="round,pad=0.05",
                                 facecolor=color, edgecolor="white", lw=2, alpha=0.3))
    ax2.text(2.5, y, step, ha="center", va="center",
             fontsize=9, color=DARK, fontweight="bold")
    ax2.text(7, y, result, ha="center", va="center",
             fontsize=9, color=color, fontweight="bold")

# Failure details
ax2.add_patch(FancyBboxPatch((1, 4.5), 8, 2.0,
                             boxstyle="round,pad=0.1",
                             facecolor=RED, edgecolor="white", lw=2, alpha=0.2))
ax2.text(5, 6.0, "🚨 Execution Error:",
         ha="center", fontsize=10, color=RED, fontweight="bold")
ax2.text(5, 5.4, '"All catering services fully booked for that date."',
         ha="center", fontsize=9, color=DARK, style="italic")
ax2.text(5, 4.8, "Original plan cannot proceed as written.",
         ha="center", fontsize=9, color=DARK)

# Decision point
ax2.add_patch(FancyBboxPatch((2, 2.5), 6, 1.5,
                             boxstyle="round,pad=0.1",
                             facecolor=ORANGE, edgecolor="white", lw=2, alpha=0.2))
ax2.text(5, 3.5, "🤔 Need to replan:",
         ha="center", fontsize=10, color=ORANGE, fontweight="bold")
ax2.text(5, 2.9, "Rigid plan would fail here.\nAdaptive agent replans.",
         ha="center", fontsize=8, color=DARK)

ax2.text(5, 1.0, "Failures trigger replanning,\nnot cascading errors.",
         ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 3 — Replanning & adaptation ═════════════
ax3.set_title("3 · Replan → Adapt Strategy",
              fontsize=13, fontweight="bold", color=DARK)
ax3.set_xlim(0, 10); ax3.set_ylim(0, 10); ax3.axis("off")

# Replanning process
ax3.add_patch(FancyBboxPatch((0.5, 7.5), 9, 2.0,
                             boxstyle="round,pad=0.1",
                             facecolor=PURPLE, edgecolor="white", lw=2, alpha=0.2))
ax3.text(5, 9.0, "♻️ Replanner analyzes failure:",
         ha="center", fontsize=10, color=PURPLE, fontweight="bold")
ax3.text(5, 8.4, "• Root cause: Date conflict",
         ha="center", fontsize=9, color=DARK)
ax3.text(5, 7.9, "• Alternative: Change date OR find alternate catering",
         ha="center", fontsize=9, color=DARK)

# Revised plan
revised_steps = [
    ("1. ✅ Venue found (keep)", GREEN, 6.0),
    ("2a. Check venue date flexibility", ORANGE, 5.2),
    ("2b. If flexible → Move to next week", BLUE, 4.4),
    ("2c. Else → Find food trucks / potluck", BLUE, 3.6),
    ("3. Update invitations (new date)", GREEN, 2.8),
    ("4. Arrange transportation", GREEN, 2.0),
]

ax3.text(5, 6.8, "📝 Revised Plan:", ha="center", fontsize=10,
         color=DARK, fontweight="bold")

for i, (step, color, y) in enumerate(revised_steps):
    ax3.add_patch(FancyBboxPatch((1, y - 0.25), 8, 0.5,
                                 boxstyle="round,pad=0.03",
                                 facecolor=color, edgecolor="white", lw=1, alpha=0.3))
    ax3.text(5, y, step, ha="center", va="center",
             fontsize=8, color=DARK)

ax3.text(5, 0.8, "Replanner inserts contingency branches\nand recovery steps.",
         ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 4 — Pattern & metrics ═══════════════════
ax4.set_title("4 · Plan-Execute-Replan Loop & Impact",
              fontsize=13, fontweight="bold", color=DARK)
ax4.set_xlim(0, 10); ax4.set_ylim(0, 10); ax4.axis("off")

# Cycle diagram
center_x, center_y = 5, 7.5
radius = 1.2

cycle_steps = [
    ("Plan", BLUE, 90),
    ("Execute", GREEN, 210),
    ("Monitor", ORANGE, 330),
]

for label, color, angle in cycle_steps:
    angle_rad = np.radians(angle)
    x = center_x + radius * np.cos(angle_rad)
    y = center_y + radius * np.sin(angle_rad)
    
    circle = Circle((x, y), 0.4, facecolor=color, edgecolor="white", lw=2)
    ax4.add_artist(circle)
    ax4.text(x, y, label, ha="center", va="center",
             fontsize=8, color="white", fontweight="bold")
    
    # Draw arrows
    next_angle = angle - 120
    next_angle_rad = np.radians(next_angle)
    x_next = center_x + radius * np.cos(next_angle_rad)
    y_next = center_y + radius * np.sin(next_angle_rad)
    
    arrow = FancyArrowPatch(
        (x - 0.25*np.cos(angle_rad), y - 0.25*np.sin(angle_rad)),
        (x_next + 0.25*np.cos(next_angle_rad), y_next + 0.25*np.sin(next_angle_rad)),
        arrowstyle="->,head_width=0.3,head_length=0.2",
        color=DARK, lw=2, connectionstyle="arc3,rad=0.3"
    )
    ax4.add_artist(arrow)

ax4.text(5, 5.5, "Continuous adaptation loop",
         ha="center", fontsize=9, color=PURPLE, fontweight="bold")

# Metrics
metrics_y = 4.5
ax4.text(2.5, metrics_y, "Metric", ha="center", fontsize=9,
         color=DARK, fontweight="bold")
ax4.text(5, metrics_y, "Rigid Plan", ha="center", fontsize=9,
         color=DARK, fontweight="bold")
ax4.text=7.5, metrics_y, "Adaptive", ha="center", fontsize=9,
         color=DARK, fontweight="bold")

comparisons = [
    ("Task success rate", "45%", "87%", GREEN),
    ("Recovery from failure", "10%", "78%", GREEN),
    ("Avg. replanning rounds", "0", "1.8", ORANGE),
    ("Token cost", "$0.15", "$0.35", RED),
]

y = 3.9
for metric, rigid, adaptive, color in comparisons:
    ax4.text(2.5, y, metric, ha="center", fontsize=8, color=DARK)
    ax4.text(5, y, rigid, ha="center", fontsize=8, color=DARK)
    ax4.text(7.5, y, adaptive, ha="center", fontsize=8,
             color=color, fontweight="bold")
    y -= 0.5

ax4.text(5, 1.5, "✅ Use When: Multi-step tasks with uncertain outcomes",
         ha="center", fontsize=9, color=GREEN)
ax4.text(5, 1.0, "❌ Avoid When: Simple linear workflows (booking flight)",
         ha="center", fontsize=9, color=RED)

ax4.text(5, 0.3, "Planning ≠ Prediction. Plans must adapt to reality.",
         ha="center", fontsize=9, color="#555", style="italic")

out = str(Path(__file__).resolve().parent.parent / "img" / "Plan and Execute.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out}")
