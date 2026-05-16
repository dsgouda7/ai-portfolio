"""
gen_ch09_model_registry_stages.py
Generates: ../img/ch09_model_registry_stages.png

Shows the model lifecycle through registry stages:
None → Staging → Production → Archived

Visual state machine with transitions, annotations, and stage descriptions.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patches as mpatches

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "img", "ch09_model_registry_stages.png")

# Ensure output directory exists
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# Create figure
fig, ax = plt.subplots(figsize=(16, 10), facecolor="#0f172a")
ax.set_facecolor("#0f172a")
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis("off")

# Title
ax.text(8, 9.3, "Model Registry Lifecycle", 
        ha="center", va="center", fontsize=26, fontweight="bold", color="white")
ax.text(8, 8.7, "Safe model promotion: test → stage → prod → archive",
        ha="center", va="center", fontsize=12, color="#94a3b8", style="italic")

# Define colors for each stage
NONE_COLOR = "#64748b"      # Gray
STAGING_COLOR = "#f59e0b"   # Amber
PRODUCTION_COLOR = "#10b981" # Green
ARCHIVED_COLOR = "#6366f1"  # Indigo

# Stage positions (x, y)
stages = [
    {"name": "None", "x": 2.5, "y": 5.0, "color": NONE_COLOR, 
     "desc": "Model trained\nbut not registered\nin any environment"},
    {"name": "Staging", "x": 6.5, "y": 5.0, "color": STAGING_COLOR,
     "desc": "Testing in staging env\nRunning A/B tests\nValidating performance"},
    {"name": "Production", "x": 10.5, "y": 5.0, "color": PRODUCTION_COLOR,
     "desc": "Serving live traffic\nMonitored 24/7\nRollback-ready"},
    {"name": "Archived", "x": 13.5, "y": 5.0, "color": ARCHIVED_COLOR,
     "desc": "Retired from service\nKept for compliance\nRead-only"},
]

# Draw stages as rounded boxes
box_width = 2.5
box_height = 2.0

for stage in stages:
    # Main stage box
    ax.add_patch(FancyBboxPatch((stage["x"] - box_width/2, stage["y"] - box_height/2), 
                                box_width, box_height,
                                boxstyle="round,pad=0.15",
                                facecolor=stage["color"], edgecolor="white", linewidth=3))
    
    # Stage name
    ax.text(stage["x"], stage["y"] + 0.5, stage["name"], 
            ha="center", va="center", fontsize=18, fontweight="bold", color="white")
    
    # Stage description
    for i, line in enumerate(stage["desc"].split("\n")):
        ax.text(stage["x"], stage["y"] - 0.1 - i*0.3, line,
                ha="center", va="center", fontsize=9, color="white")

# Draw transition arrows
arrow_y = 5.0
arrow_style = "->,head_width=0.3,head_length=0.3"

# None → Staging
ax.add_patch(FancyArrowPatch((3.75, arrow_y), (5.25, arrow_y),
                            arrowstyle=arrow_style,
                            color="#fbbf24", linewidth=3))
ax.text(4.5, arrow_y + 0.6, "register", ha="center", va="center",
        fontsize=10, color="#fbbf24", fontweight="bold")
ax.text(4.5, arrow_y + 0.3, "mlflow.register_model()", ha="center", va="center",
        fontsize=8, color="#fde68a", family="monospace")

# Staging → Production
ax.add_patch(FancyArrowPatch((7.75, arrow_y), (9.25, arrow_y),
                            arrowstyle=arrow_style,
                            color="#34d399", linewidth=3))
ax.text(8.5, arrow_y + 0.6, "promote", ha="center", va="center",
        fontsize=10, color="#34d399", fontweight="bold")
ax.text(8.5, arrow_y + 0.3, "transition_stage('Production')", ha="center", va="center",
        fontsize=8, color="#a7f3d0", family="monospace")

# Production → Archived
ax.add_patch(FancyArrowPatch((11.75, arrow_y), (12.25, arrow_y),
                            arrowstyle=arrow_style,
                            color="#818cf8", linewidth=3))
ax.text(12.0, arrow_y + 0.6, "retire", ha="center", va="center",
        fontsize=10, color="#818cf8", fontweight="bold")
ax.text(12.0, arrow_y + 0.3, "archive_model()", ha="center", va="center",
        fontsize=8, color="#c7d2fe", family="monospace")

# Rollback arrow (Production → Staging)
rollback_start_x = 10.5
rollback_end_x = 6.5
rollback_y_top = 6.3
rollback_y_mid = 7.2

ax.add_patch(FancyArrowPatch((rollback_start_x, rollback_y_top), 
                            (rollback_start_x, rollback_y_mid),
                            arrowstyle="-",
                            color="#ef4444", linewidth=2.5, linestyle="dashed"))
ax.add_patch(FancyArrowPatch((rollback_start_x, rollback_y_mid), 
                            (rollback_end_x, rollback_y_mid),
                            arrowstyle="-",
                            color="#ef4444", linewidth=2.5, linestyle="dashed"))
ax.add_patch(FancyArrowPatch((rollback_end_x, rollback_y_mid), 
                            (rollback_end_x, rollback_y_top),
                            arrowstyle=arrow_style,
                            color="#ef4444", linewidth=2.5, linestyle="dashed"))

ax.text(8.5, rollback_y_mid + 0.3, "🚨 ROLLBACK (if bugs detected)", 
        ha="center", va="center", fontsize=10, color="#fca5a5", fontweight="bold")

# Version tracking boxes below stages
version_y = 2.8
ax.text(8, version_y + 0.5, "Model Version Tracking", 
        ha="center", va="center", fontsize=14, fontweight="bold", color="#94a3b8")

versions = [
    {"stage": "Staging", "x": 6.5, "versions": ["v7", "v8"], "active": "v8"},
    {"stage": "Production", "x": 10.5, "versions": ["v5", "v6"], "active": "v6"},
    {"stage": "Archived", "x": 13.5, "versions": ["v1", "v2", "v3", "v4"], "active": None},
]

for ver_info in versions:
    # Version box
    ax.add_patch(FancyBboxPatch((ver_info["x"] - 1.0, version_y - 0.8), 2.0, 1.0,
                                boxstyle="round,pad=0.08",
                                facecolor="#1e293b", edgecolor="#475569", linewidth=1.5))
    
    ax.text(ver_info["x"], version_y + 0.1, f'{ver_info["stage"]}:', 
            ha="center", va="center", fontsize=10, color="#cbd5e1", fontweight="bold")
    
    version_text = ", ".join(ver_info["versions"])
    ax.text(ver_info["x"], version_y - 0.25, version_text,
            ha="center", va="center", fontsize=9, color="#94a3b8", family="monospace")
    
    if ver_info["active"]:
        ax.text(ver_info["x"], version_y - 0.55, f'Active: {ver_info["active"]} ✓',
                ha="center", va="center", fontsize=9, color="#10b981", fontweight="bold")

# Example timeline
timeline_y = 1.2
ax.add_patch(FancyBboxPatch((1.0, timeline_y - 0.3), 14.0, 0.8,
                            boxstyle="round,pad=0.08",
                            facecolor="#18181b", edgecolor="#3f3f46", linewidth=1.5))

ax.text(1.3, timeline_y + 0.25, "Example Timeline:", 
        ha="left", va="center", fontsize=11, color="#e4e4e7", fontweight="bold")

timeline_events = [
    ("Day 1", "Train v8, register to Staging"),
    ("Day 2-3", "A/B test v8 vs v6 (10% traffic)"),
    ("Day 4", "Promote v8 → Production"),
    ("Day 30", "Archive v1-v4 (compliance retention)"),
]

event_x_start = 5.0
for i, (day, event) in enumerate(timeline_events):
    x_pos = event_x_start + i * 2.5
    ax.text(x_pos, timeline_y + 0.25, day, 
            ha="center", va="center", fontsize=9, color="#a1a1aa", fontweight="bold", family="monospace")
    ax.text(x_pos, timeline_y - 0.1, event,
            ha="center", va="center", fontsize=8, color="#d4d4d8")

# Key takeaway
ax.text(8, 0.3, "🔑 Key: Never deploy straight to production. Always stage → test → promote with rollback plan.",
        ha="center", va="center", fontsize=11, color="#fbbf24", fontweight="bold")

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, facecolor="#0f172a", edgecolor="none")
print(f"✓ Saved: {OUT_PATH}")
