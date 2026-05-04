"""
gen_ch09_mlflow_ui_comparison.py
Generates: ../img/ch09_mlflow_ui_comparison.png

Shows a side-by-side comparison of multiple experiment runs in an MLflow-style UI table.
Demonstrates how to compare hyperparameters, metrics, and identify the best run.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.patches as mpatches

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "img", "ch09_mlflow_ui_comparison.png")

# Ensure output directory exists
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# Create figure with light background (mimicking web UI)
fig, ax = plt.subplots(figsize=(16, 10), facecolor="#f8fafc")
ax.set_facecolor("#f8fafc")
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis("off")

# Title section
ax.text(8, 9.4, "MLflow Experiment Comparison", 
        ha="center", va="center", fontsize=24, fontweight="bold", color="#1e293b")
ax.text(8, 8.9, "Experiment: BERT Fine-tuning Sweep  |  12 runs  |  Filter: ✓ Completed",
        ha="center", va="center", fontsize=11, color="#64748b", family="monospace")

# Define colors
HEADER_COLOR = "#334155"
BEST_ROW_COLOR = "#dcfce7"  # Light green for best run
ALT_ROW_COLOR = "#ffffff"
BORDER_COLOR = "#cbd5e1"
HIGHLIGHT_COLOR = "#10b981"  # Green for best metric

# Table data (12 runs)
runs = [
    {"run": "exp_001", "lr": "1e-5", "batch": "16", "warmup": "500", "acc": "91.2", "f1": "0.898", "time": "3.2h"},
    {"run": "exp_002", "lr": "1e-5", "batch": "16", "warmup": "1000", "acc": "91.8", "f1": "0.905", "time": "3.3h"},
    {"run": "exp_003", "lr": "1e-5", "batch": "32", "warmup": "500", "acc": "92.1", "f1": "0.910", "time": "2.1h"},
    {"run": "exp_004", "lr": "1e-5", "batch": "32", "warmup": "1000", "acc": "92.5", "f1": "0.915", "time": "2.2h"},
    {"run": "exp_005", "lr": "5e-5", "batch": "16", "warmup": "500", "acc": "93.4", "f1": "0.925", "time": "3.1h"},
    {"run": "exp_006", "lr": "5e-5", "batch": "16", "warmup": "1000", "acc": "93.8", "f1": "0.930", "time": "3.2h"},
    {"run": "exp_007", "lr": "5e-5", "batch": "32", "warmup": "500", "acc": "94.7", "f1": "0.941", "time": "2.0h"},  # BEST
    {"run": "exp_008", "lr": "5e-5", "batch": "32", "warmup": "1000", "acc": "94.3", "f1": "0.937", "time": "2.1h"},
    {"run": "exp_009", "lr": "1e-4", "batch": "16", "warmup": "500", "acc": "90.8", "f1": "0.893", "time": "3.0h"},
    {"run": "exp_010", "lr": "1e-4", "batch": "16", "warmup": "1000", "acc": "91.3", "f1": "0.900", "time": "3.1h"},
    {"run": "exp_011", "lr": "1e-4", "batch": "32", "warmup": "500", "acc": "92.9", "f1": "0.920", "time": "1.9h"},
    {"run": "exp_012", "lr": "1e-4", "batch": "32", "warmup": "1000", "acc": "93.1", "f1": "0.922", "time": "2.0h"},
]

# Table dimensions
table_top = 8.3
row_height = 0.5
col_widths = [1.5, 1.2, 1.2, 1.5, 1.8, 1.5, 1.5]  # Run, LR, Batch, Warmup, Accuracy, F1, Time
col_starts = [1.5, 3.0, 4.2, 5.4, 6.9, 8.7, 10.2]
headers = ["Run ID", "LR", "Batch", "Warmup", "Accuracy ↓", "F1 Score", "Time"]

# Draw header row
header_y = table_top
for i, (header, x, w) in enumerate(zip(headers, col_starts, col_widths)):
    ax.add_patch(Rectangle((x, header_y), w, row_height, 
                           facecolor=HEADER_COLOR, edgecolor=BORDER_COLOR, linewidth=1))
    color = "white"
    weight = "bold"
    if "Accuracy" in header:
        # Add sort indicator
        ax.text(x + w/2, header_y + row_height/2, header, 
                ha="center", va="center", fontsize=10, color=color, fontweight=weight)
    else:
        ax.text(x + w/2, header_y + row_height/2, header, 
                ha="center", va="center", fontsize=10, color=color, fontweight=weight)

# Draw data rows
best_idx = 6  # exp_007 has highest accuracy
for idx, run in enumerate(runs):
    row_y = header_y - (idx + 1) * row_height
    
    # Alternate row colors, highlight best run
    if idx == best_idx:
        row_color = BEST_ROW_COLOR
        border_width = 2.5
        border_color = HIGHLIGHT_COLOR
    else:
        row_color = ALT_ROW_COLOR if idx % 2 == 0 else "#f8fafc"
        border_width = 1
        border_color = BORDER_COLOR
    
    # Draw row cells
    values = [run["run"], run["lr"], run["batch"], run["warmup"], 
              run["acc"] + "%", run["f1"], run["time"]]
    
    for val, x, w in zip(values, col_starts, col_widths):
        ax.add_patch(Rectangle((x, row_y), w, row_height,
                               facecolor=row_color, edgecolor=border_color, linewidth=border_width))
        
        # Text color and weight
        text_color = "#1e293b"
        text_weight = "normal"
        fontsize = 9
        
        # Highlight best metrics
        if idx == best_idx and val in [run["acc"] + "%", run["f1"]]:
            text_color = HIGHLIGHT_COLOR
            text_weight = "bold"
            fontsize = 10
        
        # Monospace for run IDs and numbers
        if val in [run["run"], run["lr"], run["batch"], run["warmup"], run["f1"], run["time"]]:
            family = "monospace"
        else:
            family = "sans-serif"
        
        ax.text(x + w/2, row_y + row_height/2, val,
                ha="center", va="center", fontsize=fontsize, 
                color=text_color, fontweight=text_weight, family=family)

# Add star icon for best run
star_x = col_starts[0] - 0.3
star_y = header_y - (best_idx + 1) * row_height + row_height/2
ax.text(star_x, star_y, "⭐", ha="center", va="center", fontsize=18)

# Add action buttons for best run
button_y = header_y - (best_idx + 1) * row_height + row_height/2
button_x = col_starts[-1] + col_widths[-1] + 0.2

# Register button
ax.add_patch(FancyBboxPatch((button_x, button_y - 0.15), 1.5, 0.3,
                            boxstyle="round,pad=0.02",
                            facecolor="#3b82f6", edgecolor="#2563eb", linewidth=1.5))
ax.text(button_x + 0.75, button_y, "Register Model",
        ha="center", va="center", fontsize=8, color="white", fontweight="bold")

# Insights panel
insights_y = 1.5
ax.add_patch(FancyBboxPatch((1.5, insights_y), 13.2, 1.2,
                            boxstyle="round,pad=0.1",
                            facecolor="#fef3c7", edgecolor="#fbbf24", linewidth=2))

ax.text(8.1, insights_y + 0.85, "🔍 Key Insights from this Sweep",
        ha="center", va="center", fontsize=12, fontweight="bold", color="#92400e")

insights = [
    "• Best run: exp_007 with lr=5e-5, batch=32, warmup=500",
    "• Sweet spot: Mid-range learning rate (5e-5) with larger batch size",
    "• Time vs. Accuracy: Batch size 32 is 35% faster than batch size 16 with better results",
]

for i, insight in enumerate(insights):
    ax.text(1.8, insights_y + 0.5 - i*0.25, insight,
            ha="left", va="center", fontsize=9, color="#78350f", family="sans-serif")

# Bottom note
ax.text(8, 0.5, "💡 Without experiment tracking, finding the best hyperparameters requires tedious manual log parsing",
        ha="center", va="center", fontsize=10, color="#64748b", style="italic")

ax.text(8, 0.15, "With MLflow: One click to compare runs, identify winners, and register models for production",
        ha="center", va="center", fontsize=10, color="#64748b", style="italic")

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, facecolor="#f8fafc", edgecolor="none")
print(f"✓ Saved: {OUT_PATH}")
