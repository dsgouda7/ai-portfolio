"""
gen_ch10_drift_types.py
Generates: ../img/ch10_drift_types.png

Shows three types of drift in production ML systems:
- Data Drift: Input distribution changes
- Prediction Drift: Model output distribution changes  
- Concept Drift: Relationship between X and Y changes

Visual diagram with examples and detection methods for each type.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "img", "ch10_drift_types.png")

# Ensure output directory exists
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# Create figure with dark background
fig, ax = plt.subplots(figsize=(16, 9), facecolor="#1a1a2e")
ax.set_facecolor("#1a1a2e")
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")

# Title
ax.text(8, 8.2, "Three Types of Model Drift", 
        ha="center", va="center", fontsize=26, fontweight="bold", color="white")
ax.text(8, 7.6, "Understanding when and why models degrade in production",
        ha="center", va="center", fontsize=12, color="#94a3b8", style="italic")

# Define colors
DATA_COLOR = "#3b82f6"      # Blue - Data Drift
PRED_COLOR = "#8b5cf6"      # Purple - Prediction Drift
CONCEPT_COLOR = "#f59e0b"   # Amber - Concept Drift
LABEL_COLOR = "#94a3b8"     # Gray for labels

# ============ DATA DRIFT (LEFT) ============
x_data = 1.5
ax.add_patch(FancyBboxPatch((0.3, 3.0), 4.5, 3.5,
                            boxstyle="round,pad=0.15",
                            facecolor=DATA_COLOR, edgecolor="white", linewidth=2, alpha=0.9))

ax.text(x_data + 0.75, 6.1, "Data Drift", ha="center", va="center",
        fontsize=18, fontweight="bold", color="white")

ax.text(x_data + 0.75, 5.6, "Input distribution changes", ha="center", va="center",
        fontsize=11, color="white", style="italic")

# Example
ax.text(x_data + 0.75, 5.0, "Example:", ha="center", va="center",
        fontsize=11, fontweight="bold", color="#dbeafe")
ax.text(x_data + 0.75, 4.6, "House prices model trained on", ha="center", va="center",
        fontsize=10, color="white")
ax.text(x_data + 0.75, 4.3, "2019 data; avg price $300k", ha="center", va="center",
        fontsize=10, color="white")
ax.text(x_data + 0.75, 4.0, "→ 2024: avg price $450k", ha="center", va="center",
        fontsize=10, color="#fbbf24", fontweight="bold")

# Detection
ax.text(x_data + 0.75, 3.5, "Detection:", ha="center", va="center",
        fontsize=11, fontweight="bold", color="#dbeafe")
ax.text(x_data + 0.75, 3.2, "KS test, PSI, histograms", ha="center", va="center",
        fontsize=10, color="white", family="monospace")

# ============ PREDICTION DRIFT (CENTER) ============
x_pred = 6.5
ax.add_patch(FancyBboxPatch((5.3, 3.0), 4.5, 3.5,
                            boxstyle="round,pad=0.15",
                            facecolor=PRED_COLOR, edgecolor="white", linewidth=2, alpha=0.9))

ax.text(x_pred + 0.75, 6.1, "Prediction Drift", ha="center", va="center",
        fontsize=18, fontweight="bold", color="white")

ax.text(x_pred + 0.75, 5.6, "Output distribution changes", ha="center", va="center",
        fontsize=11, color="white", style="italic")

# Example
ax.text(x_pred + 0.75, 5.0, "Example:", ha="center", va="center",
        fontsize=11, fontweight="bold", color="#e9d5ff")
ax.text(x_pred + 0.75, 4.6, "Fraud model predicted 2% fraud", ha="center", va="center",
        fontsize=10, color="white")
ax.text(x_pred + 0.75, 4.3, "in training", ha="center", va="center",
        fontsize=10, color="white")
ax.text(x_pred + 0.75, 4.0, "→ Now: predicting 15% fraud", ha="center", va="center",
        fontsize=10, color="#fbbf24", fontweight="bold")

# Detection
ax.text(x_pred + 0.75, 3.5, "Detection:", ha="center", va="center",
        fontsize=11, fontweight="bold", color="#e9d5ff")
ax.text(x_pred + 0.75, 3.2, "Track output stats, alerts", ha="center", va="center",
        fontsize=10, color="white", family="monospace")

# ============ CONCEPT DRIFT (RIGHT) ============
x_concept = 11.5
ax.add_patch(FancyBboxPatch((10.3, 3.0), 4.5, 3.5,
                            boxstyle="round,pad=0.15",
                            facecolor=CONCEPT_COLOR, edgecolor="white", linewidth=2, alpha=0.9))

ax.text(x_concept + 0.75, 6.1, "Concept Drift", ha="center", va="center",
        fontsize=18, fontweight="bold", color="white")

ax.text(x_concept + 0.75, 5.6, "X → Y relationship changes", ha="center", va="center",
        fontsize=11, color="white", style="italic")

# Example
ax.text(x_concept + 0.75, 5.0, "Example:", ha="center", va="center",
        fontsize=11, fontweight="bold", color="#fef3c7")
ax.text(x_concept + 0.75, 4.6, "Email subject \"Free money!\"", ha="center", va="center",
        fontsize=10, color="white")
ax.text(x_concept + 0.75, 4.3, "was 99% spam in 2010", ha="center", va="center",
        fontsize=10, color="white")
ax.text(x_concept + 0.75, 4.0, "→ 2024: legitimate fintech", ha="center", va="center",
        fontsize=10, color="#fbbf24", fontweight="bold")

# Detection
ax.text(x_concept + 0.75, 3.5, "Detection:", ha="center", va="center",
        fontsize=11, fontweight="bold", color="#fef3c7")
ax.text(x_concept + 0.75, 3.2, "Requires ground truth labels", ha="center", va="center",
        fontsize=10, color="white", family="monospace")

# ============ BOTTOM COMPARISON TABLE ============
table_y = 1.8
ax.text(8, table_y, "When to Detect Each Type", ha="center", va="top",
        fontsize=14, fontweight="bold", color="white")

# Table headers
headers_y = table_y - 0.5
ax.text(2.75, headers_y, "Drift Type", ha="center", va="center",
        fontsize=11, fontweight="bold", color=LABEL_COLOR)
ax.text(7.25, headers_y, "Can Detect Without Labels?", ha="center", va="center",
        fontsize=11, fontweight="bold", color=LABEL_COLOR)
ax.text(12.5, headers_y, "Hardest to Catch?", ha="center", va="center",
        fontsize=11, fontweight="bold", color=LABEL_COLOR)

# Row 1: Data
row1_y = headers_y - 0.5
ax.text(2.75, row1_y, "Data Drift", ha="center", va="center",
        fontsize=10, color=DATA_COLOR, fontweight="bold")
ax.text(7.25, row1_y, "✓ Yes (compare distributions)", ha="center", va="center",
        fontsize=10, color="#10b981")
ax.text(12.5, row1_y, "Easiest", ha="center", va="center",
        fontsize=10, color="#10b981")

# Row 2: Prediction
row2_y = row1_y - 0.4
ax.text(2.75, row2_y, "Prediction Drift", ha="center", va="center",
        fontsize=10, color=PRED_COLOR, fontweight="bold")
ax.text(7.25, row2_y, "✓ Yes (track outputs)", ha="center", va="center",
        fontsize=10, color="#10b981")
ax.text(12.5, row2_y, "Moderate", ha="center", va="center",
        fontsize=10, color="#f59e0b")

# Row 3: Concept
row3_y = row2_y - 0.4
ax.text(2.75, row3_y, "Concept Drift", ha="center", va="center",
        fontsize=10, color=CONCEPT_COLOR, fontweight="bold")
ax.text(7.25, row3_y, "✗ No (need ground truth)", ha="center", va="center",
        fontsize=10, color="#ef4444")
ax.text(12.5, row3_y, "Hardest", ha="center", va="center",
        fontsize=10, color="#ef4444")

# Save figure
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"✓ Saved: {OUT_PATH}")
