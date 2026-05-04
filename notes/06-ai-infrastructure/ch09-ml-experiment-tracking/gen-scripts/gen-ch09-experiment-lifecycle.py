"""
gen_ch09_experiment_lifecycle.py
Generates: ../img/ch09_experiment_lifecycle.png

Shows the complete lifecycle of an ML experiment:
Code + Data + Hyperparameters → Run → Metrics + Artifacts

Visual flow diagram with clear stages and arrows showing the experiment pipeline.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patches import Rectangle

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "img", "ch09_experiment_lifecycle.png")

# Ensure output directory exists
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# Create figure with dark background
fig, ax = plt.subplots(figsize=(16, 9), facecolor="#1a1a2e")
ax.set_facecolor("#1a1a2e")
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")

# Title
ax.text(8, 8.2, "ML Experiment Lifecycle", 
        ha="center", va="center", fontsize=26, fontweight="bold", color="white")
ax.text(8, 7.6, "Every experiment is a function: (code, data, params) → (metrics, artifacts)",
        ha="center", va="center", fontsize=12, color="#94a3b8", style="italic")

# Define colors
INPUT_COLOR = "#3b82f6"   # Blue
RUN_COLOR = "#8b5cf6"     # Purple
OUTPUT_COLOR = "#10b981"  # Green
ARROW_COLOR = "#64748b"   # Gray

# ============ INPUT STAGE ============
# Code
ax.add_patch(FancyBboxPatch((0.5, 4.5), 2.5, 1.8,
                            boxstyle="round,pad=0.1",
                            facecolor=INPUT_COLOR, edgecolor="white", linewidth=2))
ax.text(1.75, 5.8, "Code", ha="center", va="center",
        fontsize=16, fontweight="bold", color="white")
ax.text(1.75, 5.3, "train.py", ha="center", va="center",
        fontsize=11, color="white", family="monospace")
ax.text(1.75, 5.0, "git hash:", ha="center", va="center",
        fontsize=10, color="#bfdbfe")
ax.text(1.75, 4.7, "a3f2c9d", ha="center", va="center",
        fontsize=10, color="white", family="monospace")

# Data
ax.add_patch(FancyBboxPatch((0.5, 2.2), 2.5, 1.8,
                            boxstyle="round,pad=0.1",
                            facecolor=INPUT_COLOR, edgecolor="white", linewidth=2))
ax.text(1.75, 3.5, "Data", ha="center", va="center",
        fontsize=16, fontweight="bold", color="white")
ax.text(1.75, 3.0, "train.csv", ha="center", va="center",
        fontsize=11, color="white", family="monospace")
ax.text(1.75, 2.7, "DVC hash:", ha="center", va="center",
        fontsize=10, color="#bfdbfe")
ax.text(1.75, 2.4, "d9e8f3a", ha="center", va="center",
        fontsize=10, color="white", family="monospace")

# Hyperparameters
ax.add_patch(FancyBboxPatch((0.5, 0.0), 2.5, 1.8,
                            boxstyle="round,pad=0.1",
                            facecolor=INPUT_COLOR, edgecolor="white", linewidth=2))
ax.text(1.75, 1.3, "Params", ha="center", va="center",
        fontsize=16, fontweight="bold", color="white")
ax.text(1.75, 0.85, "lr: 5e-5", ha="center", va="center",
        fontsize=10, color="white", family="monospace")
ax.text(1.75, 0.55, "batch: 32", ha="center", va="center",
        fontsize=10, color="white", family="monospace")
ax.text(1.75, 0.25, "epochs: 3", ha="center", va="center",
        fontsize=10, color="white", family="monospace")

# ============ ARROWS TO RUN ============
# Arrow from Code
ax.add_patch(FancyArrowPatch((3.2, 5.4), (5.8, 4.5),
                            arrowstyle="->,head_width=0.4,head_length=0.4",
                            color=ARROW_COLOR, linewidth=2.5))

# Arrow from Data
ax.add_patch(FancyArrowPatch((3.2, 3.1), (5.8, 3.5),
                            arrowstyle="->,head_width=0.4,head_length=0.4",
                            color=ARROW_COLOR, linewidth=2.5))

# Arrow from Params
ax.add_patch(FancyArrowPatch((3.2, 0.9), (5.8, 2.5),
                            arrowstyle="->,head_width=0.4,head_length=0.4",
                            color=ARROW_COLOR, linewidth=2.5))

# ============ RUN STAGE ============
ax.add_patch(FancyBboxPatch((6.0, 1.5), 4.0, 4.0,
                            boxstyle="round,pad=0.15",
                            facecolor=RUN_COLOR, edgecolor="white", linewidth=3))
ax.text(8.0, 4.7, "Experiment Run", ha="center", va="center",
        fontsize=18, fontweight="bold", color="white")
ax.text(8.0, 4.1, "run_id: exp_7f8a2b", ha="center", va="center",
        fontsize=11, color="white", family="monospace")

# Run details box
ax.add_patch(Rectangle((6.5, 2.0), 3.0, 1.7, 
                       facecolor="#6d28d9", edgecolor="#a78bfa", linewidth=1.5))
ax.text(8.0, 3.35, "Training...", ha="center", va="center",
        fontsize=11, color="white", fontweight="bold")
ax.text(8.0, 2.95, "Epoch 1/3  loss: 0.42", ha="center", va="center",
        fontsize=9, color="#e0e7ff", family="monospace")
ax.text(8.0, 2.65, "Epoch 2/3  loss: 0.28", ha="center", va="center",
        fontsize=9, color="#e0e7ff", family="monospace")
ax.text(8.0, 2.35, "Epoch 3/3  loss: 0.19", ha="center", va="center",
        fontsize=9, color="#e0e7ff", family="monospace")
ax.text(8.0, 2.05, "✓ Training complete", ha="center", va="center",
        fontsize=9, color="#a7f3d0", fontweight="bold")

# ============ ARROWS TO OUTPUT ============
# Arrow to Metrics
ax.add_patch(FancyArrowPatch((10.2, 4.8), (12.8, 5.4),
                            arrowstyle="->,head_width=0.4,head_length=0.4",
                            color=ARROW_COLOR, linewidth=2.5))

# Arrow to Artifacts
ax.add_patch(FancyArrowPatch((10.2, 2.2), (12.8, 1.3),
                            arrowstyle="->,head_width=0.4,head_length=0.4",
                            color=ARROW_COLOR, linewidth=2.5))

# ============ OUTPUT STAGE ============
# Metrics
ax.add_patch(FancyBboxPatch((13.0, 4.5), 2.5, 2.3,
                            boxstyle="round,pad=0.1",
                            facecolor=OUTPUT_COLOR, edgecolor="white", linewidth=2))
ax.text(14.25, 6.4, "Metrics", ha="center", va="center",
        fontsize=16, fontweight="bold", color="white")
ax.text(14.25, 5.9, "accuracy: 94.2%", ha="center", va="center",
        fontsize=10, color="white", family="monospace")
ax.text(14.25, 5.5, "f1_score: 0.931", ha="center", va="center",
        fontsize=10, color="white", family="monospace")
ax.text(14.25, 5.1, "val_loss: 0.19", ha="center", va="center",
        fontsize=10, color="white", family="monospace")
ax.text(14.25, 4.7, "train_time: 2h", ha="center", va="center",
        fontsize=10, color="white", family="monospace")

# Artifacts
ax.add_patch(FancyBboxPatch((13.0, 0.0), 2.5, 2.3,
                            boxstyle="round,pad=0.1",
                            facecolor=OUTPUT_COLOR, edgecolor="white", linewidth=2))
ax.text(14.25, 2.0, "Artifacts", ha="center", va="center",
        fontsize=16, fontweight="bold", color="white")
ax.text(14.25, 1.55, "model.pt", ha="center", va="center",
        fontsize=10, color="white", family="monospace")
ax.text(14.25, 1.2, "confusion.png", ha="center", va="center",
        fontsize=10, color="white", family="monospace")
ax.text(14.25, 0.85, "feature_imp.csv", ha="center", va="center",
        fontsize=10, color="white", family="monospace")
ax.text(14.25, 0.5, "config.yaml", ha="center", va="center",
        fontsize=10, color="white", family="monospace")

# Add bottom note
ax.text(8, 0.3, "💡 Pro tip: Without tracking, this experiment is lost forever. With MLflow, it's searchable and reproducible.",
        ha="center", va="center", fontsize=10, color="#fbbf24", style="italic")

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, facecolor="#1a1a2e", edgecolor="none")
print(f"✓ Saved: {OUT_PATH}")
