"""
gen_ch06_cv_folds.py
Generates: ../img/ch06-cv-folds.png

Visual representation of 5-fold cross-validation showing how data is split
into train/test sets across folds.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "img", "ch06-cv-folds.png")

# ── Constants ─────────────────────────────────────────────────────────────────
N_FOLDS = 5
FOLD_NAMES = ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"]
MAES = [37, 40, 38, 39, 36]  # Example MAE values in $k

# Colors
COLOR_TRAIN = "#4ade80"  # green
COLOR_TEST = "#fb923c"   # orange
COLOR_BG = "#1a1a2e"
COLOR_TEXT = "white"
COLOR_GRID = "#4a4a6a"

# ── Figure setup ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor(COLOR_BG)
ax.set_facecolor(COLOR_BG)

# ── Draw folds ────────────────────────────────────────────────────────────────
y_positions = np.arange(N_FOLDS, 0, -1)  # Top to bottom: 5, 4, 3, 2, 1

for i, (fold_name, mae, y_pos) in enumerate(zip(FOLD_NAMES, MAES, y_positions)):
    # Each fold has 5 segments (representing 5 equal parts of the data)
    for j in range(N_FOLDS):
        x_start = j
        x_width = 1
        
        # Fold i uses segment i as test, others as train
        if j == i:
            color = COLOR_TEST
            label = "TEST" if i == 0 else None  # Label only once
        else:
            color = COLOR_TRAIN
            label = "TRAIN" if i == 0 and j == 1 else None  # Label only once
        
        # Draw rectangle
        rect = mpatches.Rectangle((x_start, y_pos - 0.4), x_width, 0.8,
                                   facecolor=color, edgecolor=COLOR_TEXT,
                                   linewidth=1.5, alpha=0.9)
        ax.add_patch(rect)
        
        # Add label text in the middle of first fold segments
        if i == 0:
            if j == i:
                ax.text(x_start + 0.5, y_pos, "TEST",
                       ha='center', va='center', fontsize=10,
                       color=COLOR_BG, weight='bold')
            elif j == 1:
                ax.text(x_start + 0.5, y_pos, "TRAIN",
                       ha='center', va='center', fontsize=10,
                       color=COLOR_BG, weight='bold')
    
    # Add fold label and MAE on the right
    ax.text(-0.5, y_pos, fold_name,
           ha='right', va='center', fontsize=11,
           color=COLOR_TEXT, weight='bold')
    ax.text(5.5, y_pos, f"MAE = ${mae}k",
           ha='left', va='center', fontsize=11,
           color=COLOR_TEXT)

# ── Summary statistics at bottom ──────────────────────────────────────────────
mean_mae = np.mean(MAES)
std_mae = np.std(MAES, ddof=1)

summary_y = 0.3
ax.text(2.5, summary_y, f"Mean MAE = ${mean_mae:.0f}k ± ${std_mae:.1f}k",
       ha='center', va='center', fontsize=13,
       color=COLOR_TEXT, weight='bold',
       bbox=dict(boxstyle='round,pad=0.6', facecolor=COLOR_GRID, 
                 edgecolor=COLOR_TEXT, linewidth=2))

# ── Styling ───────────────────────────────────────────────────────────────────
ax.set_xlim(-1.2, 6.5)
ax.set_ylim(0, N_FOLDS + 0.8)
ax.set_aspect('equal')
ax.axis('off')

# Title
ax.text(2.5, N_FOLDS + 0.5, "5-Fold Cross-Validation",
       ha='center', va='bottom', fontsize=15,
       color=COLOR_TEXT, weight='bold')
ax.text(2.5, N_FOLDS + 0.2, "Each fold uses 80% of data for training, 20% for testing",
       ha='center', va='bottom', fontsize=10,
       color=COLOR_TEXT, style='italic')

# ── Save ──────────────────────────────────────────────────────────────────────
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight', facecolor=COLOR_BG)
plt.close()

print(f"✓ Generated: {OUT_PATH}")
