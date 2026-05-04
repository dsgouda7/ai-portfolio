"""Generate static progress check diagram for Ch.2 — Class Imbalance.

Shows constraint status after Ch.2:
- RealtyML constraints: MAE 174k→148k (partial), Recall 40%→78% (achieved)
- Uses checkmarks for achieved, X for not achieved
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np
from pathlib import Path

DARK = "#1a1a2e"
BLUE = "#2563eb"
GREEN = "#16a34a"
AMBER = "#d97706"
RED = "#dc2626"
GREY = "#9ca3af"
LIGHT = "#e5e7eb"

def main():
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=DARK)
    ax.set_facecolor(DARK)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    
    # Title
    fig.text(0.5, 0.95, "Ch.2 Progress Check — RealtyML Constraints",
             ha="center", va="top", fontsize=20, fontweight="bold", color=LIGHT)
    
    # Mission box
    mission_box = FancyBboxPatch((0.5, 7.2), 9.0, 1.8,
                                 boxstyle="round,pad=0.15",
                                 facecolor=BLUE, edgecolor=LIGHT, linewidth=2)
    ax.add_patch(mission_box)
    
    ax.text(5.0, 8.5, "Mission: Predict home fraud with balance",
           ha="center", va="center", fontsize=14, fontweight="bold", color=LIGHT)
    ax.text(5.0, 7.8, "RealtyML needs: MAE < $100k AND Recall > 70% on fraud cases",
           ha="center", va="center", fontsize=11, color=LIGHT)
    
    # Constraint 1: MAE
    y_pos = 5.8
    constraint_box = FancyBboxPatch((0.8, y_pos - 0.8), 4.0, 2.0,
                                    boxstyle="round,pad=0.12",
                                    facecolor=AMBER, edgecolor=LIGHT, linewidth=2, alpha=0.8)
    ax.add_patch(constraint_box)
    
    ax.text(2.8, y_pos + 0.8, "Constraint 1: MAE < $100k",
           ha="center", va="center", fontsize=12, fontweight="bold", color=DARK)
    
    ax.text(2.8, y_pos + 0.3, "Before: MAE = $174k",
           ha="center", va="center", fontsize=10, color=DARK)
    ax.text(2.8, y_pos - 0.1, "After Ch.2: MAE = $148k",
           ha="center", va="center", fontsize=10, color=DARK, fontweight="bold")
    
    # Status: Partial (X)
    ax.text(2.8, y_pos - 0.5, "❌ NOT YET",
           ha="center", va="center", fontsize=16, color=RED, fontweight="bold")
    
    # Constraint 2: Recall
    constraint_box2 = FancyBboxPatch((5.2, y_pos - 0.8), 4.0, 2.0,
                                     boxstyle="round,pad=0.12",
                                     facecolor=GREEN, edgecolor=LIGHT, linewidth=2, alpha=0.8)
    ax.add_patch(constraint_box2)
    
    ax.text(7.2, y_pos + 0.8, "Constraint 2: Recall > 70%",
           ha="center", va="center", fontsize=12, fontweight="bold", color=DARK)
    
    ax.text(7.2, y_pos + 0.3, "Before: Recall = 40%",
           ha="center", va="center", fontsize=10, color=DARK)
    ax.text(7.2, y_pos - 0.1, "After Ch.2: Recall = 78%",
           ha="center", va="center", fontsize=10, color=DARK, fontweight="bold")
    
    # Status: Achieved (✓)
    ax.text(7.2, y_pos - 0.5, "✅ ACHIEVED",
           ha="center", va="center", fontsize=16, color=DARK, fontweight="bold")
    
    # Summary box at bottom
    summary_box = FancyBboxPatch((1.5, 2.0), 7.0, 2.2,
                                 boxstyle="round,pad=0.15",
                                 facecolor=DARK, edgecolor=AMBER, linewidth=3)
    ax.add_patch(summary_box)
    
    ax.text(5.0, 3.6, "Summary: Partial Success",
           ha="center", va="center", fontsize=14, fontweight="bold", color=LIGHT)
    
    ax.text(5.0, 3.1, "✅ SMOTE balanced the classes → Recall improved",
           ha="center", va="center", fontsize=11, color=GREEN)
    ax.text(5.0, 2.7, "❌ MAE still too high → Need better features or regularization",
           ha="center", va="center", fontsize=11, color=RED)
    ax.text(5.0, 2.3, "Next: Ch.3 will add data validation to reduce MAE further",
           ha="center", va="center", fontsize=11, color=LIGHT, style="italic")
    
    # Footer
    ax.text(5.0, 0.5, "Key Insight: Class imbalance fixes recall but doesn't solve prediction accuracy alone",
           ha="center", va="center", fontsize=10, color=GREY, style="italic")
    
    # Save
    out_dir = Path(__file__).parent.parent / "img"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "ch02-class-imbalance-progress-check.png"
    
    fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    
    print(f"wrote {png_path}")

if __name__ == "__main__":
    main()
