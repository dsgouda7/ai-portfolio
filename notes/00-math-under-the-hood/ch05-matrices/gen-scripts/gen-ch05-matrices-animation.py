"""Generate animation for Ch.5 — Matrix multiplication as feature transformation.

Shows how a design matrix X (n_samples × n_features) multiplied by weight vector w
produces predictions ŷ. Visualizes the California housing dataset with 8 features
flowing through matrix multiplication, showing intermediate dot products accumulating.

Animation shows:
1. Feature vector [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]
2. Weight vector w learned during training
3. Dot product computation accumulating step-by-step
4. Final prediction ŷ
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np
from pathlib import Path
from PIL import Image

DARK = "#1a1a2e"
BLUE = "#2563eb"
GREEN = "#16a34a"
AMBER = "#d97706"
RED = "#dc2626"
PURPLE = "#7c3aed"
GREY = "#9ca3af"
LIGHT = "#e5e7eb"

# California housing features and typical values
FEATURE_NAMES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", 
                 "Population", "AveOccup", "Latitude", "Longitude"]
FEATURE_VALUES = np.array([3.87, 28.0, 5.43, 1.10, 1425.0, 3.16, 35.64, -119.57])
WEIGHTS = np.array([4.37, 0.95, -1.16, 0.79, 0.00, -0.04, -4.23, -4.15])  # learned weights

def create_frame(frame_idx, n_frames):
    """Create a single frame of the animation."""
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=DARK)
    ax.set_facecolor(DARK)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    
    # Progress through the computation
    progress = frame_idx / max(n_frames - 1, 1)
    active_feature = min(int(progress * len(FEATURE_NAMES)), len(FEATURE_NAMES) - 1)
    
    # Title
    fig.text(0.5, 0.95, "Ch.5 — Matrix Multiplication as Feature Transformation",
             ha="center", va="top", fontsize=18, fontweight="bold", color=LIGHT)
    
    # Draw feature vector X on the left
    x_start = 0.8
    y_start = 7.5
    feature_height = 0.55
    
    ax.text(x_start, y_start + 0.8, "Feature Vector X", 
            fontsize=12, fontweight="bold", color=LIGHT, ha="left")
    
    for i, (name, val) in enumerate(zip(FEATURE_NAMES, FEATURE_VALUES)):
        y_pos = y_start - i * feature_height
        color = GREEN if i <= active_feature else GREY
        alpha = 1.0 if i <= active_feature else 0.4
        
        rect = Rectangle((x_start, y_pos - 0.35), 1.8, 0.45, 
                        facecolor=color, edgecolor=LIGHT, linewidth=1.5, alpha=alpha)
        ax.add_patch(rect)
        
        ax.text(x_start + 0.9, y_pos - 0.125, f"{name}: {val:.2f}",
               ha="center", va="center", fontsize=9, color=DARK, fontweight="bold")
    
    # Draw weight vector w
    w_start = 3.5
    ax.text(w_start, y_start + 0.8, "Weight Vector w", 
            fontsize=12, fontweight="bold", color=LIGHT, ha="left")
    
    for i, (name, weight) in enumerate(zip(FEATURE_NAMES, WEIGHTS)):
        y_pos = y_start - i * feature_height
        color = BLUE if i <= active_feature else GREY
        alpha = 1.0 if i <= active_feature else 0.4
        
        rect = Rectangle((w_start, y_pos - 0.35), 1.6, 0.45,
                        facecolor=color, edgecolor=LIGHT, linewidth=1.5, alpha=alpha)
        ax.add_patch(rect)
        
        ax.text(w_start + 0.8, y_pos - 0.125, f"w[{i}]: {weight:.2f}",
               ha="center", va="center", fontsize=9, color=LIGHT, fontweight="bold")
    
    # Draw multiplication arrows and products
    prod_start = 6.0
    if active_feature >= 0:
        ax.text(prod_start + 0.5, y_start + 0.8, "Products", 
                fontsize=12, fontweight="bold", color=LIGHT, ha="center")
        
        for i in range(min(active_feature + 1, len(FEATURE_NAMES))):
            y_pos = y_start - i * feature_height
            product = FEATURE_VALUES[i] * WEIGHTS[i]
            
            # Arrow from feature to product
            ax.add_patch(FancyArrowPatch((x_start + 1.8, y_pos - 0.125), 
                                        (prod_start - 0.1, y_pos - 0.125),
                                        arrowstyle="->", color=GREEN, lw=2, alpha=0.6))
            # Arrow from weight to product
            ax.add_patch(FancyArrowPatch((w_start + 1.6, y_pos - 0.125),
                                        (prod_start - 0.1, y_pos - 0.125),
                                        arrowstyle="->", color=BLUE, lw=2, alpha=0.6))
            
            # Product value
            rect = Rectangle((prod_start, y_pos - 0.35), 1.2, 0.45,
                           facecolor=PURPLE, edgecolor=LIGHT, linewidth=1.5)
            ax.add_patch(rect)
            ax.text(prod_start + 0.6, y_pos - 0.125, f"{product:.2f}",
                   ha="center", va="center", fontsize=9, color=LIGHT, fontweight="bold")
    
    # Draw accumulating sum on the right
    sum_start = 8.0
    ax.text(sum_start + 0.8, y_start + 0.8, "Accumulating Sum", 
            fontsize=12, fontweight="bold", color=LIGHT, ha="center")
    
    # Compute running sum
    running_sum = 0.0
    for i in range(min(active_feature + 1, len(FEATURE_NAMES))):
        running_sum += FEATURE_VALUES[i] * WEIGHTS[i]
    
    # Display current sum
    sum_y = y_start - 1.5
    rect = Rectangle((sum_start, sum_y - 0.6), 1.6, 1.0,
                    facecolor=AMBER, edgecolor=LIGHT, linewidth=2.5)
    ax.add_patch(rect)
    
    ax.text(sum_start + 0.8, sum_y - 0.1, f"Σ = {running_sum:.2f}",
           ha="center", va="center", fontsize=14, color=DARK, fontweight="bold")
    
    # Final prediction at the bottom
    if active_feature >= len(FEATURE_NAMES) - 1:
        pred_y = 1.5
        rect = Rectangle((3.5, pred_y - 0.7), 3.0, 1.2,
                        facecolor=RED, edgecolor=LIGHT, linewidth=3)
        ax.add_patch(rect)
        
        ax.text(5.0, pred_y - 0.1, f"Prediction ŷ = {running_sum:.2f}",
               ha="center", va="center", fontsize=16, color=LIGHT, fontweight="bold")
        
        ax.text(5.0, pred_y - 0.5, "(predicted house value in $100k)",
               ha="center", va="center", fontsize=10, color=LIGHT, style="italic")
    
    # Progress indicator
    caption = f"Computing dot product: feature {min(active_feature + 1, len(FEATURE_NAMES))}/{len(FEATURE_NAMES)}"
    if active_feature >= len(FEATURE_NAMES) - 1:
        caption = "Complete! Matrix multiplication produces prediction ŷ"
    
    ax.text(5.0, 0.3, caption,
           ha="center", va="center", fontsize=11, color=LIGHT,
           bbox=dict(boxstyle="round,pad=0.5", facecolor=DARK, edgecolor=LIGHT, linewidth=1.5))
    
    return fig

def main():
    out_dir = Path(__file__).parent.parent / "img"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    n_frames = 30
    frames = []
    
    print("Generating frames...")
    for frame_idx in range(n_frames):
        fig = create_frame(frame_idx, n_frames)
        
        # Save frame to temporary buffer
        temp_path = out_dir / f"temp_frame_{frame_idx:03d}.png"
        fig.savefig(temp_path, dpi=100, bbox_inches="tight", facecolor=DARK)
        plt.close(fig)
        
        frames.append(Image.open(temp_path))
        print(f"  Frame {frame_idx + 1}/{n_frames}")
    
    # Save as GIF
    gif_path = out_dir / "ch05_matrices-animation.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0
    )
    
    # Clean up temporary frames
    for frame_idx in range(n_frames):
        temp_path = out_dir / f"temp_frame_{frame_idx:03d}.png"
        temp_path.unlink(missing_ok=True)
    
    print(f"\nwrote {gif_path}")

if __name__ == "__main__":
    main()
