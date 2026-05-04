"""Generate needle animation for Ch.2 — Class Imbalance.

Shows accuracy needle moving from 89% → 92% after SMOTE, along with
precision, recall, and F1 improvements. Uses speedometer-style dial.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, FancyArrowPatch
import numpy as np
from pathlib import Path
from PIL import Image

DARK = "#1a1a2e"
BLUE = "#2563eb"
GREEN = "#16a34a"
AMBER = "#d97706"
RED = "#dc2626"
GREY = "#9ca3af"
LIGHT = "#e5e7eb"

STAGES = [
    {"label": "Naive baseline", "accuracy": 89.0, "recall": 40.0, "f1": 52.0,
     "caption": "High accuracy masks poor recall on minority class."},
    {"label": "After SMOTE", "accuracy": 92.0, "recall": 78.0, "f1": 84.0,
     "caption": "Synthetic samples improve recall while maintaining accuracy."},
]

def create_frame(frame_idx, n_frames):
    """Create a single frame showing metric improvements."""
    fig = plt.figure(figsize=(14, 8), facecolor=DARK)
    
    # Interpolate between stages
    progress = frame_idx / max(n_frames - 1, 1)
    stage_idx = min(int(progress * len(STAGES)), len(STAGES) - 1)
    
    if stage_idx < len(STAGES) - 1:
        local_progress = (progress * len(STAGES)) - stage_idx
        s0, s1 = STAGES[stage_idx], STAGES[stage_idx + 1]
        accuracy = s0["accuracy"] + (s1["accuracy"] - s0["accuracy"]) * local_progress
        recall = s0["recall"] + (s1["recall"] - s0["recall"]) * local_progress
        f1 = s0["f1"] + (s1["f1"] - s0["f1"]) * local_progress
        label = s1["label"] if local_progress > 0.5 else s0["label"]
        caption = s1["caption"] if local_progress > 0.5 else s0["caption"]
    else:
        accuracy = STAGES[-1]["accuracy"]
        recall = STAGES[-1]["recall"]
        f1 = STAGES[-1]["f1"]
        label = STAGES[-1]["label"]
        caption = STAGES[-1]["caption"]
    
    # Main title
    fig.text(0.5, 0.96, "Ch.2 — Class Imbalance: Accuracy alone misleads",
             ha="center", va="top", fontsize=18, fontweight="bold", color=LIGHT)
    
    # Create three gauge panels
    metrics = [
        ("Accuracy (%)", accuracy, 85, 95, 0.15, 0.7),
        ("Recall (%)", recall, 30, 85, 0.45, 0.7),
        ("F1-Score", f1, 40, 90, 0.75, 0.7),
    ]
    
    for metric_name, value, min_val, max_val, x_pos, y_pos in metrics:
        ax = fig.add_axes([x_pos, y_pos, 0.22, 0.22])
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.3, 1.2)
        ax.axis("off")
        
        # Draw speedometer arc
        theta1, theta2 = 180, 0
        ax.add_patch(Wedge((0, 0), 1.0, theta1, theta2, width=0.15,
                          facecolor=GREY, edgecolor=LIGHT, linewidth=2))
        
        # Fill green portion based on value
        if max_val > min_val:
            ratio = (value - min_val) / (max_val - min_val)
        else:
            ratio = 0.5
        ratio = max(0.0, min(1.0, ratio))
        
        angle = 180 - (ratio * 180)
        ax.add_patch(Wedge((0, 0), 1.0, angle, theta2, width=0.15,
                          facecolor=GREEN, edgecolor=LIGHT, linewidth=2))
        
        # Draw needle
        needle_angle = np.radians(angle)
        needle_x = 0.85 * np.cos(needle_angle)
        needle_y = 0.85 * np.sin(needle_angle)
        
        ax.plot([0, needle_x], [0, needle_y], color=RED, linewidth=4, zorder=10)
        ax.plot(0, 0, 'o', color=RED, markersize=12, zorder=11)
        
        # Labels
        ax.text(0, -0.25, metric_name, ha="center", va="top",
               fontsize=11, fontweight="bold", color=LIGHT)
        ax.text(0, 0.5, f"{value:.1f}", ha="center", va="center",
               fontsize=16, fontweight="bold", color=LIGHT)
        
        # Min/max labels
        ax.text(-1.1, 0, f"{min_val}", ha="right", va="center",
               fontsize=8, color=GREY)
        ax.text(1.1, 0, f"{max_val}", ha="left", va="center",
               fontsize=8, color=GREY)
    
    # Stage label and caption at bottom
    fig.text(0.5, 0.25, f"Stage: {label}",
             ha="center", va="center", fontsize=14, fontweight="bold", color=AMBER)
    fig.text(0.5, 0.18, caption,
             ha="center", va="center", fontsize=11, color=LIGHT)
    
    # Progress bar at bottom
    bar_y = 0.08
    bar_width = 0.6
    bar_x = 0.2
    ax_bar = fig.add_axes([bar_x, bar_y, bar_width, 0.02])
    ax_bar.set_xlim(0, 1)
    ax_bar.set_ylim(0, 1)
    ax_bar.axis("off")
    ax_bar.barh(0.5, progress, height=0.8, color=GREEN, edgecolor=LIGHT, linewidth=1.5)
    ax_bar.barh(0.5, 1, height=0.8, color=GREY, alpha=0.3)
    
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
        temp_path = out_dir / f"temp_ch02_frame_{frame_idx:03d}.png"
        fig.savefig(temp_path, dpi=100, bbox_inches="tight", facecolor=DARK)
        plt.close(fig)
        
        frames.append(Image.open(temp_path))
        print(f"  Frame {frame_idx + 1}/{n_frames}")
    
    # Save as GIF
    gif_path = out_dir / "ch02-class-imbalance-needle.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0
    )
    
    # Clean up temporary frames
    for frame_idx in range(n_frames):
        temp_path = out_dir / f"temp_ch02_frame_{frame_idx:03d}.png"
        temp_path.unlink(missing_ok=True)
    
    print(f"\nwrote {gif_path}")

if __name__ == "__main__":
    main()
