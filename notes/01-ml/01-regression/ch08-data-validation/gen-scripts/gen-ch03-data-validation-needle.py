"""Generate needle animation for Ch.3 — Data Validation.

Shows MAE needle: 128k → 89k after validation pipeline.
Shows PSI violations caught (0.31 → 0.08 after filtering).
Speedometer-style with "SAFE" zone (<0.10) and "DANGER" zone (>0.25).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
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
    {"label": "Raw data", "mae": 128.0, "psi": 0.31,
     "caption": "Noisy features and drift inflate error."},
    {"label": "Schema validation", "mae": 105.0, "psi": 0.22,
     "caption": "Type checks and range constraints catch obvious issues."},
    {"label": "PSI filtering", "mae": 89.0, "psi": 0.08,
     "caption": "Drift detection ensures stable predictions."},
]

def create_frame(frame_idx, n_frames):
    """Create a single frame showing validation improvements."""
    fig = plt.figure(figsize=(14, 8), facecolor=DARK)
    
    # Interpolate between stages
    progress = frame_idx / max(n_frames - 1, 1)
    stage_progress = progress * (len(STAGES) - 1)
    stage_idx = min(int(stage_progress), len(STAGES) - 1)
    
    if stage_idx < len(STAGES) - 1:
        local_progress = stage_progress - stage_idx
        s0, s1 = STAGES[stage_idx], STAGES[stage_idx + 1]
        mae = s0["mae"] + (s1["mae"] - s0["mae"]) * local_progress
        psi = s0["psi"] + (s1["psi"] - s0["psi"]) * local_progress
        label = s1["label"] if local_progress > 0.5 else s0["label"]
        caption = s1["caption"] if local_progress > 0.5 else s0["caption"]
    else:
        mae = STAGES[-1]["mae"]
        psi = STAGES[-1]["psi"]
        label = STAGES[-1]["label"]
        caption = STAGES[-1]["caption"]
    
    # Main title
    fig.text(0.5, 0.96, "Ch.3 — Data Validation: Catch drift before it hurts",
             ha="center", va="top", fontsize=18, fontweight="bold", color=LIGHT)
    
    # MAE gauge (left) - lower is better
    ax_mae = fig.add_axes([0.18, 0.55, 0.3, 0.3])
    ax_mae.set_xlim(-1.2, 1.2)
    ax_mae.set_ylim(-0.3, 1.2)
    ax_mae.axis("off")
    
    # Draw MAE speedometer (inverted - lower is better)
    ax_mae.add_patch(Wedge((0, 0), 1.0, 180, 0, width=0.15,
                          facecolor=GREY, edgecolor=LIGHT, linewidth=2))
    
    mae_ratio = (128 - mae) / (128 - 85)  # Inverted: lower MAE is better
    mae_ratio = max(0.0, min(1.0, mae_ratio))
    mae_angle = 180 - (mae_ratio * 180)
    
    ax_mae.add_patch(Wedge((0, 0), 1.0, mae_angle, 0, width=0.15,
                          facecolor=GREEN, edgecolor=LIGHT, linewidth=2))
    
    # MAE needle
    needle_angle = np.radians(mae_angle)
    needle_x = 0.85 * np.cos(needle_angle)
    needle_y = 0.85 * np.sin(needle_angle)
    ax_mae.plot([0, needle_x], [0, needle_y], color=RED, linewidth=4, zorder=10)
    ax_mae.plot(0, 0, 'o', color=RED, markersize=12, zorder=11)
    
    ax_mae.text(0, -0.25, "MAE ($k)", ha="center", va="top",
               fontsize=12, fontweight="bold", color=LIGHT)
    ax_mae.text(0, 0.5, f"{mae:.0f}", ha="center", va="center",
               fontsize=18, fontweight="bold", color=LIGHT)
    ax_mae.text(-1.1, 0, "128", ha="right", va="center", fontsize=9, color=GREY)
    ax_mae.text(1.1, 0, "85", ha="left", va="center", fontsize=9, color=GREY)
    
    # PSI gauge (right) - lower is better, with zones
    ax_psi = fig.add_axes([0.52, 0.55, 0.3, 0.3])
    ax_psi.set_xlim(-1.2, 1.2)
    ax_psi.set_ylim(-0.3, 1.2)
    ax_psi.axis("off")
    
    # Draw PSI zones: SAFE (<0.10), WARNING (0.10-0.25), DANGER (>0.25)
    # Green zone: 0-0.10 (right side, 0-36 degrees from right)
    ax_psi.add_patch(Wedge((0, 0), 1.0, 0, 36, width=0.15,
                          facecolor=GREEN, edgecolor=LIGHT, linewidth=1))
    # Amber zone: 0.10-0.25 (36-90 degrees)
    ax_psi.add_patch(Wedge((0, 0), 1.0, 36, 90, width=0.15,
                          facecolor=AMBER, edgecolor=LIGHT, linewidth=1))
    # Red zone: 0.25-0.35 (90-180 degrees)
    ax_psi.add_patch(Wedge((0, 0), 1.0, 90, 180, width=0.15,
                          facecolor=RED, edgecolor=LIGHT, linewidth=1))
    
    # PSI needle
    psi_ratio = psi / 0.35
    psi_ratio = max(0.0, min(1.0, psi_ratio))
    psi_angle = 180 - (psi_ratio * 180)
    
    needle_angle = np.radians(psi_angle)
    needle_x = 0.85 * np.cos(needle_angle)
    needle_y = 0.85 * np.sin(needle_angle)
    ax_psi.plot([0, needle_x], [0, needle_y], color=DARK, linewidth=4, zorder=10)
    ax_psi.plot(0, 0, 'o', color=DARK, markersize=12, zorder=11)
    
    ax_psi.text(0, -0.25, "PSI (drift)", ha="center", va="top",
               fontsize=12, fontweight="bold", color=LIGHT)
    ax_psi.text(0, 0.5, f"{psi:.2f}", ha="center", va="center",
               fontsize=18, fontweight="bold", color=LIGHT)
    
    # Zone labels
    ax_psi.text(0.9, 0.15, "SAFE", ha="center", va="center",
               fontsize=8, color=GREEN, fontweight="bold")
    ax_psi.text(0.3, 0.8, "WARN", ha="center", va="center",
               fontsize=8, color=DARK, fontweight="bold")
    ax_psi.text(-0.8, 0.5, "DANGER", ha="center", va="center",
               fontsize=8, color=LIGHT, fontweight="bold")
    
    # Stage label and caption at bottom
    fig.text(0.5, 0.32, f"Stage: {label}",
             ha="center", va="center", fontsize=14, fontweight="bold", color=AMBER)
    fig.text(0.5, 0.25, caption,
             ha="center", va="center", fontsize=11, color=LIGHT)
    
    # Key insight
    fig.text(0.5, 0.16, "Key: PSI < 0.10 = Safe | PSI > 0.25 = Retrain needed",
             ha="center", va="center", fontsize=10, color=GREY, style="italic")
    
    # Progress bar
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
        temp_path = out_dir / f"temp_ch03_frame_{frame_idx:03d}.png"
        fig.savefig(temp_path, dpi=100, bbox_inches="tight", facecolor=DARK)
        plt.close(fig)
        
        frames.append(Image.open(temp_path))
        print(f"  Frame {frame_idx + 1}/{n_frames}")
    
    # Save as GIF
    gif_path = out_dir / "ch03-data-validation-needle.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0
    )
    
    # Clean up temporary frames
    for frame_idx in range(n_frames):
        temp_path = out_dir / f"temp_ch03_frame_{frame_idx:03d}.png"
        temp_path.unlink(missing_ok=True)
    
    print(f"\nwrote {gif_path}")

if __name__ == "__main__":
    main()
