"""
Generate hierarchical coordination Gantt chart: Worker task timeline
Shows parallel vs sequential execution comparison with timing annotations.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import os

# Output paths
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'img')
os.makedirs(OUT_DIR, exist_ok=True)
OUTPUT_PNG = os.path.join(OUT_DIR, 'ch11-hierarchical-coordination.png')

# Color scheme
COLOR_BG = '#1a1a2e'
COLOR_RUNNING = '#4A90E2'    # Blue
COLOR_COMPLETE = '#27AE60'   # Green
COLOR_SEQUENTIAL = '#F39C12' # Yellow/Orange
COLOR_TEXT = '#FFFFFF'       # White text
COLOR_GRID = '#2a2a3e'       # Dark gray

def main():
    """Generate the hierarchical coordination Gantt chart."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_facecolor(COLOR_BG)
    fig.patch.set_facecolor(COLOR_BG)
    
    # Title
    ax.text(3, 4.5, "Hierarchical Agent Coordination Timeline", 
            fontsize=16, fontweight='bold', color=COLOR_TEXT)
    
    # Subtitle
    ax.text(3, 4.2, "Parallel Execution vs Sequential Baseline",
            fontsize=12, color=COLOR_TEXT, style='italic')
    
    # Setup axes
    ax.set_xlim(-0.5, 7)
    ax.set_ylim(-0.5, 4.8)
    
    # Y-axis labels (Workers)
    workers = ['Worker 1\n(11am batch)', 'Worker 2\n(12pm batch)', 'Worker 3\n(1pm batch)']
    y_positions = [3, 2, 1]
    
    for i, (worker, y) in enumerate(zip(workers, y_positions)):
        ax.text(-0.3, y, worker, ha='right', va='center',
                fontsize=11, color=COLOR_TEXT, fontweight='bold')
    
    # X-axis (time)
    time_labels = ['0s', '1s', '2s', '3s', '4s', '5s', '6s']
    for i, label in enumerate(time_labels):
        ax.text(i, -0.3, label, ha='center', va='top',
                fontsize=10, color=COLOR_TEXT)
        # Grid line
        ax.axvline(i, color=COLOR_GRID, linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Grid lines for workers
    for y in y_positions:
        ax.axhline(y - 0.4, color=COLOR_GRID, linestyle='-', linewidth=0.5, alpha=0.3)
        ax.axhline(y + 0.4, color=COLOR_GRID, linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Parallel execution bars (Workers 1-3 running 0-2s)
    bar_height = 0.6
    for i, y in enumerate(y_positions):
        # Running phase (0-2s) - blue
        bar = Rectangle((0, y - bar_height/2), 2, bar_height,
                       facecolor=COLOR_RUNNING, edgecolor=COLOR_TEXT,
                       linewidth=2, alpha=0.9)
        ax.add_patch(bar)
        ax.text(1, y, 'Running', ha='center', va='center',
                fontsize=10, color=COLOR_TEXT, fontweight='bold')
        
        # Completion marker at 2s - green
        complete_marker = Rectangle((2, y - bar_height/2), 0.15, bar_height,
                                   facecolor=COLOR_COMPLETE, edgecolor=COLOR_TEXT,
                                   linewidth=2)
        ax.add_patch(complete_marker)
        ax.text(2.3, y, '✓', fontsize=16, color=COLOR_COMPLETE, fontweight='bold')
    
    # Sequential baseline (dashed yellow line showing 0-6s)
    seq_y = 0.2
    ax.plot([0, 6], [seq_y, seq_y], color=COLOR_SEQUENTIAL,
            linestyle='--', linewidth=3, label='Sequential Execution')
    ax.text(3, seq_y + 0.15, 'Sequential Baseline (1 worker)', ha='center',
            fontsize=10, color=COLOR_SEQUENTIAL, fontweight='bold')
    
    # Markers for sequential phases
    for i in range(3):
        start = i * 2
        end = start + 2
        ax.plot([start, end], [seq_y, seq_y], color=COLOR_SEQUENTIAL,
                linewidth=6, alpha=0.6, solid_capstyle='round')
        ax.text((start + end) / 2, seq_y - 0.15, f'W{i+1}', ha='center',
                fontsize=9, color=COLOR_SEQUENTIAL, fontweight='bold')
    
    # Insight annotation box
    insight_box = FancyBboxPatch(
        (6.5, 1.5), 0.1, 2,  # Placeholder for positioning
        boxstyle="round,pad=0.15",
        facecolor='#2a2a3e',
        edgecolor=COLOR_TEXT,
        linewidth=2,
        alpha=0.9
    )
    # Adjust position and size
    insight_box.set_bounds(3.5, 3.5, 3, 0.6)
    ax.add_patch(insight_box)
    
    ax.text(5, 3.8, "⚡ Parallel Speedup", ha='center', va='center',
            fontsize=12, fontweight='bold', color=COLOR_TEXT)
    ax.text(5, 3.6, "3 workers finish in 2s vs 1 worker in 6s (3× faster)", ha='center', va='center',
            fontsize=10, color=COLOR_TEXT)
    
    # Color legend
    legend_y = 0.5
    legend_items = [
        (COLOR_RUNNING, 'Running'),
        (COLOR_COMPLETE, 'Completed'),
        (COLOR_SEQUENTIAL, 'Sequential')
    ]
    
    legend_x_start = 4.5
    for i, (color, label) in enumerate(legend_items):
        x = legend_x_start + i * 0.8
        # Color box
        box = Rectangle((x, legend_y - 0.1), 0.3, 0.2,
                       facecolor=color, edgecolor=COLOR_TEXT, linewidth=1)
        ax.add_patch(box)
        # Label
        ax.text(x + 0.4, legend_y, label, ha='left', va='center',
                fontsize=9, color=COLOR_TEXT)
    
    ax.text(legend_x_start, legend_y + 0.3, "Legend:", ha='left',
            fontsize=10, fontweight='bold', color=COLOR_TEXT)
    
    # Remove default axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Save
    print(f"Saving PNG to {OUTPUT_PNG}...")
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, facecolor=COLOR_BG, edgecolor='none')
    plt.close()
    print("✓ Complete!")

if __name__ == '__main__':
    main()
