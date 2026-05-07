"""
Generate hierarchical flow animation: Planner → Workers → Verifier orchestration
Shows catering order decomposition, parallel worker execution, and constraint verification.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Wedge
import imageio
import os

# Output paths
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'img')
os.makedirs(OUT_DIR, exist_ok=True)
OUTPUT_GIF = os.path.join(OUT_DIR, 'ch11-hierarchical-flow.gif')

# Color scheme
COLOR_BG = '#1a1a2e'
COLOR_PLANNER = '#4A90E2'    # Blue
COLOR_WORKER = '#50C878'     # Green
COLOR_VERIFIER = '#9B59B6'   # Purple
COLOR_SUCCESS = '#27AE60'    # Bright green
COLOR_TEXT = '#FFFFFF'       # White text
COLOR_ACCENT = '#FFD93D'     # Yellow

def draw_box(ax, x, y, w, h, label, color, alpha=0.9, linewidth=2):
    """Draw a rounded box with label."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.1",
        facecolor=color,
        edgecolor=COLOR_TEXT,
        linewidth=linewidth,
        alpha=alpha
    )
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, label,
            ha='center', va='center',
            fontsize=11, fontweight='bold',
            color=COLOR_TEXT)

def draw_arrow(ax, x1, y1, x2, y2, color=COLOR_TEXT, linewidth=2):
    """Draw an arrow between points."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->,head_width=0.3,head_length=0.6',
        color=color,
        linewidth=linewidth,
        zorder=3
    )
    ax.add_patch(arrow)

def draw_progress_bar(ax, x, y, w, h, progress, color):
    """Draw a progress bar."""
    # Background
    bg = Rectangle((x, y), w, h, facecolor='#2a2a3e', edgecolor=COLOR_TEXT, linewidth=1)
    ax.add_patch(bg)
    # Progress
    if progress > 0:
        prog = Rectangle((x, y), w * progress, h, facecolor=color, alpha=0.8)
        ax.add_patch(prog)
    # Percentage
    ax.text(x + w/2, y + h/2, f"{int(progress*100)}%",
            ha='center', va='center', fontsize=9, color=COLOR_TEXT, fontweight='bold')

def draw_checkmark(ax, x, y, size=0.3, color=COLOR_SUCCESS):
    """Draw a checkmark symbol."""
    # Checkmark path
    check_x = [x - size/2, x - size/4, x + size/2]
    check_y = [y, y - size/2, y + size/2]
    ax.plot(check_x, check_y, color=color, linewidth=4, zorder=10)

def draw_budget_gauge(ax, x, y, current, total, radius=0.5):
    """Draw a budget gauge showing usage."""
    # Background arc
    arc_bg = Wedge((x, y), radius, 0, 180, facecolor='#2a2a3e', edgecolor=COLOR_TEXT, linewidth=2)
    ax.add_patch(arc_bg)
    
    # Usage arc
    angle = 180 * (current / total)
    color = COLOR_SUCCESS if current <= total else '#E74C3C'
    arc_usage = Wedge((x, y), radius, 0, angle, facecolor=color, alpha=0.7)
    ax.add_patch(arc_usage)
    
    # Label
    ax.text(x, y - 0.7, f"${current}/${total}", ha='center', va='top',
            fontsize=10, color=COLOR_TEXT, fontweight='bold')

def create_frame(stage):
    """Create a single frame of the hierarchical flow animation."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor(COLOR_BG)
    fig.patch.set_facecolor(COLOR_BG)
    
    # Title
    ax.text(7, 9.5, "Hierarchical Agent Orchestration", ha='center',
            fontsize=16, fontweight='bold', color=COLOR_TEXT)
    
    # Frame 1: Input order
    if stage == 1:
        # Input box
        draw_box(ax, 4, 6.5, 6, 1.5, "", COLOR_ACCENT, alpha=0.3)
        ax.text(7, 7.8, "Catering Order Input", ha='center', va='top',
                fontsize=13, fontweight='bold', color=COLOR_TEXT)
        ax.text(7, 7.3, "15 pizzas", ha='center', fontsize=11, color=COLOR_TEXT)
        ax.text(7, 6.9, "3 time slots: 11am, 12pm, 1pm", ha='center', fontsize=11, color=COLOR_TEXT)
        ax.text(7, 6.5, "$200 budget", ha='center', fontsize=11, color=COLOR_TEXT)
    
    # Frame 2: Planner decomposes
    elif stage == 2:
        # Planner at top
        draw_box(ax, 5.5, 7.5, 3, 1, "Planner", COLOR_PLANNER)
        
        # Tree structure - 3 branches
        draw_arrow(ax, 6.5, 7.5, 2.5, 5.5)
        draw_arrow(ax, 7, 7.5, 7, 5.5)
        draw_arrow(ax, 7.5, 7.5, 11.5, 5.5)
        
        # Worker nodes (grayed out)
        draw_box(ax, 1.5, 4.5, 2, 1, "Worker 1\n11am batch", COLOR_WORKER, alpha=0.3)
        draw_box(ax, 6, 4.5, 2, 1, "Worker 2\n12pm batch", COLOR_WORKER, alpha=0.3)
        draw_box(ax, 10.5, 4.5, 2, 1, "Worker 3\n1pm batch", COLOR_WORKER, alpha=0.3)
        
        # Decomposition label
        ax.text(7, 6.2, "Decomposing into 3 parallel batches", ha='center',
                fontsize=11, color=COLOR_ACCENT, style='italic')
    
    # Frame 3: Workers execute in parallel
    elif stage == 3:
        # Planner (faded)
        draw_box(ax, 5.5, 7.5, 3, 1, "Planner", COLOR_PLANNER, alpha=0.3)
        
        # Active workers
        draw_box(ax, 1.5, 4.5, 2, 1, "Worker 1\n11am batch", COLOR_WORKER)
        draw_box(ax, 6, 4.5, 2, 1, "Worker 2\n12pm batch", COLOR_WORKER)
        draw_box(ax, 10.5, 4.5, 2, 1, "Worker 3\n1pm batch", COLOR_WORKER)
        
        # Progress bars
        draw_progress_bar(ax, 1.5, 3.8, 2, 0.3, 0.65, COLOR_WORKER)
        draw_progress_bar(ax, 6, 3.8, 2, 0.3, 0.50, COLOR_WORKER)
        draw_progress_bar(ax, 10.5, 3.8, 2, 0.3, 0.75, COLOR_WORKER)
        
        # Parallel execution label
        ax.text(7, 2.8, "Parallel Execution in Progress", ha='center',
                fontsize=11, color=COLOR_ACCENT, fontweight='bold')
    
    # Frame 4: Worker results bubble up
    elif stage == 4:
        # Planner (faded)
        draw_box(ax, 5.5, 7.5, 3, 1, "Planner", COLOR_PLANNER, alpha=0.3)
        
        # Workers with results
        draw_box(ax, 1.5, 4.5, 2, 1, "Worker 1\n✓ Complete", COLOR_WORKER)
        draw_box(ax, 6, 4.5, 2, 1, "Worker 2\n✓ Complete", COLOR_WORKER)
        draw_box(ax, 10.5, 4.5, 2, 1, "Worker 3\n✓ Complete", COLOR_WORKER)
        
        # Results
        ax.text(2.5, 3.5, "5 pizzas\n$60", ha='center', fontsize=10,
                color=COLOR_TEXT, bbox=dict(boxstyle='round', facecolor='#2a2a3e', alpha=0.8))
        ax.text(7, 3.5, "5 pizzas\n$65", ha='center', fontsize=10,
                color=COLOR_TEXT, bbox=dict(boxstyle='round', facecolor='#2a2a3e', alpha=0.8))
        ax.text(11.5, 3.5, "5 pizzas\n$60", ha='center', fontsize=10,
                color=COLOR_TEXT, bbox=dict(boxstyle='round', facecolor='#2a2a3e', alpha=0.8))
        
        # Arrows bubbling up
        draw_arrow(ax, 2.5, 3.2, 6.5, 2.5, COLOR_ACCENT)
        draw_arrow(ax, 7, 3.2, 7, 2.5, COLOR_ACCENT)
        draw_arrow(ax, 11.5, 3.2, 7.5, 2.5, COLOR_ACCENT)
    
    # Frame 5: Verifier checks constraints
    elif stage == 5:
        # Workers (faded)
        draw_box(ax, 1.5, 6, 2, 0.8, "W1: ✓", COLOR_WORKER, alpha=0.3)
        draw_box(ax, 6, 6, 2, 0.8, "W2: ✓", COLOR_WORKER, alpha=0.3)
        draw_box(ax, 10.5, 6, 2, 0.8, "W3: ✓", COLOR_WORKER, alpha=0.3)
        
        # Verifier active
        draw_box(ax, 5.5, 4, 3, 1.2, "Verifier", COLOR_VERIFIER)
        
        # Constraint checks
        ax.text(7, 2.8, "Checking Constraints:", ha='center',
                fontsize=12, fontweight='bold', color=COLOR_TEXT)
        
        ax.text(4, 2.2, "Total cost: $185 < $200", ha='left',
                fontsize=11, color=COLOR_TEXT)
        draw_checkmark(ax, 10.5, 2.2, size=0.25)
        
        ax.text(4, 1.7, "All time slots met (11am, 12pm, 1pm)", ha='left',
                fontsize=11, color=COLOR_TEXT)
        draw_checkmark(ax, 10.5, 1.7, size=0.25)
        
        # Budget gauge
        draw_budget_gauge(ax, 12, 4.6, 185, 200)
    
    # Frame 6: Success confirmation
    elif stage == 6:
        # All components faded
        draw_box(ax, 5.5, 7.5, 3, 1, "Planner", COLOR_PLANNER, alpha=0.2)
        draw_box(ax, 1.5, 5.5, 2, 0.8, "W1", COLOR_WORKER, alpha=0.2)
        draw_box(ax, 6, 5.5, 2, 0.8, "W2", COLOR_WORKER, alpha=0.2)
        draw_box(ax, 10.5, 5.5, 2, 0.8, "W3", COLOR_WORKER, alpha=0.2)
        draw_box(ax, 5.5, 3.5, 3, 1, "Verifier", COLOR_VERIFIER, alpha=0.2)
        
        # Success message
        draw_box(ax, 3.5, 4.5, 7, 2, "", COLOR_SUCCESS, alpha=0.3)
        ax.text(7, 5.8, "✓ Order Successfully Orchestrated", ha='center',
                fontsize=14, fontweight='bold', color=COLOR_SUCCESS)
        ax.text(7, 5.3, "15 pizzas delivered across 3 time slots", ha='center',
                fontsize=11, color=COLOR_TEXT)
        ax.text(7, 4.9, "Total cost: $185 (under budget)", ha='center',
                fontsize=11, color=COLOR_TEXT)
        
        # Large checkmarks
        draw_checkmark(ax, 5, 5.5, size=0.4, color=COLOR_SUCCESS)
        draw_checkmark(ax, 9, 5.5, size=0.4, color=COLOR_SUCCESS)
    
    return fig

def main():
    """Generate the hierarchical flow animation."""
    from PIL import Image
    import tempfile
    import os
    
    frames = []
    
    print("Generating hierarchical flow frames...")
    for stage in range(1, 7):
        print(f"  Frame {stage}/6")
        fig = create_frame(stage)
        
        # Save to temporary file and load with PIL
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
        fig.savefig(tmp_path, dpi=100, bbox_inches='tight', facecolor=COLOR_BG)
        plt.close(fig)
        img = Image.open(tmp_path).convert('RGB')
        frames.append(img)
        os.unlink(tmp_path)
    
    # Create GIF with 1s per frame
    print(f"Saving GIF to {OUTPUT_GIF}...")
    imageio.mimsave(OUTPUT_GIF, frames, duration=1.0, loop=0)
    print("✓ Complete!")

if __name__ == '__main__':
    main()
