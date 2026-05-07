"""
Generate tool selection decision tree animation: Cost-aware tool selection
Shows decision tree with branches for cache, DB, API, and human escalation.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Circle
import imageio
import os

# Output paths
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'img')
os.makedirs(OUT_DIR, exist_ok=True)
OUTPUT_GIF = os.path.join(OUT_DIR, 'ch11-tool-selection-decision-tree.gif')

# Color scheme
COLOR_BG = '#1a1a2e'
COLOR_NODE = '#4A90E2'       # Blue
COLOR_DECISION = '#F39C12'   # Orange
COLOR_ACTION = '#27AE60'     # Green
COLOR_ESCALATE = '#E74C3C'   # Red
COLOR_HIGHLIGHT = '#FFD93D'  # Yellow
COLOR_TEXT = '#FFFFFF'       # White text

def draw_rounded_box(ax, x, y, w, h, label, color, alpha=0.9, linewidth=2):
    """Draw a rounded rectangle (for tree nodes)."""
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.08",
        facecolor=color,
        edgecolor=COLOR_TEXT,
        linewidth=linewidth,
        alpha=alpha
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center',
            fontsize=10, fontweight='bold', color=COLOR_TEXT)

def draw_diamond(ax, x, y, w, h, label, color, alpha=0.9, linewidth=2):
    """Draw a diamond shape (for decision nodes)."""
    points = [(x, y + h/2), (x + w/2, y), (x, y - h/2), (x - w/2, y)]
    diamond = Polygon(points, facecolor=color, edgecolor=COLOR_TEXT,
                     linewidth=linewidth, alpha=alpha)
    ax.add_patch(diamond)
    ax.text(x, y, label, ha='center', va='center',
            fontsize=9, fontweight='bold', color=COLOR_TEXT)

def draw_arrow(ax, x1, y1, x2, y2, label='', color=COLOR_TEXT, linewidth=2):
    """Draw an arrow with optional label."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->,head_width=0.3,head_length=0.5',
        color=color,
        linewidth=linewidth,
        zorder=2
    )
    ax.add_patch(arrow)
    
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.2, mid_y, label, ha='left', va='bottom',
                fontsize=9, color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=COLOR_BG, alpha=0.8))

def draw_glow(ax, x, y, w, h, shape='box'):
    """Draw a glow effect around a node."""
    if shape == 'box':
        for i in range(3):
            glow = FancyBboxPatch(
                (x - w/2 - 0.1*i, y - h/2 - 0.1*i),
                w + 0.2*i, h + 0.2*i,
                boxstyle="round,pad=0.08",
                facecolor='none',
                edgecolor=COLOR_HIGHLIGHT,
                linewidth=3 - i,
                alpha=0.4 - i*0.1,
                zorder=1
            )
            ax.add_patch(glow)
    elif shape == 'diamond':
        for i in range(3):
            scale = 1 + 0.15*i
            points = [
                (x, y + h/2 * scale),
                (x + w/2 * scale, y),
                (x, y - h/2 * scale),
                (x - w/2 * scale, y)
            ]
            glow = Polygon(points, facecolor='none', edgecolor=COLOR_HIGHLIGHT,
                         linewidth=3 - i, alpha=0.4 - i*0.1, zorder=1)
            ax.add_patch(glow)

def create_frame(stage):
    """Create a single frame of the decision tree animation."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor(COLOR_BG)
    fig.patch.set_facecolor(COLOR_BG)
    
    # Title
    ax.text(7, 9.5, "Tool Selection Decision Tree", ha='center',
            fontsize=16, fontweight='bold', color=COLOR_TEXT)
    ax.text(7, 9.1, "Cost-Aware Inventory Count Strategy",
            fontsize=12, color=COLOR_TEXT, style='italic')
    
    # Root node
    root_x, root_y = 7, 7.5
    root_alpha = 0.9 if stage == 1 else 0.4
    draw_rounded_box(ax, root_x, root_y, 2.5, 0.8, "Need Inventory\nCount",
                    COLOR_NODE, alpha=root_alpha)
    if stage == 1:
        draw_glow(ax, root_x, root_y, 2.5, 0.8, shape='box')
    
    # Branch 1: Cache path
    cache_decision_x, cache_decision_y = 3, 5.5
    cache_action_x, cache_action_y = 3, 3.5
    
    # Decision diamond
    cache_alpha = 0.9 if stage == 2 else 0.4
    draw_diamond(ax, cache_decision_x, cache_decision_y, 1.8, 1.2,
                "Cached\nestimate\navailable?", COLOR_DECISION, alpha=cache_alpha)
    if stage == 2:
        draw_glow(ax, cache_decision_x, cache_decision_y, 1.8, 1.2, shape='diamond')
    
    # Action box
    cache_action_alpha = 0.9 if stage == 2 else 0.4
    draw_rounded_box(ax, cache_action_x, cache_action_y, 2, 0.8,
                    "Use Cache\nfree, 10ms", COLOR_ACTION, alpha=cache_action_alpha)
    
    # Arrows
    arrow_color = COLOR_HIGHLIGHT if stage == 2 else COLOR_TEXT
    draw_arrow(ax, root_x - 1, root_y - 0.5, cache_decision_x + 0.5, cache_decision_y + 0.5,
              '', arrow_color)
    draw_arrow(ax, cache_decision_x, cache_decision_y - 0.6, cache_action_x, cache_action_y + 0.4,
              'Yes', arrow_color if stage == 2 else COLOR_TEXT)
    
    # Branch 2: DB path
    db_decision_x, db_decision_y = 7, 5.5
    db_action_x, db_action_y = 7, 3.5
    
    db_alpha = 0.9 if stage == 3 else 0.4
    draw_diamond(ax, db_decision_x, db_decision_y, 1.8, 1.2,
                "Cache\nstale?", COLOR_DECISION, alpha=db_alpha)
    if stage == 3:
        draw_glow(ax, db_decision_x, db_decision_y, 1.8, 1.2, shape='diamond')
    
    db_action_alpha = 0.9 if stage == 3 else 0.4
    draw_rounded_box(ax, db_action_x, db_action_y, 2, 0.8,
                    "Query DB\n$0.0001, 50ms", COLOR_ACTION, alpha=db_action_alpha)
    
    arrow_color = COLOR_HIGHLIGHT if stage == 3 else COLOR_TEXT
    draw_arrow(ax, root_x, root_y - 0.5, db_decision_x, db_decision_y + 0.6,
              '', arrow_color)
    draw_arrow(ax, db_decision_x, db_decision_y - 0.6, db_action_x, db_action_y + 0.4,
              'Yes', arrow_color if stage == 3 else COLOR_TEXT)
    
    # Branch 3: API path
    api_decision_x, api_decision_y = 11, 5.5
    api_action_x, api_action_y = 11, 3.5
    
    api_alpha = 0.9 if stage == 4 else 0.4
    draw_diamond(ax, api_decision_x, api_decision_y, 1.8, 1.2,
                "DB\ntimeout?", COLOR_DECISION, alpha=api_alpha)
    if stage == 4:
        draw_glow(ax, api_decision_x, api_decision_y, 1.8, 1.2, shape='diamond')
    
    api_action_alpha = 0.9 if stage == 4 else 0.4
    draw_rounded_box(ax, api_action_x, api_action_y, 2.2, 0.8,
                    "Call API\n$0.001, 200ms", COLOR_ACTION, alpha=api_action_alpha)
    
    arrow_color = COLOR_HIGHLIGHT if stage == 4 else COLOR_TEXT
    draw_arrow(ax, root_x + 1, root_y - 0.5, api_decision_x - 0.5, api_decision_y + 0.5,
              '', arrow_color)
    draw_arrow(ax, api_decision_x, api_decision_y - 0.6, api_action_x, api_action_y + 0.4,
              'Yes', arrow_color if stage == 4 else COLOR_TEXT)
    
    # Branch 4: Escalate path
    escalate_decision_x, escalate_decision_y = 11, 1.8
    escalate_action_x, escalate_action_y = 11, 0.5
    
    escalate_alpha = 0.9 if stage == 5 else 0.4
    draw_diamond(ax, escalate_decision_x, escalate_decision_y, 1.8, 1.2,
                "API\nunavailable?", COLOR_DECISION, alpha=escalate_alpha)
    if stage == 5:
        draw_glow(ax, escalate_decision_x, escalate_decision_y, 1.8, 1.2, shape='diamond')
    
    escalate_action_alpha = 0.9 if stage == 5 else 0.4
    draw_rounded_box(ax, escalate_action_x, escalate_action_y, 2.2, 0.6,
                    "Escalate to Human", COLOR_ESCALATE, alpha=escalate_action_alpha)
    
    arrow_color = COLOR_HIGHLIGHT if stage == 5 else COLOR_TEXT
    draw_arrow(ax, api_action_x, api_action_y - 0.4, escalate_decision_x, escalate_decision_y + 0.6,
              '', arrow_color if stage == 5 else COLOR_TEXT)
    draw_arrow(ax, escalate_decision_x, escalate_decision_y - 0.6, escalate_action_x, escalate_action_y + 0.3,
              'Yes', arrow_color if stage == 5 else COLOR_TEXT)
    
    # Legend
    legend_x = 0.5
    legend_items = [
        (COLOR_NODE, 'Start Node'),
        (COLOR_DECISION, 'Decision'),
        (COLOR_ACTION, 'Action'),
        (COLOR_ESCALATE, 'Escalation')
    ]
    
    ax.text(legend_x, 1.8, "Legend:", fontsize=10, fontweight='bold', color=COLOR_TEXT)
    for i, (color, label) in enumerate(legend_items):
        y = 1.5 - i * 0.25
        circle = Circle((legend_x + 0.15, y), 0.1, facecolor=color, edgecolor=COLOR_TEXT, linewidth=1)
        ax.add_patch(circle)
        ax.text(legend_x + 0.35, y, label, ha='left', va='center',
                fontsize=9, color=COLOR_TEXT)
    
    return fig

def main():
    """Generate the tool selection decision tree animation."""
    from PIL import Image
    import tempfile
    
    frames = []
    
    print("Generating tool selection decision tree frames...")
    for stage in range(1, 6):
        print(f"  Frame {stage}/5")
        fig = create_frame(stage)
        
        # Save to temporary file and load with PIL
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
        fig.savefig(tmp_path, dpi=100, bbox_inches='tight', facecolor=COLOR_BG)
        plt.close(fig)
        img = Image.open(tmp_path).convert('RGB')
        frames.append(img)
        import os as _os
        _os.unlink(tmp_path)
    
    # Create GIF with 1s per frame
    print(f"Saving GIF to {OUTPUT_GIF}...")
    imageio.mimsave(OUTPUT_GIF, frames, duration=1.0, loop=0)
    print("✓ Complete!")

if __name__ == '__main__':
    main()
