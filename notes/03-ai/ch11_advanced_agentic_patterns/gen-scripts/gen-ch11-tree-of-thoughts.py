"""
Generate tree of thoughts animation: Branching reasoning tree
Shows multiple paths explored with best path highlighted after evaluation.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import imageio
import os

# Output paths
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'img')
os.makedirs(OUT_DIR, exist_ok=True)
OUTPUT_GIF = os.path.join(OUT_DIR, 'ch11_tree_of_thoughts.gif')

# Color scheme
COLOR_ROOT = '#9B59B6'       # Purple
COLOR_BRANCH = '#3498DB'     # Blue
COLOR_GOOD = '#27AE60'       # Green
COLOR_BAD = '#E74C3C'        # Red
COLOR_BEST = '#F39C12'       # Orange/Gold
COLOR_TEXT = '#2C3E50'       # Dark blue-gray
COLOR_BG = '#ECF0F1'         # Light gray

def draw_thought_node(ax, x, y, label, score, color, active=False, size='normal'):
    """Draw a thought node with score."""
    r = 0.4 if size == 'normal' else 0.5
    alpha = 1.0 if active else 0.5
    linewidth = 3 if active else 1.5
    
    circle = Circle((x, y), r, facecolor=color, edgecolor=COLOR_TEXT,
                   linewidth=linewidth, alpha=alpha, zorder=5)
    ax.add_patch(circle)
    
    # Label
    ax.text(x, y + 0.15, label, ha='center', va='center',
            fontsize=8 if size == 'normal' else 10, fontweight='bold',
            color='white')
    
    # Score
    if score is not None:
        ax.text(x, y - 0.15, f"{score:.1f}", ha='center', va='center',
                fontsize=7 if size == 'normal' else 9,
                color='white', style='italic')

def draw_edge(ax, x1, y1, x2, y2, color=COLOR_TEXT, width=2, style='solid'):
    """Draw an edge between nodes."""
    arrow = FancyArrowPatch(
        (x1, y1 - 0.4), (x2, y2 + 0.4),
        arrowstyle='-',
        color=color,
        linewidth=width,
        linestyle=style
    )
    ax.add_patch(arrow)

def create_frame(stage):
    """Create a single frame of the tree of thoughts animation."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor(COLOR_BG)
    fig.patch.set_facecolor(COLOR_BG)
    
    # Title
    ax.text(7, 9.5, "Tree of Thoughts: Branching Reasoning Exploration", ha='center',
            fontsize=18, fontweight='bold', color=COLOR_TEXT)
    
    # Problem statement
    problem_box = FancyBboxPatch(
        (2, 8.3), 10, 0.8,
        boxstyle="round,pad=0.05",
        facecolor='white',
        edgecolor=COLOR_TEXT,
        linewidth=1.5
    )
    ax.add_patch(problem_box)
    ax.text(7, 8.7, '🍕 Problem: Optimize ingredient combinations for 3 specialty pizzas',
            ha='center', va='center', fontsize=10, color=COLOR_TEXT)
    
    # Tree structure positions
    root_x, root_y = 7, 7
    
    # Level 1 (3 branches)
    l1_positions = [(3.5, 5.5), (7, 5.5), (10.5, 5.5)]
    l1_labels = ['A', 'B', 'C']
    l1_scores = [6.5, 7.8, 5.2]
    
    # Level 2 (2 branches per L1 node - only showing from best L1)
    l2_positions = [(5.5, 3.5), (8.5, 3.5)]
    l2_labels = ['B1', 'B2']
    l2_scores = [8.1, 7.3]
    
    # Level 3 (2 branches from best L2)
    l3_positions = [(4.5, 1.5), (6.5, 1.5)]
    l3_labels = ['B1a', 'B1b']
    l3_scores = [8.9, 7.6]
    
    # Stage 0: Just root
    if stage == 0:
        draw_thought_node(ax, root_x, root_y, "Root", None, COLOR_ROOT, active=True, size='large')
        ax.text(7, 6.2, "Initial state", ha='center', va='top',
                fontsize=9, color=COLOR_TEXT)
        
        ax.text(7, 4, "Generating candidate approaches...", ha='center',
                fontsize=12, color=COLOR_TEXT, style='italic')
    
    # Stage 1: Level 1 branches generated
    elif stage == 1:
        draw_thought_node(ax, root_x, root_y, "Root", None, COLOR_ROOT, size='large')
        
        # Draw edges
        for (x, y) in l1_positions:
            draw_edge(ax, root_x, root_y, x, y, style='dashed')
        
        # Draw L1 nodes (not evaluated yet)
        for (x, y), label in zip(l1_positions, l1_labels):
            draw_thought_node(ax, x, y, label, None, COLOR_BRANCH, active=True)
        
        ax.text(7, 4.5, "Generated 3 candidate approaches", ha='center',
                fontsize=11, color=COLOR_BRANCH, fontweight='bold')
    
    # Stage 2: Level 1 evaluated
    elif stage == 2:
        draw_thought_node(ax, root_x, root_y, "Root", None, COLOR_ROOT, size='large')
        
        # Draw edges
        for (x, y) in l1_positions:
            draw_edge(ax, root_x, root_y, x, y)
        
        # Draw L1 nodes with scores
        colors = [COLOR_GOOD if s > 6 else COLOR_BAD for s in l1_scores]
        for (x, y), label, score, color in zip(l1_positions, l1_labels, l1_scores, colors):
            draw_thought_node(ax, x, y, label, score, color, active=True)
        
        # Highlight best
        best_idx = l1_scores.index(max(l1_scores))
        best_x, best_y = l1_positions[best_idx]
        circle_highlight = Circle((best_x, best_y), 0.6, 
                                 facecolor='none', edgecolor=COLOR_BEST,
                                 linewidth=3, linestyle='--', zorder=4)
        ax.add_patch(circle_highlight)
        
        ax.text(7, 4.5, f"Best L1: {l1_labels[best_idx]} (score: {l1_scores[best_idx]})", ha='center',
                fontsize=11, color=COLOR_BEST, fontweight='bold')
        ax.text(7, 4, "Expanding best branch...", ha='center',
                fontsize=10, color=COLOR_TEXT, style='italic')
    
    # Stage 3: Level 2 branches from best L1
    elif stage == 3:
        draw_thought_node(ax, root_x, root_y, "Root", None, COLOR_ROOT, size='large')
        
        # L1 edges
        for (x, y) in l1_positions:
            alpha = 1.0 if x == 7 else 0.3
            draw_edge(ax, root_x, root_y, x, y)
        
        # L1 nodes
        colors = [COLOR_GOOD if s > 6 else COLOR_BAD for s in l1_scores]
        for i, ((x, y), label, score, color) in enumerate(zip(l1_positions, l1_labels, l1_scores, colors)):
            active = (i == 1)  # B is active
            draw_thought_node(ax, x, y, label, score, color if active else COLOR_BRANCH, active=active)
        
        # L2 edges from B (middle node)
        for (x, y) in l2_positions:
            draw_edge(ax, l1_positions[1][0], l1_positions[1][1], x, y, style='dashed')
        
        # L2 nodes
        for (x, y), label in zip(l2_positions, l2_labels):
            draw_thought_node(ax, x, y, label, None, COLOR_BRANCH, active=True)
        
        ax.text(7, 2.5, "Expanding from B: generated 2 refinements", ha='center',
                fontsize=10, color=COLOR_BRANCH, fontweight='bold')
    
    # Stage 4: Level 2 evaluated
    elif stage == 4:
        draw_thought_node(ax, root_x, root_y, "Root", None, COLOR_ROOT, size='large')
        
        # L1 edges
        for (x, y) in l1_positions:
            draw_edge(ax, root_x, root_y, x, y)
        
        # L1 nodes (faded except B)
        colors = [COLOR_GOOD if s > 6 else COLOR_BAD for s in l1_scores]
        for i, ((x, y), label, score, color) in enumerate(zip(l1_positions, l1_labels, l1_scores, colors)):
            draw_thought_node(ax, x, y, label, score, color)
        
        # L2 edges
        for (x, y) in l2_positions:
            draw_edge(ax, l1_positions[1][0], l1_positions[1][1], x, y)
        
        # L2 nodes with scores
        l2_colors = [COLOR_GOOD if s > 7.5 else COLOR_BRANCH for s in l2_scores]
        for (x, y), label, score, color in zip(l2_positions, l2_labels, l2_scores, l2_colors):
            draw_thought_node(ax, x, y, label, score, color, active=True)
        
        # Highlight best L2
        best_l2_idx = l2_scores.index(max(l2_scores))
        best_x, best_y = l2_positions[best_l2_idx]
        circle_highlight = Circle((best_x, best_y), 0.6,
                                 facecolor='none', edgecolor=COLOR_BEST,
                                 linewidth=3, linestyle='--', zorder=4)
        ax.add_patch(circle_highlight)
        
        ax.text(7, 2.5, f"Best L2: {l2_labels[best_l2_idx]} (score: {l2_scores[best_l2_idx]})", ha='center',
                fontsize=11, color=COLOR_BEST, fontweight='bold')
    
    # Stage 5: Level 3 expansion
    elif stage == 5:
        draw_thought_node(ax, root_x, root_y, "Root", None, COLOR_ROOT, size='large')
        
        # All previous edges and nodes (faded)
        for (x, y) in l1_positions:
            draw_edge(ax, root_x, root_y, x, y)
        colors = [COLOR_GOOD if s > 6 else COLOR_BAD for s in l1_scores]
        for ((x, y), label, score, color) in zip(l1_positions, l1_labels, l1_scores, colors):
            draw_thought_node(ax, x, y, label, score, color)
        
        for (x, y) in l2_positions:
            draw_edge(ax, l1_positions[1][0], l1_positions[1][1], x, y)
        l2_colors = [COLOR_GOOD if s > 7.5 else COLOR_BRANCH for s in l2_scores]
        for i, ((x, y), label, score, color) in enumerate(zip(l2_positions, l2_labels, l2_scores, l2_colors)):
            active = (i == 0)
            draw_thought_node(ax, x, y, label, score, color, active=active)
        
        # L3 edges from B1
        for (x, y) in l3_positions:
            draw_edge(ax, l2_positions[0][0], l2_positions[0][1], x, y, style='dashed')
        
        # L3 nodes with scores
        l3_colors = [COLOR_GOOD if s > 8.5 else COLOR_BRANCH for s in l3_scores]
        for (x, y), label, score, color in zip(l3_positions, l3_labels, l3_scores, l3_colors):
            draw_thought_node(ax, x, y, label, score, color, active=True)
        
        ax.text(7, 0.5, "Final refinements evaluated", ha='center',
                fontsize=10, color=COLOR_TEXT, fontweight='bold')
    
    # Stage 6: Best path highlighted
    elif stage == 6:
        draw_thought_node(ax, root_x, root_y, "Root", None, COLOR_ROOT, size='large')
        
        # All edges
        for (x, y) in l1_positions:
            color = COLOR_BEST if x == 7 else COLOR_TEXT
            width = 4 if x == 7 else 1.5
            draw_edge(ax, root_x, root_y, x, y, color=color, width=width)
        
        # All L1 nodes
        colors = [COLOR_GOOD if s > 6 else COLOR_BAD for s in l1_scores]
        for i, ((x, y), label, score, color) in enumerate(zip(l1_positions, l1_labels, l1_scores, colors)):
            highlight = (i == 1)
            draw_thought_node(ax, x, y, label, score, COLOR_BEST if highlight else color)
        
        # L2 edges
        for i, (x, y) in enumerate(l2_positions):
            color = COLOR_BEST if i == 0 else COLOR_TEXT
            width = 4 if i == 0 else 1.5
            draw_edge(ax, l1_positions[1][0], l1_positions[1][1], x, y, color=color, width=width)
        
        # L2 nodes
        l2_colors = [COLOR_GOOD if s > 7.5 else COLOR_BRANCH for s in l2_scores]
        for i, ((x, y), label, score, color) in enumerate(zip(l2_positions, l2_labels, l2_scores, l2_colors)):
            highlight = (i == 0)
            draw_thought_node(ax, x, y, label, score, COLOR_BEST if highlight else color)
        
        # L3 edges
        for i, (x, y) in enumerate(l3_positions):
            color = COLOR_BEST if i == 0 else COLOR_TEXT
            width = 4 if i == 0 else 1.5
            draw_edge(ax, l2_positions[0][0], l2_positions[0][1], x, y, color=color, width=width)
        
        # L3 nodes
        l3_colors = [COLOR_GOOD if s > 8.5 else COLOR_BRANCH for s in l3_scores]
        for i, ((x, y), label, score, color) in enumerate(zip(l3_positions, l3_labels, l3_scores, l3_colors)):
            highlight = (i == 0)
            draw_thought_node(ax, x, y, label, score, COLOR_BEST if highlight else color)
        
        # Best path indicator
        result_box = FancyBboxPatch(
            (3.5, 0.2), 7, 0.6,
            boxstyle="round,pad=0.08",
            facecolor=COLOR_BEST,
            edgecolor=COLOR_TEXT,
            linewidth=3
        )
        ax.add_patch(result_box)
        ax.text(7, 0.5, "✓ Best Path: Root → B → B1 → B1a (score: 8.9)", ha='center',
                fontsize=11, fontweight='bold', color='white')
    
    # Stats box
    if stage >= 1:
        stats_text = f"Nodes explored: {1 if stage == 0 else 4 if stage <= 2 else 6 if stage <= 4 else 9}"
        ax.text(12, 0.5, stats_text, ha='right', fontsize=9,
                color=COLOR_TEXT, style='italic')
    
    return fig

# Generate frames
frames = []
frame_durations = []

stages_with_timing = [
    (0, 1.0),  # Root
    (1, 1.3),  # L1 generated
    (2, 1.5),  # L1 evaluated
    (3, 1.3),  # L2 generated
    (4, 1.5),  # L2 evaluated
    (5, 1.3),  # L3 generated
    (6, 2.0),  # Best path
]

for stage, duration in stages_with_timing:
    fig = create_frame(stage)
    fname = os.path.join(OUT_DIR, f'tot_frame_{len(frames):03d}.png')
    fig.tight_layout()
    fig.savefig(fname, dpi=100, bbox_inches='tight', facecolor=COLOR_BG)
    plt.close(fig)
    frames.append(imageio.v2.imread(fname))
    frame_durations.append(duration)

# Create GIF
imageio.mimsave(OUTPUT_GIF, frames, duration=frame_durations, loop=0)
print(f'✓ Saved {OUTPUT_GIF}')

# Clean up temp frames
for i in range(len(frames)):
    fname = os.path.join(OUT_DIR, f'tot_frame_{i:03d}.png')
    if os.path.exists(fname):
        os.remove(fname)

print(f'Generated {len(frames)} frames')
print(f'Total animation duration: {sum(frame_durations):.1f}s')
