"""
Generate reflection loop animation: Generate → Critique → Revise
Shows contradiction detection and resolution with color coding.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import imageio
import os

# Output paths
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'img')
os.makedirs(OUT_DIR, exist_ok=True)
OUTPUT_GIF = os.path.join(OUT_DIR, 'ch11-reflection-loop.gif')

# Color scheme (dark theme)
BG = '#1a1a2e'
GRID_C = '#2d3748'
TEXT_C = '#e2e8f0'
COLOR_ERROR = '#dc2626'      # Red
COLOR_CRITIQUE = '#fbbf24'   # Yellow
COLOR_SUCCESS = '#10b981'    # Green
COLOR_NEUTRAL = '#64748b'    # Gray

def draw_box(ax, x, y, w, h, label, color, alpha=0.9):
    """Draw a rounded box with label."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05",
        facecolor=color,
        edgecolor=TEXT_C,
        linewidth=2,
        alpha=alpha
    )
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, label,
            ha='center', va='center',
            fontsize=11, fontweight='bold',
            color=TEXT_C if alpha < 0.5 else BG)

def draw_arrow(ax, x1, y1, x2, y2, color=None):
    """Draw a curved arrow between points."""
    if color is None:
        color = TEXT_C
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->,head_width=0.4,head_length=0.8',
        color=color,
        linewidth=2.5,
        connectionstyle="arc3,rad=0.2"
    )
    ax.add_patch(arrow)

def create_frame(stage):
    """Create a single frame of the reflection loop animation."""
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    title = "Reflection Pattern: Self-Critique Loop"
    ax.text(6, 9.5, title, ha='center', va='top',
            fontsize=16, fontweight='bold', color=TEXT_C)
    
    # Frame 1: Customer query
    if stage == 1:
        # Query box at top
        query_box = FancyBboxPatch(
            (2, 8), 8, 0.8,
            boxstyle="round,pad=0.1",
            facecolor='#374151',
            edgecolor=TEXT_C,
            linewidth=2
        )
        ax.add_patch(query_box)
        ax.text(6, 8.4, '🛒 Customer Query:', ha='center', va='center',
                fontsize=11, fontweight='bold', color=TEXT_C)
        ax.text(6, 8.05, '"Gluten-free dairy-free pizza with extra cheese"', ha='center', va='center',
                fontsize=11, color=TEXT_C, style='italic')
        
        # Process boxes (inactive)
        draw_box(ax, 1.5, 5, 2.5, 1.2, "Generate", COLOR_NEUTRAL, alpha=0.3)
        draw_box(ax, 4.8, 5, 2.5, 1.2, "Critique", COLOR_NEUTRAL, alpha=0.3)
        draw_box(ax, 8.1, 5, 2.5, 1.2, "Revise", COLOR_NEUTRAL, alpha=0.3)
        
        ax.text(6, 3.2, 'Processing query...', ha='center',
                fontsize=13, color=TEXT_C, style='italic')
    
    # Frame 2: Draft response with contradiction
    elif stage == 2:
        # Draft response box with error
        response_box = FancyBboxPatch(
            (2, 7.5), 8, 1.2,
            boxstyle="round,pad=0.1",
            facecolor='#7f1d1d',
            edgecolor=COLOR_ERROR,
            linewidth=3
        )
        ax.add_patch(response_box)
        ax.text(6, 8.4, '❌ Draft Response (CONTRADICTION):', ha='center', va='center',
                fontsize=11, fontweight='bold', color=COLOR_ERROR)
        ax.text(6, 8, '"Here\'s your gluten-free dairy-free pizza', ha='center', va='center',
                fontsize=10, color=TEXT_C)
        ax.text(6, 7.7, 'with extra cheddar cheese"', ha='center', va='center',
                fontsize=10, color=TEXT_C)
        
        # Highlight contradiction
        error_marker = Circle((9.5, 8.1), 0.3, color=COLOR_ERROR, zorder=10)
        ax.add_patch(error_marker)
        ax.text(9.5, 8.1, "!", ha='center', va='center',
                fontsize=18, fontweight='bold', color='white')
        
        # Process boxes
        draw_box(ax, 1.5, 5, 2.5, 1.2, "Generate", COLOR_ERROR, alpha=0.7)
        draw_box(ax, 4.8, 5, 2.5, 1.2, "Critique", COLOR_NEUTRAL, alpha=0.3)
        draw_box(ax, 8.1, 5, 2.5, 1.2, "Revise", COLOR_NEUTRAL, alpha=0.3)
        
        ax.text(6, 3.5, '⚠️ dairy-free ≠ cheddar cheese', ha='center',
                fontsize=12, color=COLOR_ERROR, fontweight='bold')
        ax.text(6, 3, '(Cheddar contains dairy)', ha='center',
                fontsize=10, color=TEXT_C, style='italic')
    
    # Frame 3: Self-critique bubble
    elif stage == 3:
        # Critique bubble
        critique_box = FancyBboxPatch(
            (2, 7.5), 8, 1.2,
            boxstyle="round,pad=0.1",
            facecolor='#78350f',
            edgecolor=COLOR_CRITIQUE,
            linewidth=3
        )
        ax.add_patch(critique_box)
        ax.text(6, 8.5, '🔍 Self-Critique:', ha='center', va='center',
                fontsize=11, fontweight='bold', color=COLOR_CRITIQUE)
        ax.text(6, 8.1, '"Detected contradiction: dairy-free + extra cheese', ha='center', va='center',
                fontsize=10, color=TEXT_C)
        ax.text(6, 7.8, 'Cheddar is a dairy product. Must use vegan cheese."', ha='center', va='center',
                fontsize=10, color=TEXT_C)
        
        # Process boxes
        draw_box(ax, 1.5, 5, 2.5, 1.2, "Generate", COLOR_ERROR, alpha=0.4)
        draw_box(ax, 4.8, 5, 2.5, 1.2, "Critique", COLOR_CRITIQUE, alpha=0.8)
        draw_box(ax, 8.1, 5, 2.5, 1.2, "Revise", COLOR_NEUTRAL, alpha=0.3)
        draw_arrow(ax, 4, 5.6, 4.8, 5.6)
        
        ax.text(6, 3.2, 'Analyzing response for contradictions...', ha='center',
                fontsize=11, color=TEXT_C, style='italic')
    
    # Frame 4: Revised response with vegan cheese
    elif stage == 4:
        # Revised response box
        response_box = FancyBboxPatch(
            (2, 7.5), 8, 1.2,
            boxstyle="round,pad=0.1",
            facecolor='#064e3b',
            edgecolor=COLOR_SUCCESS,
            linewidth=3
        )
        ax.add_patch(response_box)
        ax.text(6, 8.4, '✅ Revised Response (RESOLVED):', ha='center', va='center',
                fontsize=11, fontweight='bold', color=COLOR_SUCCESS)
        ax.text(6, 8, '"Here\'s your gluten-free dairy-free pizza', ha='center', va='center',
                fontsize=10, color=TEXT_C)
        ax.text(6, 7.7, 'with extra vegan cheese"', ha='center', va='center',
                fontsize=10, color=TEXT_C)
        
        # Success marker
        success_marker = Circle((9.5, 8.1), 0.3, color=COLOR_SUCCESS, zorder=10)
        ax.add_patch(success_marker)
        ax.text(9.5, 8.1, "✓", ha='center', va='center',
                fontsize=16, fontweight='bold', color='white')
        
        # Process boxes
        draw_box(ax, 1.5, 5, 2.5, 1.2, "Generate", COLOR_ERROR, alpha=0.3)
        draw_box(ax, 4.8, 5, 2.5, 1.2, "Critique", COLOR_CRITIQUE, alpha=0.4)
        draw_box(ax, 8.1, 5, 2.5, 1.2, "Revise", COLOR_SUCCESS, alpha=0.8)
        draw_arrow(ax, 4, 5.6, 4.8, 5.6)
        draw_arrow(ax, 7.3, 5.6, 8.1, 5.6)
        
        ax.text(6, 3.5, '✓ Contradiction resolved', ha='center',
                fontsize=12, color=COLOR_SUCCESS, fontweight='bold')
        ax.text(6, 3, '(Vegan cheese is dairy-free)', ha='center',
                fontsize=10, color=TEXT_C, style='italic')
    
    # Frame 5: Side-by-side comparison
    elif stage == 5:
        # Left side: Single-pass FAIL
        fail_box = FancyBboxPatch(
            (0.5, 6), 5, 2.5,
            boxstyle="round,pad=0.1",
            facecolor='#7f1d1d',
            edgecolor=COLOR_ERROR,
            linewidth=3
        )
        ax.add_patch(fail_box)
        ax.text(3, 8.2, '❌ Single-Pass (FAIL)', ha='center', va='center',
                fontsize=12, fontweight='bold', color=COLOR_ERROR)
        ax.text(3, 7.6, '"...gluten-free dairy-free', ha='center', va='center',
                fontsize=9, color=TEXT_C)
        ax.text(3, 7.3, 'pizza with extra', ha='center', va='center',
                fontsize=9, color=TEXT_C)
        ax.text(3, 7, 'cheddar cheese"', ha='center', va='center',
                fontsize=9, color=TEXT_C)
        ax.text(3, 6.5, '⚠️ Contradiction', ha='center', va='center',
                fontsize=10, color=COLOR_ERROR, fontweight='bold')
        
        # Right side: Reflection SUCCESS
        success_box = FancyBboxPatch(
            (6.5, 6), 5, 2.5,
            boxstyle="round,pad=0.1",
            facecolor='#064e3b',
            edgecolor=COLOR_SUCCESS,
            linewidth=3
        )
        ax.add_patch(success_box)
        ax.text(9, 8.2, '✅ Reflection (SUCCESS)', ha='center', va='center',
                fontsize=12, fontweight='bold', color=COLOR_SUCCESS)
        ax.text(9, 7.6, '"...gluten-free dairy-free', ha='center', va='center',
                fontsize=9, color=TEXT_C)
        ax.text(9, 7.3, 'pizza with extra', ha='center', va='center',
                fontsize=9, color=TEXT_C)
        ax.text(9, 7, 'vegan cheese"', ha='center', va='center',
                fontsize=9, color=TEXT_C)
        ax.text(9, 6.5, '✓ Consistent', ha='center', va='center',
                fontsize=10, color=COLOR_SUCCESS, fontweight='bold')
        
        # Bottom comparison
        ax.text(6, 4.5, 'Why Reflection Matters:', ha='center',
                fontsize=14, fontweight='bold', color=TEXT_C)
        ax.text(6, 3.8, 'Self-critique catches logical contradictions that single-pass generation misses', ha='center',
                fontsize=11, color=TEXT_C)
        ax.text(6, 3.3, 'Cost: 3× tokens  •  Quality: ↑ 85% error reduction', ha='center',
                fontsize=10, color=TEXT_C, style='italic')
    
    # Key legend (always visible)
    key_y = 1.2
    ax.text(2, key_y, "●", color=COLOR_ERROR, fontsize=16, ha='center')
    ax.text(2.8, key_y, "Error", fontsize=9, va='center', color=TEXT_C)
    ax.text(4.5, key_y, "●", color=COLOR_CRITIQUE, fontsize=16, ha='center')
    ax.text(5.5, key_y, "Critique", fontsize=9, va='center', color=TEXT_C)
    ax.text(7.5, key_y, "●", color=COLOR_SUCCESS, fontsize=16, ha='center')
    ax.text(8.4, key_y, "Success", fontsize=9, va='center', color=TEXT_C)
    
    return fig

# Generate frames
frames = []
frame_durations = []

# 5 frames as specified
stages_with_timing = [
    (1, 0.8),  # Frame 1: Customer query
    (2, 0.8),  # Frame 2: Draft response with contradiction
    (3, 0.8),  # Frame 3: Self-critique bubble
    (4, 0.8),  # Frame 4: Revised response
    (5, 1.2),  # Frame 5: Side-by-side comparison (longer)
]

for stage, duration in stages_with_timing:
    fig = create_frame(stage)
    fname = os.path.join(OUT_DIR, f'reflection_frame_{len(frames):03d}.png')
    fig.tight_layout()
    fig.savefig(fname, dpi=100, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    frames.append(imageio.v2.imread(fname))
    frame_durations.append(duration)

# Create GIF
imageio.mimsave(OUTPUT_GIF, frames, duration=frame_durations, loop=0)
print(f'✓ Saved {OUTPUT_GIF}')

# Clean up temp frames
for i in range(len(frames)):
    fname = os.path.join(OUT_DIR, f'reflection_frame_{i:03d}.png')
    if os.path.exists(fname):
        os.remove(fname)

print(f'Generated {len(frames)} frames')
print(f'Total animation duration: {sum(frame_durations):.1f}s')
