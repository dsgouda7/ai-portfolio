"""
Generate debate flow animation: Multi-agent debate with speech bubbles
Shows 2 agents proposing solutions, 1 judge deciding, with RAG lookup.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Wedge, Rectangle
import imageio
import os

# Output paths
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'img')
os.makedirs(OUT_DIR, exist_ok=True)
OUTPUT_GIF = os.path.join(OUT_DIR, 'ch11-debate-flow.gif')

# Color scheme (dark theme)
BG = '#1a1a2e'
GRID_C = '#2d3748'
TEXT_C = '#e2e8f0'
COLOR_AGENT1 = '#3b82f6'     # Blue
COLOR_AGENT2 = '#f59e0b'     # Orange
COLOR_JUDGE = '#8b5cf6'      # Purple
COLOR_RAG = '#fbbf24'        # Yellow
COLOR_SUCCESS = '#10b981'    # Green
COLOR_ERROR = '#dc2626'      # Red

def draw_agent(ax, x, y, color, label, active=False):
    """Draw an agent as a circle with label."""
    alpha = 1.0 if active else 0.4
    linewidth = 3 if active else 1.5
    
    # Head
    circle = Circle((x, y), 0.4, facecolor=color, edgecolor=TEXT_C,
                   linewidth=linewidth, alpha=alpha, zorder=5)
    ax.add_patch(circle)
    
    # Label below
    ax.text(x, y - 0.7, label, ha='center', va='top',
            fontsize=10, fontweight='bold' if active else 'normal',
            color=TEXT_C)

def draw_speech_bubble(ax, x, y, w, h, text, color, tail_dir='down'):
    """Draw a speech bubble with text."""
    # Bubble
    box = FancyBboxPatch(
        (x - w/2, y), w, h,
        boxstyle="round,pad=0.1",
        facecolor=color,
        edgecolor=TEXT_C,
        linewidth=2,
        alpha=0.9,
        zorder=10
    )
    ax.add_patch(box)
    
    # Tail (triangle pointing to agent)
    if tail_dir == 'down':
        tail_points = [(x - 0.15, y), (x + 0.15, y), (x, y - 0.3)]
    else:  # up
        tail_points = [(x - 0.15, y + h), (x + 0.15, y + h), (x, y + h + 0.3)]
    
    tail = plt.Polygon(tail_points, facecolor=color, edgecolor=TEXT_C,
                      linewidth=2, alpha=0.9, zorder=9)
    ax.add_patch(tail)
    
    # Text
    ax.text(x, y + h/2, text, ha='center', va='center',
            fontsize=9, color=BG if color != BG else TEXT_C, wrap=True, zorder=11)

def draw_rag_lookup(ax, x, y, visible=False):
    """Draw RAG database lookup indicator."""
    if not visible:
        return
    
    # Database icon (stacked cylinders)
    for i, offset in enumerate([0, 0.2, 0.4]):
        alpha = 0.8 - i * 0.15
        # Cylinder body
        rect = Rectangle((x - 0.4, y + offset - 0.15), 0.8, 0.3,
                        facecolor=COLOR_RAG, edgecolor=TEXT_C,
                        linewidth=1.5, alpha=alpha, zorder=8)
        ax.add_patch(rect)
    
    ax.text(x, y - 0.5, "Policy\nDatabase", ha='center', va='top',
            fontsize=8, color=TEXT_C, fontweight='bold')

def create_frame(stage):
    """Create a single frame of the debate flow animation."""
    fig, ax = plt.subplots(figsize=(14, 10), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, "Multi-Agent Debate: Discount Stacking Dispute", ha='center',
            fontsize=18, fontweight='bold', color=TEXT_C)
    
    # Scenario box
    scenario_box = FancyBboxPatch(
        (2, 8.4), 10, 0.8,
        boxstyle="round,pad=0.1",
        facecolor='#374151',
        edgecolor=TEXT_C,
        linewidth=2
    )
    ax.add_patch(scenario_box)
    ax.text(7, 8.8, '🍕 3 overlapping discounts: Coupon $5 + Loyalty 10% + Promo 15%', 
            ha='center', va='center', fontsize=11, color=TEXT_C)
    
    # Agent positions
    agent1_x, agent1_y = 3, 5
    agent2_x, agent2_y = 11, 5
    judge_x, judge_y = 7, 2
    rag_x, rag_y = 11.5, 7
    
    # Frame 1: Scenario introduction
    if stage == 1:
        draw_agent(ax, agent1_x, agent1_y, COLOR_AGENT1, "Agent 1")
        draw_agent(ax, agent2_x, agent2_y, COLOR_AGENT2, "Agent 2")
        draw_agent(ax, judge_x, judge_y, COLOR_JUDGE, "Judge")
        
        ax.text(7, 3.5, 'Two agents will propose pricing interpretations...', ha='center',
                fontsize=13, color=TEXT_C, style='italic')
    
    # Frame 2: Agent 1 proposes "Stack all"
    elif stage == 2:
        draw_agent(ax, agent1_x, agent1_y, COLOR_AGENT1, "Agent 1", active=True)
        draw_agent(ax, agent2_x, agent2_y, COLOR_AGENT2, "Agent 2")
        draw_agent(ax, judge_x, judge_y, COLOR_JUDGE, "Judge")
        
        draw_speech_bubble(ax, agent1_x, agent1_y + 1.3, 2.8, 1.3,
                          "Stack all discounts!\n$20 -$5 -10% -15%\n= $10.75 final",
                          COLOR_AGENT1)
        
        # Arrow showing calculation
        ax.text(agent1_x, agent1_y - 1.2, '💡 Most generous', ha='center',
                fontsize=9, color=COLOR_AGENT1, style='italic')
    
    # Frame 3: Agent 2 challenges "Only one discount"
    elif stage == 3:
        draw_agent(ax, agent1_x, agent1_y, COLOR_AGENT1, "Agent 1")
        draw_agent(ax, agent2_x, agent2_y, COLOR_AGENT2, "Agent 2", active=True)
        draw_agent(ax, judge_x, judge_y, COLOR_JUDGE, "Judge")
        
        # Agent 1's bubble (faded)
        draw_speech_bubble(ax, agent1_x, agent1_y + 1.3, 2.8, 1.3,
                          "Stack all discounts!\n$20 -$5 -10% -15%\n= $10.75 final",
                          COLOR_AGENT1)
        
        draw_speech_bubble(ax, agent2_x, agent2_y + 1.3, 2.8, 1.3,
                          "Policy: Only ONE\ndiscount per order!\n$15.30 (15% off)",
                          COLOR_AGENT2)
        
        # Conflict indicator
        ax.text(7, 6, '⚡ CONFLICT', ha='center',
                fontsize=14, color=COLOR_ERROR, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=BG, 
                          edgecolor=COLOR_ERROR, linewidth=2))
    
    # Frame 4: Agent 1 defends with loyalty priority
    elif stage == 4:
        draw_agent(ax, agent1_x, agent1_y, COLOR_AGENT1, "Agent 1", active=True)
        draw_agent(ax, agent2_x, agent2_y, COLOR_AGENT2, "Agent 2")
        draw_agent(ax, judge_x, judge_y, COLOR_JUDGE, "Judge")
        
        # Agent 2's bubble (faded)
        draw_speech_bubble(ax, agent2_x, agent2_y + 1.3, 2.8, 1.3,
                          "Policy: Only ONE\ndiscount per order!\n$15.30 (15% off)",
                          COLOR_AGENT2)
        
        draw_speech_bubble(ax, agent1_x, agent1_y + 1.3, 3, 1.4,
                          "Wait! If only one,\nloyalty > coupon!\nCustomer retention\npriority!",
                          COLOR_AGENT1)
        
        ax.text(7, 3.5, 'Agents disagree on interpretation...', ha='center',
                fontsize=12, color=TEXT_C, style='italic')
    
    # Frame 5: Judge checks policy doc (RAG retrieval)
    elif stage == 5:
        draw_agent(ax, agent1_x, agent1_y, COLOR_AGENT1, "Agent 1")
        draw_agent(ax, agent2_x, agent2_y, COLOR_AGENT2, "Agent 2")
        draw_agent(ax, judge_x, judge_y, COLOR_JUDGE, "Judge", active=True)
        
        draw_rag_lookup(ax, rag_x, rag_y, visible=True)
        
        # Arrow from judge to RAG with document icon
        arrow = FancyArrowPatch(
            (judge_x + 1, judge_y + 1), (rag_x - 0.5, rag_y - 0.3),
            arrowstyle='->,head_width=0.5,head_length=0.7',
            color=COLOR_RAG,
            linewidth=3,
            linestyle='--',
            zorder=7
        )
        ax.add_patch(arrow)
        
        # Document icon popup
        doc_icon = FancyBboxPatch(
            (rag_x - 0.6, rag_y + 0.8), 1.2, 0.6,
            boxstyle="round,pad=0.05",
            facecolor=COLOR_RAG,
            edgecolor=TEXT_C,
            linewidth=2,
            alpha=0.9,
            zorder=10
        )
        ax.add_patch(doc_icon)
        ax.text(rag_x, rag_y + 1.1, '📄', ha='center', va='center',
                fontsize=24, zorder=11)
        
        draw_speech_bubble(ax, judge_x, judge_y + 1.5, 3.5, 1.2,
                          'Checking policy:\n"Section 4.2.1:\nOnly one discount"',
                          COLOR_JUDGE, tail_dir='down')
    
    # Frame 6: Judge decides with green checkmark
    elif stage == 6:
        draw_agent(ax, agent1_x, agent1_y, COLOR_AGENT1, "Agent 1")
        draw_agent(ax, agent2_x, agent2_y, COLOR_AGENT2, "Agent 2", active=True)  # Winner
        draw_agent(ax, judge_x, judge_y, COLOR_JUDGE, "Judge", active=True)
        
        draw_rag_lookup(ax, rag_x, rag_y, visible=True)
        
        # Green checkmark on Agent 2
        checkmark = Circle((agent2_x + 0.5, agent2_y + 0.5), 0.3, 
                          color=COLOR_SUCCESS, zorder=10)
        ax.add_patch(checkmark)
        ax.text(agent2_x + 0.5, agent2_y + 0.5, "✓", ha='center', va='center',
                fontsize=20, fontweight='bold', color='white', zorder=11)
        
        # Final decision box
        decision_box = FancyBboxPatch(
            (3, 6.5), 8, 1.2,
            boxstyle="round,pad=0.1",
            facecolor=COLOR_SUCCESS,
            edgecolor=TEXT_C,
            linewidth=3,
            alpha=0.9
        )
        ax.add_patch(decision_box)
        ax.text(7, 7.3, '✓ AGENT 2 WINS', ha='center',
                fontsize=14, fontweight='bold', color='white')
        ax.text(7, 6.9, 'Policy: Only one discount per order', ha='center',
                fontsize=11, color='white', style='italic')
        
        # Cost summary at bottom
        ax.text(7, 1, 'Debate Rounds: 2  •  Agents: 3  •  RAG Lookups: 1', ha='center',
                fontsize=10, color=TEXT_C,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#374151', 
                          edgecolor=TEXT_C, linewidth=1.5))
    
    return fig

# Generate frames
frames = []
frame_durations = []

# 6 frames as specified
stages_with_timing = [
    (1, 1.0),  # Frame 1: Scenario
    (2, 1.0),  # Frame 2: Agent 1 proposes
    (3, 1.0),  # Frame 3: Agent 2 challenges
    (4, 1.0),  # Frame 4: Agent 1 defends
    (5, 1.0),  # Frame 5: Judge checks policy (RAG)
    (6, 1.5),  # Frame 6: Judge decides (longer)
]

for stage, duration in stages_with_timing:
    fig = create_frame(stage)
    fname = os.path.join(OUT_DIR, f'debate_frame_{len(frames):03d}.png')
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
    fname = os.path.join(OUT_DIR, f'debate_frame_{i:03d}.png')
    if os.path.exists(fname):
        os.remove(fname)

print(f'Generated {len(frames)} frames')
print(f'Total animation duration: {sum(frame_durations):.1f}s')
