"""
Generate debate consensus Venn diagram: Agreement zones between 3 agents
Shows small consensus zone with warning about groupthink risk.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as CirclePatch, FancyBboxPatch
from pathlib import Path

# Output paths
OUT_DIR = Path(__file__).parent.parent / 'img'
OUT_DIR.mkdir(exist_ok=True)
OUTPUT_PNG = OUT_DIR / 'ch11-debate-consensus.png'

# Color scheme (dark theme)
BG = '#1a1a2e'
GRID_C = '#2d3748'
TEXT_C = '#e2e8f0'
AGENT1_COLOR = '#3b82f6'    # Blue
AGENT2_COLOR = '#f59e0b'    # Orange
JUDGE_COLOR = '#8b5cf6'     # Purple
ERROR = '#dc2626'           # Red
WARNING = '#fbbf24'         # Yellow

def create_venn_diagram():
    """Create Venn diagram showing agent agreement zones."""
    fig, ax = plt.subplots(figsize=(12, 10), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.axis('off')
    
    # Title
    ax.text(0, 2.8, 'Multi-Agent Debate: Consensus vs Correctness', 
            ha='center', va='top', fontsize=18, fontweight='bold', color=TEXT_C)
    
    # Circle positions (arranged in triangle)
    circle1_x, circle1_y = -0.8, 0.8    # Agent 1 (top-left)
    circle2_x, circle2_y = 0.8, 0.8     # Agent 2 (top-right)
    circle3_x, circle3_y = 0, -0.5      # Judge (bottom)
    radius = 1.2
    
    # Draw circles with translucent fill
    circle1 = CirclePatch((circle1_x, circle1_y), radius, 
                          facecolor=AGENT1_COLOR, edgecolor=AGENT1_COLOR,
                          linewidth=3, alpha=0.3, zorder=2)
    circle2 = CirclePatch((circle2_x, circle2_y), radius,
                          facecolor=AGENT2_COLOR, edgecolor=AGENT2_COLOR,
                          linewidth=3, alpha=0.3, zorder=2)
    circle3 = CirclePatch((circle3_x, circle3_y), radius,
                          facecolor=JUDGE_COLOR, edgecolor=JUDGE_COLOR,
                          linewidth=3, alpha=0.3, zorder=2)
    
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)
    
    # Agent labels
    ax.text(circle1_x - 1.5, circle1_y + 1.2, 'Agent 1', 
            ha='center', va='center', fontsize=14, fontweight='bold',
            color=AGENT1_COLOR)
    ax.text(circle1_x - 1.5, circle1_y + 0.9, '"Apply all\ndiscounts"', 
            ha='center', va='center', fontsize=10, color=TEXT_C, style='italic')
    
    ax.text(circle2_x + 1.5, circle2_y + 1.2, 'Agent 2', 
            ha='center', va='center', fontsize=14, fontweight='bold',
            color=AGENT2_COLOR)
    ax.text(circle2_x + 1.5, circle2_y + 0.9, '"Apply best\ndiscount only"', 
            ha='center', va='center', fontsize=10, color=TEXT_C, style='italic')
    
    ax.text(circle3_x, circle3_y - 1.9, 'Judge', 
            ha='center', va='center', fontsize=14, fontweight='bold',
            color=JUDGE_COLOR)
    ax.text(circle3_x, circle3_y - 2.2, '"Check policy\ndocument"', 
            ha='center', va='center', fontsize=10, color=TEXT_C, style='italic')
    
    # Consensus zone (center intersection)
    consensus_x, consensus_y = 0, 0.2
    consensus_radius = 0.25
    consensus_circle = CirclePatch((consensus_x, consensus_y), consensus_radius,
                                   facecolor=WARNING, edgecolor=WARNING,
                                   linewidth=2, alpha=0.6, zorder=5)
    ax.add_patch(consensus_circle)
    
    ax.text(consensus_x, consensus_y, 'Consensus\nZone', 
            ha='center', va='center', fontsize=9, fontweight='bold',
            color=BG)
    
    # Warning circle around consensus
    warning_circle = CirclePatch((consensus_x, consensus_y), consensus_radius + 0.3,
                                facecolor='none', edgecolor=ERROR,
                                linewidth=3, linestyle='--', alpha=0.8, zorder=6)
    ax.add_patch(warning_circle)
    
    # Warning annotation
    ax.annotate('⚠️ GROUPTHINK RISK', 
                xy=(consensus_x, consensus_y + consensus_radius + 0.3),
                xytext=(consensus_x + 1.5, consensus_y + 1.5),
                color=ERROR, fontsize=12, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=ERROR, lw=2.5),
                bbox=dict(boxstyle='round,pad=0.5', facecolor=BG, 
                          edgecolor=ERROR, linewidth=2.5))
    
    # Key insight box
    insight_text = (
        "Consensus ≠ Correctness\n\n"
        "All 3 agents can agree on the wrong answer.\n"
        "Always validate against ground truth:\n"
        "• Policy documents (RAG)\n"
        "• Business rules database\n"
        "• Human expert review"
    )
    
    insight_box = FancyBboxPatch(
        (-2.8, -2.8), 5.6, 1.2,
        boxstyle="round,pad=0.15",
        facecolor=BG,
        edgecolor=WARNING,
        linewidth=2.5,
        alpha=0.9
    )
    ax.add_patch(insight_box)
    
    ax.text(0, -2.35, insight_text, 
            ha='center', va='center', fontsize=10, color=TEXT_C,
            linespacing=1.5)
    
    # Example scenarios in each zone
    # Agent 1 only (left)
    ax.text(circle1_x - 0.5, circle1_y, '$11.05\n(all stack)', 
            ha='center', va='center', fontsize=8, color=AGENT1_COLOR,
            bbox=dict(boxstyle='round,pad=0.2', facecolor=BG, 
                      edgecolor=AGENT1_COLOR, linewidth=1))
    
    # Agent 2 only (right)
    ax.text(circle2_x + 0.5, circle2_y, '$15.00\n(best only)', 
            ha='center', va='center', fontsize=8, color=AGENT2_COLOR,
            bbox=dict(boxstyle='round,pad=0.2', facecolor=BG, 
                      edgecolor=AGENT2_COLOR, linewidth=1))
    
    # Judge only (bottom)
    ax.text(circle3_x, circle3_y - 0.6, 'Query\nRAG', 
            ha='center', va='center', fontsize=8, color=JUDGE_COLOR,
            bbox=dict(boxstyle='round,pad=0.2', facecolor=BG, 
                      edgecolor=JUDGE_COLOR, linewidth=1))
    
    # Agent 1 + 2 overlap (top)
    ax.text(0, circle1_y + 0.5, '$13.50\n(middle)', 
            ha='center', va='center', fontsize=7, color=TEXT_C,
            bbox=dict(boxstyle='round,pad=0.15', facecolor=BG, 
                      edgecolor=TEXT_C, linewidth=0.8))
    
    # Agent 1 + Judge overlap (left-bottom)
    ax.text(circle1_x - 0.2, circle1_y - 0.5, 'Check\npolicy', 
            ha='center', va='center', fontsize=7, color=TEXT_C,
            bbox=dict(boxstyle='round,pad=0.15', facecolor=BG, 
                      edgecolor=TEXT_C, linewidth=0.8))
    
    # Agent 2 + Judge overlap (right-bottom)
    ax.text(circle2_x + 0.2, circle2_y - 0.5, 'Verify\nrules', 
            ha='center', va='center', fontsize=7, color=TEXT_C,
            bbox=dict(boxstyle='round,pad=0.15', facecolor=BG, 
                      edgecolor=TEXT_C, linewidth=0.8))
    
    # Legend
    legend_y = -2.0
    ax.text(-2.7, legend_y, '●', color=AGENT1_COLOR, fontsize=20, ha='center')
    ax.text(-2.2, legend_y, 'Agent 1 proposals', fontsize=9, va='center', color=TEXT_C)
    
    ax.text(-0.5, legend_y, '●', color=AGENT2_COLOR, fontsize=20, ha='center')
    ax.text(0.05, legend_y, 'Agent 2 proposals', fontsize=9, va='center', color=TEXT_C)
    
    ax.text(1.7, legend_y, '●', color=JUDGE_COLOR, fontsize=20, ha='center')
    ax.text(2.2, legend_y, 'Judge proposals', fontsize=9, va='center', color=TEXT_C)
    
    # Best practice callout
    best_practice = (
        "✓ Best Practice: Use debate for exploration, "
        "then ground final decision in authoritative sources"
    )
    ax.text(0, 2.4, best_practice, 
            ha='center', va='center', fontsize=10, color=TEXT_C,
            bbox=dict(boxstyle='round,pad=0.4', facecolor=BG, 
                      edgecolor=AGENT1_COLOR, linewidth=2))
    
    return fig

# Generate and save
fig = create_venn_diagram()
fig.tight_layout()
fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(fig)

print(f'✓ Saved {OUTPUT_PNG}')
print(f'Dimensions: 12×10 inches at 150 DPI')
