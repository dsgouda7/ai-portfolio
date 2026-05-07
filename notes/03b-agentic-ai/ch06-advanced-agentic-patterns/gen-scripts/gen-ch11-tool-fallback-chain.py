"""
Generate tool fallback chain waterfall diagram: Cost-aware fallback sequence
Shows waterfall flow with cache → DB → API → human escalation, with failure/success indicators.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import os

# Output paths
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'img')
os.makedirs(OUT_DIR, exist_ok=True)
OUTPUT_PNG = os.path.join(OUT_DIR, 'ch11-tool-fallback-chain.png')

# Color scheme
COLOR_BG = '#1a1a2e'
COLOR_SUCCESS = '#27AE60'    # Green
COLOR_FAILURE = '#E74C3C'    # Red
COLOR_SKIPPED = '#555566'    # Gray
COLOR_STAGE = '#4A90E2'      # Blue
COLOR_TEXT = '#FFFFFF'       # White text
COLOR_ACCENT = '#F39C12'     # Orange

def draw_stage_box(ax, x, y, w, h, label, sublabel, status, stage_num):
    """Draw a stage box with status indicator."""
    # Determine color based on status
    if status == 'success':
        color = COLOR_SUCCESS
        alpha = 0.9
        linewidth = 3
    elif status == 'failure':
        color = COLOR_FAILURE
        alpha = 0.9
        linewidth = 3
    else:  # skipped
        color = COLOR_SKIPPED
        alpha = 0.4
        linewidth = 1
    
    # Main box
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.1",
        facecolor=COLOR_STAGE,
        edgecolor=color,
        linewidth=linewidth,
        alpha=alpha
    )
    ax.add_patch(box)
    
    # Stage number
    ax.text(x + 0.3, y + h - 0.2, f"Stage {stage_num}",
            fontsize=9, color=COLOR_TEXT, fontweight='bold')
    
    # Main label
    ax.text(x + w/2, y + h/2 + 0.1, label,
            ha='center', va='center',
            fontsize=12, fontweight='bold', color=COLOR_TEXT)
    
    # Sublabel (cost/latency)
    ax.text(x + w/2, y + h/2 - 0.25, sublabel,
            ha='center', va='center',
            fontsize=10, color=COLOR_TEXT, style='italic')
    
    # Status indicator (circle with symbol)
    indicator_x = x + w - 0.4
    indicator_y = y + h/2
    
    if status == 'success':
        circle = Circle((indicator_x, indicator_y), 0.25,
                       facecolor=COLOR_SUCCESS, edgecolor=COLOR_TEXT, linewidth=2)
        ax.add_patch(circle)
        ax.text(indicator_x, indicator_y, '✓', ha='center', va='center',
                fontsize=18, color=COLOR_TEXT, fontweight='bold')
    elif status == 'failure':
        circle = Circle((indicator_x, indicator_y), 0.25,
                       facecolor=COLOR_FAILURE, edgecolor=COLOR_TEXT, linewidth=2)
        ax.add_patch(circle)
        ax.text(indicator_x, indicator_y, '✗', ha='center', va='center',
                fontsize=18, color=COLOR_TEXT, fontweight='bold')
    
    # Failure reason (if failed)
    if status == 'failure':
        ax.text(x + w + 0.6, y + h/2, sublabel.split('\n')[0] if '\n' in sublabel else sublabel,
                ha='left', va='center',
                fontsize=10, color=COLOR_FAILURE, style='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=COLOR_BG, alpha=0.8))

def draw_waterfall_arrow(ax, x1, y1, x2, y2, color=COLOR_TEXT, linewidth=2.5):
    """Draw a downward waterfall arrow."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->,head_width=0.4,head_length=0.6',
        color=color,
        linewidth=linewidth,
        linestyle='-',
        zorder=2
    )
    ax.add_patch(arrow)

def main():
    """Generate the tool fallback chain waterfall diagram."""
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 14)
    ax.axis('off')
    ax.set_facecolor(COLOR_BG)
    fig.patch.set_facecolor(COLOR_BG)
    
    # Title
    ax.text(6, 13.5, "Tool Fallback Chain", ha='center',
            fontsize=16, fontweight='bold', color=COLOR_TEXT)
    ax.text(6, 13, "Waterfall Execution with Cost Optimization",
            fontsize=12, color=COLOR_TEXT, style='italic')
    
    # Stage dimensions
    stage_w = 6
    stage_h = 1.5
    stage_x = 3
    stage_spacing = 2.2
    
    # Stage 1: Cache (failed - stale data)
    stage1_y = 11
    draw_stage_box(ax, stage_x, stage1_y, stage_w, stage_h,
                  "Try Cache", "free, 10ms", "failure", 1)
    ax.text(stage_x + stage_w + 0.6, stage1_y + stage_h/2, "Stale data",
            ha='left', va='center',
            fontsize=10, color=COLOR_FAILURE, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLOR_BG, alpha=0.8))
    
    # Arrow 1 → 2
    draw_waterfall_arrow(ax, stage_x + stage_w/2, stage1_y,
                        stage_x + stage_w/2, stage1_y - stage_spacing + stage_h)
    
    # Stage 2: DB (failed - timeout)
    stage2_y = stage1_y - stage_spacing
    draw_stage_box(ax, stage_x, stage2_y, stage_w, stage_h,
                  "Try Database", "$0.0001, 50ms", "failure", 2)
    ax.text(stage_x + stage_w + 0.6, stage2_y + stage_h/2, "Timeout",
            ha='left', va='center',
            fontsize=10, color=COLOR_FAILURE, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLOR_BG, alpha=0.8))
    
    # Arrow 2 → 3
    draw_waterfall_arrow(ax, stage_x + stage_w/2, stage2_y,
                        stage_x + stage_w/2, stage2_y - stage_spacing + stage_h)
    
    # Stage 3: API (success)
    stage3_y = stage2_y - stage_spacing
    draw_stage_box(ax, stage_x, stage3_y, stage_w, stage_h,
                  "Call External API", "$0.001, 200ms", "success", 3)
    ax.text(stage_x + stage_w + 0.6, stage3_y + stage_h/2, "Success",
            ha='left', va='center',
            fontsize=10, color=COLOR_SUCCESS, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLOR_BG, alpha=0.8))
    
    # Arrow 3 → 4 (dashed, not taken)
    draw_waterfall_arrow(ax, stage_x + stage_w/2, stage3_y,
                        stage_x + stage_w/2, stage3_y - stage_spacing + stage_h)
    # Overlay with dashed gray line to show "not reached"
    ax.plot([stage_x + stage_w/2, stage_x + stage_w/2],
            [stage3_y, stage3_y - stage_spacing + stage_h],
            color=COLOR_SKIPPED, linestyle='--', linewidth=2, alpha=0.6)
    
    # Stage 4: Human escalation (not reached, grayed out)
    stage4_y = stage3_y - stage_spacing
    draw_stage_box(ax, stage_x, stage4_y, stage_w, stage_h,
                  "Escalate to Human", "manual intervention", "skipped", 4)
    ax.text(stage_x + stage_w + 0.6, stage4_y + stage_h/2, "Not reached",
            ha='left', va='center',
            fontsize=10, color=COLOR_SKIPPED, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLOR_BG, alpha=0.8))
    
    # Total metrics box at bottom
    metrics_y = 1.5
    metrics_box = FancyBboxPatch(
        (1.5, metrics_y), 9, 1.8,
        boxstyle="round,pad=0.15",
        facecolor='#2a2a3e',
        edgecolor=COLOR_ACCENT,
        linewidth=3,
        alpha=0.95
    )
    ax.add_patch(metrics_box)
    
    ax.text(6, metrics_y + 1.4, "Execution Summary", ha='center',
            fontsize=13, fontweight='bold', color=COLOR_TEXT)
    
    metrics_text = [
        f"Total Cost: $0.001",
        f"Total Latency: 300ms (cache: 10ms + DB: 50ms + API: 200ms + overhead: 40ms)",
        f"Success Rate: 75% (succeeded at stage 3 of 4)",
        f"Attempts: 3 (cache → DB → API)"
    ]
    
    for i, text in enumerate(metrics_text):
        ax.text(6, metrics_y + 0.95 - i*0.25, text, ha='center',
                fontsize=10, color=COLOR_TEXT)
    
    # Cost breakdown annotation
    cost_y = 0.5
    ax.text(6, cost_y, "💡 Cost optimization: Tried free cache first, then low-cost DB, before expensive API",
            ha='center', fontsize=10, color=COLOR_ACCENT, style='italic')
    
    # Legend
    legend_x = 0.5
    legend_y = 12
    legend_items = [
        (COLOR_SUCCESS, 'Success'),
        (COLOR_FAILURE, 'Failure'),
        (COLOR_SKIPPED, 'Not Reached')
    ]
    
    ax.text(legend_x, legend_y + 0.5, "Status:", fontsize=10, fontweight='bold', color=COLOR_TEXT)
    for i, (color, label) in enumerate(legend_items):
        y = legend_y - i * 0.3
        circle = Circle((legend_x + 0.2, y), 0.15, facecolor=color, edgecolor=COLOR_TEXT, linewidth=1.5)
        ax.add_patch(circle)
        ax.text(legend_x + 0.5, y, label, ha='left', va='center',
                fontsize=9, color=COLOR_TEXT)
    
    # Save
    print(f"Saving PNG to {OUTPUT_PNG}...")
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, facecolor=COLOR_BG, edgecolor='none')
    plt.close()
    print("✓ Complete!")

if __name__ == '__main__':
    main()
