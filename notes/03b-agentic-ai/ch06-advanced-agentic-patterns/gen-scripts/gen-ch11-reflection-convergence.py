"""
Generate reflection convergence graph: Error rate vs number of refinement loops
Shows diminishing returns after 3 loops with 1% error rate threshold.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Output paths
OUT_DIR = Path(__file__).parent.parent / 'img'
OUT_DIR.mkdir(exist_ok=True)
OUTPUT_PNG = OUT_DIR / 'ch11-reflection-convergence.png'

# Color scheme (dark theme)
BG = '#1a1a2e'
GRID_C = '#2d3748'
TEXT_C = '#e2e8f0'
PRIMARY = '#3b82f6'    # Blue
ERROR = '#dc2626'      # Red
WARNING = '#fbbf24'    # Yellow
SUCCESS = '#10b981'    # Green

# Data points: (loops, error_rate_percent)
data_points = [
    (1, 8.0),      # Single-pass: 8% error rate
    (2, 4.0),      # 2 loops: 4% error rate
    (3, 1.2),      # 3 loops: 1.2% error rate
    (5, 0.5),      # 5 loops: 0.5% error rate (diminishing returns)
    (10, 0.5),     # 10 loops: 0.5% error rate (no improvement)
]

loops = np.array([p[0] for p in data_points])
error_rates = np.array([p[1] for p in data_points])

# Create plot
fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
ax.set_facecolor(BG)

# Target line: 1% error rate threshold
ax.axhline(1.0, color=ERROR, linestyle='--', linewidth=2, 
           label='1% error target', alpha=0.8, zorder=2)

# Error rate curve with markers
ax.plot(loops, error_rates, color=PRIMARY, linewidth=3, 
        marker='o', markersize=10, markerfacecolor=PRIMARY, 
        markeredgecolor=TEXT_C, markeredgewidth=1.5,
        label='Error rate', zorder=3)

# Highlight diminishing returns zone (after 3 loops)
ax.axvspan(3, 10, color=WARNING, alpha=0.1, zorder=1)
ax.text(6.5, 7, 'Diminishing returns\nafter 3 loops', 
        ha='center', va='center', fontsize=11, color=WARNING,
        bbox=dict(boxstyle='round,pad=0.5', facecolor=BG, 
                  edgecolor=WARNING, linewidth=2))

# Annotate key points
# Point 1: Single-pass (poor)
ax.annotate('Single-pass\n8% error', 
            xy=(1, 8.0), xytext=(1.5, 9),
            color=ERROR, fontsize=10, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=ERROR, lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor=BG, 
                      edgecolor=ERROR, linewidth=1.5))

# Point 2: Sweet spot at 3 loops
ax.annotate('Sweet spot\n1.2% error', 
            xy=(3, 1.2), xytext=(4, 2.5),
            color=SUCCESS, fontsize=10, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=SUCCESS, lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor=BG, 
                      edgecolor=SUCCESS, linewidth=1.5))

# Point 3: Plateau at 5+ loops
ax.annotate('Plateau\n0.5% error', 
            xy=(5, 0.5), xytext=(7, 0.8),
            color=TEXT_C, fontsize=10,
            arrowprops=dict(arrowstyle='->', color=TEXT_C, lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor=BG, 
                      edgecolor=TEXT_C, linewidth=1.5))

# Labels and title
ax.set_xlabel('Number of Refinement Loops', color=TEXT_C, fontsize=13, fontweight='bold')
ax.set_ylabel('Error Rate (%)', color=TEXT_C, fontsize=13, fontweight='bold')
ax.set_title('Reflection Loop Convergence — Diminishing Returns After 3 Loops', 
             color=TEXT_C, fontsize=15, fontweight='bold', pad=20)

# Styling
ax.tick_params(colors=TEXT_C, labelsize=11)
for spine in ax.spines.values():
    spine.set_edgecolor(GRID_C)
    spine.set_linewidth(1.5)

ax.grid(True, which='both', color=GRID_C, linewidth=0.8, linestyle='--', alpha=0.5)
ax.set_xticks([1, 2, 3, 5, 10])
ax.set_xlim(0.5, 10.5)
ax.set_ylim(-0.5, 9)

# Legend
legend = ax.legend(loc='upper right', fontsize=11, frameon=True, 
                   facecolor=BG, edgecolor=GRID_C, labelcolor=TEXT_C)
legend.get_frame().set_linewidth(1.5)

# Cost insight box
cost_text = (
    "💰 Cost trade-off:\n"
    "• 1 loop: 1× tokens, 8% error\n"
    "• 3 loops: 3× tokens, 1.2% error\n"
    "• 10 loops: 10× tokens, 0.5% error"
)
ax.text(0.02, 0.98, cost_text, transform=ax.transAxes,
        ha='left', va='top', fontsize=9, color=TEXT_C,
        bbox=dict(boxstyle='round,pad=0.6', facecolor=BG, 
                  edgecolor=PRIMARY, linewidth=2))

# Save
fig.tight_layout()
fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(fig)

print(f'✓ Saved {OUTPUT_PNG}')
print(f'Dimensions: 10×6 inches at 150 DPI')
