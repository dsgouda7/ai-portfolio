"""
Generate feature store architecture diagram.

Shows the complete feature store architecture:
- Data sources (PostgreSQL, Kafka)
- Feature definitions
- Offline store (Parquet, BigQuery)
- Online store (Redis, DynamoDB)
- Training and serving flows
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set dark theme
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(16, 12), facecolor='#1a1a2e')
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Color palette
colors = {
    'primary': '#1e3a8a',
    'success': '#15803d',
    'caution': '#b45309',
    'danger': '#b91c1c',
    'info': '#1d4ed8',
    'bg': '#1a1a2e',
    'text': '#ffffff'
}

def draw_box(ax, x, y, width, height, text, color, text_lines=None):
    """Draw a fancy box with text."""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.1",
        facecolor=color,
        edgecolor='white',
        linewidth=2,
        alpha=0.9
    )
    ax.add_patch(box)
    
    # Main text
    ax.text(x + width/2, y + height - 0.3, text,
            ha='center', va='top', fontsize=12, fontweight='bold', color='white')
    
    # Additional lines
    if text_lines:
        line_y = y + height - 0.7
        for line in text_lines:
            ax.text(x + width/2, line_y, line,
                   ha='center', va='top', fontsize=9, color='white')
            line_y -= 0.25

def draw_arrow(ax, x1, y1, x2, y2, label='', color='white'):
    """Draw an arrow with optional label."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->,head_width=0.4,head_length=0.4',
        color=color,
        linewidth=2,
        alpha=0.8
    )
    ax.add_patch(arrow)
    
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.2, label,
               ha='center', va='bottom', fontsize=8,
               bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['bg'], edgecolor=color, alpha=0.9),
               color='white')

# Title
ax.text(5, 11.5, 'Feature Store Architecture', ha='center', va='top',
        fontsize=18, fontweight='bold', color='white')

# Layer 1: Data Sources (bottom)
draw_box(ax, 0.5, 0.5, 2, 1.2, 'Data Sources',
         colors['info'], ['PostgreSQL', 'Kafka Streams', 'S3/ADLS'])

# Layer 2: Feature Definitions
draw_box(ax, 3.5, 0.5, 3, 1.2, 'Feature Definitions',
         colors['primary'], ['Python DSL', 'feature_views.py', 'Entity schemas'])

# Layer 3: Offline Store
draw_box(ax, 0.5, 2.5, 2.5, 1.5, 'Offline Store',
         colors['caution'], ['Parquet / BigQuery', 'Historical features', 'Point-in-time joins', 'TBs of data'])

# Layer 3: Online Store
draw_box(ax, 7, 2.5, 2.5, 1.5, 'Online Store',
         colors['success'], ['Redis / DynamoDB', 'Latest features only', '<10ms lookups', 'GBs of data'])

# Layer 4: Registry
draw_box(ax, 3.5, 2.5, 2.5, 1.5, 'Feature Registry',
         colors['primary'], ['SQLite / PostgreSQL', 'Feature schemas', 'Materialization history', 'Lineage tracking'])

# Layer 5: Training
draw_box(ax, 0.5, 5, 3, 1.5, 'Training Pipeline',
         colors['caution'], ['Jupyter / Airflow', 'Batch retrieval', '1M+ rows', '10s-10min latency'])

# Layer 5: Serving
draw_box(ax, 6.5, 5, 3, 1.5, 'Serving Pipeline',
         colors['success'], ['API Server', 'Point lookups', 'Single row', '<10ms latency'])

# Layer 6: Orchestration
draw_box(ax, 3.5, 7.5, 3, 1.2, 'Materialization',
         colors['danger'], ['Airflow / Cron', 'Hourly/Daily jobs', 'Offline → Online'])

# Arrows: Data flow
draw_arrow(ax, 1.5, 1.7, 1.5, 2.5, 'Load', colors['info'])
draw_arrow(ax, 5, 1.7, 5, 2.5, 'Register', colors['primary'])
draw_arrow(ax, 8.25, 1.7, 8.25, 2.5, 'Materialize', colors['success'])

# Arrows: Training flow
draw_arrow(ax, 2, 4, 2, 5, 'Query', colors['caution'])
draw_arrow(ax, 5, 4, 2, 5, 'Metadata', colors['primary'])

# Arrows: Serving flow
draw_arrow(ax, 8.25, 4, 8.25, 5, 'Fetch', colors['success'])

# Arrows: Materialization
draw_arrow(ax, 3, 7.5, 2, 4, 'Write', colors['danger'])
draw_arrow(ax, 6, 7.5, 8.25, 4, 'Write', colors['danger'])

# Add annotations
ax.text(0.3, 9.5, '📊 Training: Batch historical features', fontsize=10, color=colors['caution'], fontweight='bold')
ax.text(0.3, 9.1, '   • Point-in-time correct joins', fontsize=9, color='white')
ax.text(0.3, 8.7, '   • 1M+ rows, 10s-10min latency', fontsize=9, color='white')
ax.text(0.3, 8.3, '   • Prevent data leakage', fontsize=9, color='white')

ax.text(5.5, 9.5, '🚀 Serving: Real-time feature lookup', fontsize=10, color=colors['success'], fontweight='bold')
ax.text(5.5, 9.1, '   • Single entity lookup', fontsize=9, color='white')
ax.text(5.5, 8.7, '   • <10ms p95 latency', fontsize=9, color='white')
ax.text(5.5, 8.3, '   • Latest values only', fontsize=9, color='white')

# Key insight box
insight_text = [
    '💡 Key Insight: Single feature definition → training & serving',
    '   Eliminates training-serving skew',
    '   Precomputed features → 97% latency reduction (380ms → 8ms)'
]
y_pos = 10.8
for line in insight_text:
    ax.text(5, y_pos, line, ha='center', fontsize=10, color='white',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=colors['danger'], edgecolor='white', alpha=0.8))
    y_pos -= 0.35

plt.tight_layout()
plt.savefig('../img/ch08-feature-store-architecture.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
print("✓ Generated: ch08-feature-store-architecture.png")
plt.close()
