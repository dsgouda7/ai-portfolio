"""
Generate training-serving flow comparison diagram.

Shows the complete flow for both training and serving:
- Data → Features → Model → Inference
- Highlights differences in latency requirements
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), facecolor='#1a1a2e')

colors = {
    'primary': '#1e3a8a',
    'success': '#15803d',
    'caution': '#b45309',
    'danger': '#b91c1c',
    'info': '#1d4ed8',
    'bg': '#1a1a2e'
}

def setup_ax(ax, title):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')
    ax.text(5, 3.7, title, ha='center', va='top',
            fontsize=14, fontweight='bold', color='white')

def draw_box(ax, x, y, width, height, text, color, subtext=''):
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.1",
        facecolor=color,
        edgecolor='white',
        linewidth=2,
        alpha=0.9
    )
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2 + 0.1, text,
            ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    if subtext:
        ax.text(x + width/2, y + height/2 - 0.25, subtext,
               ha='center', va='center', fontsize=8, color='white')

def draw_arrow(ax, x1, y1, x2, y2, label='', color='white'):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->,head_width=0.3,head_length=0.3',
        color=color,
        linewidth=2.5,
        alpha=0.8
    )
    ax.add_patch(arrow)
    
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.15, label,
               ha='center', va='bottom', fontsize=9,
               bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['bg'], edgecolor=color),
               color='white', fontweight='bold')

# === TRAINING FLOW (Top) ===
setup_ax(ax1, '🎓 Training Flow: Historical Features (Batch)')

# Boxes
draw_box(ax1, 0.3, 1.5, 1.5, 0.8, 'Raw Data', colors['info'], 'user_uploads.parquet')
draw_box(ax1, 2.5, 1.5, 1.5, 0.8, 'Feature Store', colors['primary'], 'Offline Store')
draw_box(ax1, 4.7, 1.5, 1.5, 0.8, 'Training Set', colors['caution'], '1M rows')
draw_box(ax1, 6.9, 1.5, 1.5, 0.8, 'Model Training', colors['danger'], 'BERT fine-tune')
draw_box(ax1, 8.7, 1.5, 1.2, 0.8, 'Model', colors['success'], '96% acc')

# Arrows with timing
draw_arrow(ax1, 1.8, 1.9, 2.5, 1.9, '5min', colors['info'])
draw_arrow(ax1, 4.0, 1.9, 4.7, 1.9, '30s', colors['primary'])
draw_arrow(ax1, 6.2, 1.9, 6.9, 1.9, '2s', colors['caution'])
draw_arrow(ax1, 8.4, 1.9, 8.7, 1.9, '3hr', colors['danger'])

# Annotations
ax1.text(1, 0.9, '📊 Point-in-time join', fontsize=9, color='white')
ax1.text(1, 0.6, '   • No data leakage', fontsize=8, color='white')
ax1.text(1, 0.3, '   • Historical values', fontsize=8, color='white')

ax1.text(3.2, 0.9, '⚡ Batch query', fontsize=9, color='white')
ax1.text(3.2, 0.6, '   • 1M+ rows', fontsize=8, color='white')
ax1.text(3.2, 0.3, '   • 10s-10min OK', fontsize=8, color='white')

ax1.text(7.5, 0.9, '🔄 Runs once', fontsize=9, color='white')
ax1.text(7.5, 0.6, '   • Offline job', fontsize=8, color='white')
ax1.text(7.5, 0.3, '   • Not latency-critical', fontsize=8, color='white')

# Total time
ax1.text(5, 2.8, 'Total: ~5min + 3hr training', ha='center', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.4', facecolor=colors['caution'], edgecolor='white'),
        color='white', fontweight='bold')

# === SERVING FLOW (Bottom) ===
setup_ax(ax2, '🚀 Serving Flow: Real-Time Features (Online)')

# Boxes
draw_box(ax2, 0.3, 1.5, 1.3, 0.8, 'API Request', colors['info'], 'user_id=42')
draw_box(ax2, 2.3, 1.5, 1.5, 0.8, 'Feature Store', colors['primary'], 'Online Store (Redis)')
draw_box(ax2, 4.5, 1.5, 1.5, 0.8, 'Features', colors['success'], '8 values')
draw_box(ax2, 6.7, 1.5, 1.5, 0.8, 'Model Inference', colors['danger'], 'Llama-3-8B')
draw_box(ax2, 8.7, 1.2, 1.2, 0.8, 'Response', colors['success'], 'JSON')

# Arrows with timing
draw_arrow(ax2, 1.6, 1.9, 2.3, 1.9, '2ms', colors['info'])
draw_arrow(ax2, 3.8, 1.9, 4.5, 1.9, '8ms', colors['primary'])
draw_arrow(ax2, 6.0, 1.9, 6.7, 1.9, '3ms', colors['success'])
draw_arrow(ax2, 8.2, 1.9, 8.7, 1.9, '950ms', colors['danger'])

# Annotations
ax2.text(1, 0.9, '🔍 Single lookup', fontsize=9, color='white')
ax2.text(1, 0.6, '   • 1 entity', fontsize=8, color='white')
ax2.text(1, 0.3, '   • Latest values', fontsize=8, color='white')

ax2.text(3.3, 0.9, '⚡ Redis GET', fontsize=9, color='white')
ax2.text(3.3, 0.6, '   • Precomputed', fontsize=8, color='white')
ax2.text(3.3, 0.3, '   • <10ms p95', fontsize=8, color='white')

ax2.text(7.5, 0.9, '🚨 Latency-critical', fontsize=9, color='white')
ax2.text(7.5, 0.6, '   • 10k req/day', fontsize=8, color='white')
ax2.text(7.5, 0.3, '   • <2s SLA', fontsize=8, color='white')

# Total time
ax2.text(5, 2.8, 'Total: 971ms (p50) ✅ Target <2s', ha='center', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.4', facecolor=colors['success'], edgecolor='white'),
        color='white', fontweight='bold')

# Key differences box
ax2.text(5, 0.05, '💡 Key Difference: Training uses OFFLINE store (historical, batch) | Serving uses ONLINE store (latest, real-time)',
        ha='center', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.5', facecolor=colors['danger'], edgecolor='white', alpha=0.9),
        color='white', fontweight='bold')

plt.tight_layout()
plt.savefig('../img/ch08-training-serving-flow.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
print("✓ Generated: ch08-training-serving-flow.png")
plt.close()
