#!/usr/bin/env python3
"""
Generate decision tree for choosing GPU interconnect topology.
Helps users decide between PCIe, NVLink, and InfiniBand based on requirements.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

# Configure matplotlib for dark backgrounds
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#1a1a2e'
plt.rcParams['axes.facecolor'] = '#1a1a2e'

def draw_decision_node(ax, x, y, text, width=3, height=1, color='#1d4ed8'):
    """Draw a decision node (question box)."""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.1",
                         edgecolor=color, facecolor=color,
                         linewidth=2, alpha=0.7)
    ax.add_patch(box)
    
    # Text with line breaks
    ax.text(x, y, text, ha='center', va='center',
            fontsize=10, fontweight='bold', color='white',
            multialignment='center')

def draw_outcome_node(ax, x, y, title, details, width=3.5, height=1.8, color='#15803d'):
    """Draw an outcome node (recommendation box)."""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.15",
                         edgecolor=color, facecolor=color,
                         linewidth=3, alpha=0.8)
    ax.add_patch(box)
    
    # Title
    ax.text(x, y + 0.5, title, ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')
    
    # Details (smaller text)
    ax.text(x, y - 0.2, details, ha='center', va='center',
            fontsize=8, color='white', multialignment='center')

def draw_arrow(ax, x1, y1, x2, y2, label='', color='white'):
    """Draw an arrow with optional label."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
               arrowprops=dict(arrowstyle='->', color=color, lw=2, alpha=0.8))
    
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y, label, ha='left', va='center',
                fontsize=9, color='yellow', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

def generate_decision_tree():
    """Generate interconnect decision tree."""
    
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    # Title
    ax.text(10, 15.5, 'GPU Interconnect Decision Tree', ha='center',
            fontsize=18, fontweight='bold', color='white')
    ax.text(10, 14.8, 'Choose the Right Topology for Your AI Workload', ha='center',
            fontsize=13, color='white', style='italic')
    
    # === Level 1: Model Size ===
    draw_decision_node(ax, 10, 13, 'Model Size?\n(parameters)', color='#1d4ed8')
    
    # === Level 2: Branches ===
    # Left: Small models
    draw_arrow(ax, 8.5, 12.5, 4, 11, '<10B', color='cyan')
    draw_decision_node(ax, 4, 10, 'Budget?\n($/month)', width=2.5, color='#b45309')
    
    # Middle: Medium models
    draw_arrow(ax, 10, 12.5, 10, 11, '10-70B', color='cyan')
    draw_decision_node(ax, 10, 10, 'GPU Count?\n(required)', width=2.5, color='#b45309')
    
    # Right: Large models
    draw_arrow(ax, 11.5, 12.5, 16, 11, '>70B', color='cyan')
    draw_decision_node(ax, 16, 10, 'Cluster Size?\n(nodes)', width=2.5, color='#b45309')
    
    # === Level 3: Outcomes ===
    
    # Outcome 1: PCIe-only (consumer GPUs)
    draw_arrow(ax, 3, 9.5, 2, 7.5, 'Low\n(<$2k)', color='cyan')
    draw_outcome_node(ax, 2, 5.5, '💰 PCIe-Only',
                     'RTX 4090 / 3090\n' +
                     '1-2 GPUs, PCIe Gen4\n' +
                     '$1.50/hr cloud\n' +
                     '✅ Best price/perf\n' +
                     '⚠️  No NVLink',
                     color='#b91c1c', height=2.5)
    
    # Outcome 2: PCIe with more GPUs
    draw_arrow(ax, 5, 9.5, 6, 7.5, 'High\n(>$5k)', color='cyan')
    draw_outcome_node(ax, 6, 5.5, '🔄 Data Parallelism',
                     'Multiple RTX 4090s\n' +
                     'PCIe for grad sync\n' +
                     '$3-6/hr (2-4 GPUs)\n' +
                     '✅ Scales horizontally\n' +
                     '⚠️  Slow tensor parallel',
                     color='#b45309', height=2.5)
    
    # Outcome 3: NVLink single node
    draw_arrow(ax, 9, 9.5, 9, 7.5, '2-8 GPUs', color='cyan')
    draw_outcome_node(ax, 9, 5.5, '🚀 NVLink (Single Node)',
                     'A100 / H100 (8× GPUs)\n' +
                     'NVLink 600-900 GB/s\n' +
                     '$20-30/hr (8× A100)\n' +
                     '✅ Tensor parallelism\n' +
                     '✅ Low latency (<2μs)',
                     color='#15803d', height=2.5)
    
    # Outcome 4: Need more than 8 GPUs
    draw_arrow(ax, 11, 9.5, 12, 7.5, '>8 GPUs', color='cyan')
    draw_decision_node(ax, 12, 6.5, 'Multi-node?\n(>1 server)', width=2.5, color='#b45309')
    
    draw_arrow(ax, 11.5, 6, 10.5, 4.5, 'No', color='red')
    draw_outcome_node(ax, 10.5, 2.5, '❌ Not Feasible',
                     'Cannot fit >8 GPUs\n' +
                     'in single node\n' +
                     '→ Use multi-node',
                     width=2.5, height=2, color='#6b7280')
    
    draw_arrow(ax, 12.5, 6, 13.5, 4.5, 'Yes', color='lime')
    draw_outcome_node(ax, 13.5, 2.5, '🌐 InfiniBand Required',
                     'NVLink within nodes\n' +
                     'IB cross-node\n' +
                     '→ See multi-node path',
                     width=2.5, height=2, color='#1d4ed8')
    
    # Outcome 5: Multi-node (1-4 nodes)
    draw_arrow(ax, 15.5, 9.5, 15.5, 7.5, '1-4 nodes', color='cyan')
    draw_outcome_node(ax, 15.5, 5.5, '🔗 InfiniBand HDR',
                     '2-4 nodes × 8 GPUs\n' +
                     'NVLink + IB (200 Gb/s)\n' +
                     '$80-160/hr cluster\n' +
                     '✅ 16-32 GPU scale\n' +
                     '⚠️  2-3μs cross-node',
                     color='#1d4ed8', height=2.5)
    
    # Outcome 6: Large cluster (>4 nodes)
    draw_arrow(ax, 16.5, 9.5, 18, 7.5, '>4 nodes', color='cyan')
    draw_outcome_node(ax, 18, 5.5, '🏢 HPC Cluster',
                     '>32 GPUs (>4 nodes)\n' +
                     'InfiniBand NDR fabric\n' +
                     '$200+/hr supercomp\n' +
                     '✅ Scales to 1000s GPUs\n' +
                     '💰 High upfront cost',
                     color='#7c3aed', height=2.5)
    
    # === Legend ===
    legend_x, legend_y = 1, 1.5
    ax.text(legend_x, legend_y + 1, '🔑 Legend:', fontsize=11, fontweight='bold', color='white')
    
    legend_items = [
        ('Decision Point', '#1d4ed8'),
        ('Recommended Solution', '#15803d'),
        ('Alternative Option', '#b45309'),
        ('Not Recommended', '#6b7280')
    ]
    
    for i, (label, color) in enumerate(legend_items):
        y = legend_y - i * 0.4
        box = FancyBboxPatch((legend_x, y - 0.15), 0.4, 0.3,
                            boxstyle="round,pad=0.05",
                            edgecolor=color, facecolor=color,
                            linewidth=2, alpha=0.7)
        ax.add_patch(box)
        ax.text(legend_x + 0.6, y, label, va='center', fontsize=9, color='white')
    
    # === Cost/Performance Summary ===
    summary_x, summary_y = 1, 11
    ax.text(summary_x, summary_y + 0.5, '💡 Quick Guide:', fontsize=12, 
            fontweight='bold', color='yellow')
    
    summary_text = [
        '• <10B model → PCIe OK',
        '• 10-30B → NVLink helpful',
        '• 30-70B → NVLink required',
        '• >70B → Multi-node + IB',
        '• Training → NVLink always',
        '• Inference → PCIe if batch>1'
    ]
    
    for i, line in enumerate(summary_text):
        ax.text(summary_x, summary_y - i * 0.4, line, fontsize=9, color='white')
    
    # Footer
    fig.text(0.5, 0.02,
             '🎯 Start at top → Follow arrows based on your requirements → Arrive at recommended topology',
             ha='center', fontsize=12, color='white', weight='bold',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#1a1a2e',
                      edgecolor='white', linewidth=2))
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    
    # Save figure
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'img')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'decision_tree.png')
    
    plt.savefig(output_path, dpi=150, facecolor='#1a1a2e', bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    
    plt.close()

if __name__ == '__main__':
    generate_decision_tree()
