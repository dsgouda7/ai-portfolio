"""
Generate simplified mixed precision training visualization for Ch.10.
Shows FP32 vs FP16 training comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

plt.style.use('dark_background')
DARK_BG = '#1a1a2e'

def create_mixed_precision_diagram():
    """Create mixed precision training flow diagram."""
    fig = plt.figure(figsize=(16, 10), facecolor=DARK_BG)
    fig.patch.set_facecolor(DARK_BG)
    
    fig.suptitle('Mixed Precision Training (AMP) — 2× Speedup, Minimal Accuracy Loss',
                 fontsize=18, color='white', weight='bold', y=0.98)
    
    # Create 2x2 layout
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
    
    # Top left: FP32 training
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_title('Standard Training (FP32)', fontsize=14, color='white', weight='bold', pad=20)
    
    # Draw FP32 pipeline
    y = 9
    boxes = [
        ('Input\n(FP32)', 1, y, '#3b82f6'),
        ('Forward\n(FP32)', 3.5, y, '#3b82f6'),
        ('Loss\n(FP32)', 6, y, '#3b82f6'),
        ('Backward\n(FP32)', 8.5, y, '#3b82f6'),
    ]
    
    for label, x, y_pos, color in boxes:
        box = FancyBboxPatch((x, y_pos-0.6), 1.2, 1.2, boxstyle="round,pad=0.1",
                            edgecolor='white', facecolor=color, linewidth=2)
        ax1.add_patch(box)
        ax1.text(x+0.6, y_pos, label, ha='center', va='center', fontsize=10, color='white', weight='bold')
        
    # Arrows
    for i in range(len(boxes)-1):
        x1 = boxes[i][1] + 1.2
        x2 = boxes[i+1][1]
        ax1.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle='->', lw=2, color='white'))
    
    # Metrics
    ax1.text(5, 7, 'Latency: 70ms per batch\nMemory: 8 GB\nmAP: 82.1%',
             ha='center', fontsize=11, color='#aaa',
             bbox=dict(boxstyle='round', facecolor='#2d3748', alpha=0.8, edgecolor='white'))
    
    # Top right: Mixed precision training
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_title('Mixed Precision Training (AMP)', fontsize=14, color='white', weight='bold', pad=20)
    
    # Draw mixed precision pipeline
    y = 9
    boxes_amp = [
        ('Input\n(FP16)', 1, y, '#10b981'),
        ('Forward\n(FP16)', 3.5, y, '#10b981'),
        ('Loss\n(FP32)', 6, y, '#f59e0b'),
        ('Backward\n(FP16)', 8.5, y, '#10b981'),
    ]
    
    for label, x, y_pos, color in boxes_amp:
        box = FancyBboxPatch((x, y_pos-0.6), 1.2, 1.2, boxstyle="round,pad=0.1",
                            edgecolor='white', facecolor=color, linewidth=2)
        ax2.add_patch(box)
        ax2.text(x+0.6, y_pos, label, ha='center', va='center', fontsize=10, color='white', weight='bold')
        
    # Arrows
    for i in range(len(boxes_amp)-1):
        x1 = boxes_amp[i][1] + 1.2
        x2 = boxes_amp[i+1][1]
        ax2.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle='->', lw=2, color='white'))
    
    # Gradient scaling annotation
    ax2.text(5, 6, '⚡ Gradient Scaling:\nFP16 grads × 2048 → FP32 → update',
             ha='center', fontsize=10, color='#10b981',
             bbox=dict(boxstyle='round', facecolor='#1e3a1a', alpha=0.9, edgecolor='#10b981', linewidth=2))
    
    # Metrics
    ax2.text(5, 4.5, 'Latency: 35ms per batch ✅\nMemory: 4 GB (50% reduction)\nmAP: 82.1% (no degradation)',
             ha='center', fontsize=11, color='#10b981',
             bbox=dict(boxstyle='round', facecolor='#2d3748', alpha=0.8, edgecolor='white'))
    
    # Bottom left: Precision comparison
    ax3 = fig.add_subplot(gs[1, 0])
    
    categories = ['Forward Pass', 'Loss Calc', 'Backward Pass', 'Optimizer']
    fp32_time = [40, 5, 60, 10]
    fp16_time = [20, 5, 30, 10]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, fp32_time, width, label='FP32', color='#3b82f6', edgecolor='white', linewidth=1)
    bars2 = ax3.bar(x + width/2, fp16_time, width, label='FP16 (AMP)', color='#10b981', edgecolor='white', linewidth=1)
    
    ax3.set_xlabel('Training Phase', fontsize=12, color='white')
    ax3.set_ylabel('Time (ms)', fontsize=12, color='white')
    ax3.set_title('Training Time Breakdown', fontsize=12, color='white', weight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories, rotation=15, ha='right')
    ax3.legend(fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}ms', ha='center', va='bottom', fontsize=9, color='white')
    
    # Bottom right: Accuracy vs speed tradeoff
    ax4 = fig.add_subplot(gs[1, 1])
    
    methods = ['FP32\nBaseline', 'FP16\nNaive', 'FP16\n+ Scaling\n(AMP)']
    accuracy = [82.1, 73.5, 82.1]
    speedup = [1.0, 2.3, 2.0]
    
    x = np.arange(len(methods))
    width = 0.35
    
    ax4_acc = ax4
    ax4_speed = ax4.twinx()
    
    bars1 = ax4_acc.bar(x - width/2, accuracy, width, label='mAP@0.5', 
                        color='#3b82f6', edgecolor='white', linewidth=1, alpha=0.8)
    bars2 = ax4_speed.bar(x + width/2, speedup, width, label='Speedup', 
                          color='#10b981', edgecolor='white', linewidth=1, alpha=0.8)
    
    ax4_acc.set_xlabel('Method', fontsize=12, color='white')
    ax4_acc.set_ylabel('mAP@0.5 (%)', fontsize=12, color='#3b82f6')
    ax4_speed.set_ylabel('Speedup (×)', fontsize=12, color='#10b981')
    ax4_acc.set_title('Accuracy vs Speed Tradeoff', fontsize=12, color='white', weight='bold')
    ax4_acc.set_xticks(x)
    ax4_acc.set_xticklabels(methods)
    ax4_acc.set_ylim(70, 85)
    ax4_speed.set_ylim(0, 2.5)
    ax4_acc.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax4_acc.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9, color='white')
    
    for bar in bars2:
        height = bar.get_height()
        ax4_speed.text(bar.get_x() + bar.get_width()/2., height,
                      f'{height:.1f}×', ha='center', va='bottom', fontsize=9, color='white')
    
    # Combined legend
    lines1, labels1 = ax4_acc.get_legend_handles_labels()
    lines2, labels2 = ax4_speed.get_legend_handles_labels()
    ax4_acc.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=10)
    
    plt.tight_layout()
    
    # Save
    output_path = Path(__file__).parent.parent / 'img' / 'ch10-mixed-precision.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    print(f"✅ Generated: {output_path}")
    plt.close()

if __name__ == '__main__':
    create_mixed_precision_diagram()
