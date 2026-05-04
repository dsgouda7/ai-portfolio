"""
Generate progress check dashboard for Ch.6 Instance Segmentation.

Shows ProductionCV grand challenge status after completing Mask R-CNN:
- ✅ Constraint #1: Detection accuracy (mAP ≥ 85%) — ACHIEVED
- ✅ Constraint #2: Segmentation quality (IoU ≥ 70%) — ACHIEVED
- ❌ Constraint #3: Inference latency (<50ms) — Not yet optimized
- ❌ Constraint #4: Model size (<100 MB) — Too large (178 MB)
- ❌ Constraint #5: Data efficiency (<1k labels) — Still needs full dataset

Output: ch06-progress-check.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Dark theme
plt.style.use('dark_background')

def create_progress_dashboard():
    """Create comprehensive progress dashboard after Ch.6."""
    
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('#1a1a2e')
    
    gs = fig.add_gridspec(4, 3, hspace=0.5, wspace=0.3)
    
    # Title
    fig.suptitle('ProductionCV Progress After Ch.6 — Instance Segmentation (Mask R-CNN)', 
                fontsize=22, fontweight='bold', color='#ecf0f1', y=0.98)
    
    # Constraints
    constraints = [
        {'name': 'Detection Accuracy', 'metric': 'mAP@0.5', 'target': 0.85, 
         'current': 0.873, 'status': 'achieved', 'chapter': 'Ch.4-6'},
        {'name': 'Segmentation Quality', 'metric': 'IoU', 'target': 0.70, 
         'current': 0.712, 'status': 'achieved', 'chapter': 'Ch.5-6'},
        {'name': 'Inference Latency', 'metric': 'ms', 'target': 50, 
         'current': 95, 'status': 'not_started', 'chapter': 'Ch.9-10'},
        {'name': 'Model Size', 'metric': 'MB', 'target': 100, 
         'current': 178, 'status': 'not_started', 'chapter': 'Ch.9-10'},
        {'name': 'Data Efficiency', 'metric': 'labels', 'target': 1000, 
         'current': 1000, 'status': 'not_started', 'chapter': 'Ch.7-8'},
    ]
    
    status_colors = {'achieved': '#2ecc71', 'in_progress': '#f39c12', 'not_started': '#95a5a6'}
    status_symbols = {'achieved': '✅', 'in_progress': '⚡', 'not_started': '○'}
    
    # Constraint cards (top 2 rows)
    for idx, constraint in enumerate(constraints):
        if idx < 3:
            ax = fig.add_subplot(gs[0, idx])
        else:
            ax = fig.add_subplot(gs[1, idx-3])
        
        ax.axis('off')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        
        color = status_colors[constraint['status']]
        rect = patches.Rectangle((0.5, 0.5), 9, 9, linewidth=4, 
                                edgecolor=color, facecolor='#2c3e50', alpha=0.9)
        ax.add_patch(rect)
        
        # Status symbol
        symbol = status_symbols[constraint['status']]
        ax.text(5, 8.5, symbol, fontsize=36, ha='center', va='center')
        
        # Constraint name
        ax.text(5, 7, f"#{idx+1} {constraint['name']}", fontsize=13, ha='center', 
                fontweight='bold', color='#ecf0f1')
        
        # Current vs target
        if constraint['metric'] in ['mAP@0.5', 'IoU']:
            current_pct = constraint['current'] * 100
            target_pct = constraint['target'] * 100
            ax.text(5, 5, f"{current_pct:.1f}%", fontsize=32, ha='center', 
                    fontweight='bold', color=color)
            ax.text(5, 3.5, f"Target: {target_pct:.0f}%", fontsize=11, ha='center', 
                    color='#95a5a6')
        elif constraint['metric'] == 'ms':
            ax.text(5, 5, f"{constraint['current']}", fontsize=32, ha='center', 
                    fontweight='bold', color=color)
            ax.text(5, 3.5, f"Target: <{constraint['target']}", fontsize=11, ha='center', 
                    color='#95a5a6')
        elif constraint['metric'] == 'MB':
            ax.text(5, 5, f"{constraint['current']}", fontsize=32, ha='center', 
                    fontweight='bold', color=color)
            ax.text(5, 3.5, f"Target: <{constraint['target']}", fontsize=11, ha='center', 
                    color='#95a5a6')
        else:
            ax.text(5, 5, f"{constraint['current']}", fontsize=32, ha='center', 
                    fontweight='bold', color=color)
            ax.text(5, 3.5, f"Target: <{constraint['target']}", fontsize=11, ha='center', 
                    color='#95a5a6')
        
        ax.text(5, 2.5, constraint['metric'], fontsize=10, ha='center', 
                color='#95a5a6', style='italic')
        ax.text(5, 1.2, f"Unlock: {constraint['chapter']}", fontsize=9, ha='center', 
                color='#7f8c8d')
    
    # Key achievements (middle section)
    achievements_ax = fig.add_subplot(gs[2, :])
    achievements_ax.axis('off')
    achievements_ax.set_xlim(0, 3)
    achievements_ax.set_ylim(0, 3)
    
    achievements_ax.text(1.5, 2.5, '🎉 Major Milestones Achieved', fontsize=16, ha='center', 
                        fontweight='bold', color='#ecf0f1')
    
    achievements = [
        '✅ Instance-level product detection (count individual items)',
        '✅ Pixel-perfect segmentation masks (71.2% IoU > 70% target)',
        '✅ Per-SKU inventory tracking enabled',
        '✅ Planogram compliance verification (95% precision)',
        '⚡ Baseline for optimization (Ch.7-10 will improve speed/size/data)'
    ]
    
    y_pos = 2.0
    for achievement in achievements:
        emoji = achievement.split()[0]
        text = ' '.join(achievement.split()[1:])
        color = '#2ecc71' if emoji == '✅' else '#f39c12'
        achievements_ax.text(0.2, y_pos, emoji, fontsize=14, va='center')
        achievements_ax.text(0.4, y_pos, text, fontsize=11, va='center', color=color)
        y_pos -= 0.4
    
    # Timeline (bottom section)
    timeline_ax = fig.add_subplot(gs[3, :])
    timeline_ax.axis('off')
    timeline_ax.set_xlim(0, 12)
    timeline_ax.set_ylim(0, 4)
    
    timeline_ax.text(6, 3.8, 'ProductionCV Development Timeline', fontsize=15, ha='center', 
                    fontweight='bold', color='#ecf0f1')
    
    chapters = [
        ('Ch.1-2\nArchitecture', 'done'),
        ('Ch.3-4\nDetection', 'done'),
        ('Ch.5\nSemantic Seg', 'done'),
        ('Ch.6\nInstance Seg', 'current'),
        ('Ch.7-8\nSelf-Supervised', 'next'),
        ('Ch.9-10\nOptimization', 'future')
    ]
    
    for i, (label, status) in enumerate(chapters):
        x = 1 + i * 2
        
        if status == 'done':
            color = '#2ecc71'
            marker_style = 'o'
            alpha = 1.0
        elif status == 'current':
            color = '#f39c12'
            marker_style = 'o'
            alpha = 1.0
        else:
            color = '#95a5a6'
            marker_style = 'o'
            alpha = 0.5
        
        timeline_ax.plot(x, 2, marker=marker_style, markersize=22, color=color, alpha=alpha)
        timeline_ax.text(x, 0.8, label, fontsize=9, ha='center', va='top', color=color, alpha=alpha)
        
        # Connecting line
        if i < len(chapters) - 1:
            next_status = chapters[i + 1][1]
            line_color = '#2ecc71' if status == 'done' and next_status in ['done', 'current'] else '#95a5a6'
            timeline_ax.plot([x, x + 2], [2, 2], '-', linewidth=4, color=line_color, alpha=0.5)
    
    # Current position marker
    timeline_ax.text(7, 3.2, '👉 YOU ARE HERE', fontsize=13, ha='center', 
                    fontweight='bold', color='#f39c12')
    
    plt.tight_layout()
    return fig

def save_dashboard():
    """Generate and save progress dashboard."""
    fig = create_progress_dashboard()
    
    output_path = '../img/ch06-progress-check.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print(f"✅ Saved progress dashboard to {output_path}")
    
    plt.close()

if __name__ == '__main__':
    save_dashboard()
    print("Progress dashboard generation complete!")
