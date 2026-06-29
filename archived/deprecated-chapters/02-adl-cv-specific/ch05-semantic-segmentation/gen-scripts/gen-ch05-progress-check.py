"""
Generate progress check dashboard for Ch.5 Semantic Segmentation.

Visualizes ProductionCV grand challenge progress after Ch.5:
- Constraint #1: Detection accuracy (mAP) - already achieved in Ch.4
- Constraint #2: Segmentation quality (IoU) - in progress at 62%
- Constraint #3: Inference latency - not yet optimized
- Constraint #4: Model size - baseline established
- Constraint #5: Data efficiency - not yet addressed

Output: ch05-progress-check.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Dark theme
plt.style.use('dark_background')

def create_progress_dashboard():
    """Create progress dashboard showing constraint achievement."""
    
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#1a1a2e')
    
    # Create grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
    
    # Title
    fig.suptitle('ProductionCV Progress After Ch.5 — Semantic Segmentation', 
                fontsize=20, fontweight='bold', color='#ecf0f1', y=0.98)
    
    # Constraint status
    constraints = [
        {
            'name': 'Detection Accuracy',
            'metric': 'mAP@0.5',
            'target': 0.85,
            'current': 0.853,
            'status': 'achieved',
            'chapter': 'Ch.4'
        },
        {
            'name': 'Segmentation Quality',
            'metric': 'IoU',
            'target': 0.70,
            'current': 0.624,
            'status': 'in_progress',
            'chapter': 'Ch.5'
        },
        {
            'name': 'Inference Latency',
            'metric': 'ms per frame',
            'target': 50,
            'current': 95,
            'status': 'not_started',
            'chapter': 'Ch.9-10'
        },
        {
            'name': 'Model Size',
            'metric': 'MB',
            'target': 100,
            'current': 178,
            'status': 'not_started',
            'chapter': 'Ch.9-10'
        },
        {
            'name': 'Data Efficiency',
            'metric': 'labeled images',
            'target': 1000,
            'current': 1000,
            'status': 'not_started',
            'chapter': 'Ch.7-8'
        }
    ]
    
    # Colors for status
    status_colors = {
        'achieved': '#2ecc71',
        'in_progress': '#f39c12',
        'not_started': '#95a5a6'
    }
    
    status_symbols = {
        'achieved': '✅',
        'in_progress': '⚡',
        'not_started': '○'
    }
    
    # Main constraint cards (2x3 grid)
    for idx, constraint in enumerate(constraints):
        if idx < 3:
            ax = fig.add_subplot(gs[0, idx])
        else:
            ax = fig.add_subplot(gs[1, idx-3])
        
        ax.axis('off')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        
        # Card background
        color = status_colors[constraint['status']]
        rect = patches.Rectangle((0.5, 0.5), 9, 9, linewidth=3, 
                                edgecolor=color, facecolor='#2c3e50', alpha=0.8)
        ax.add_patch(rect)
        
        # Status symbol
        symbol = status_symbols[constraint['status']]
        ax.text(5, 8.5, symbol, fontsize=32, ha='center', va='center')
        
        # Constraint name
        ax.text(5, 7, f"#{idx+1} {constraint['name']}", fontsize=14, ha='center', 
                fontweight='bold', color='#ecf0f1')
        
        # Metric display
        if constraint['metric'] in ['mAP@0.5', 'IoU']:
            # Percentage metrics
            current_pct = constraint['current'] * 100
            target_pct = constraint['target'] * 100
            ax.text(5, 5, f"{current_pct:.1f}%", fontsize=28, ha='center', 
                    fontweight='bold', color=color)
            ax.text(5, 3.5, f"Target: {target_pct:.0f}%", fontsize=12, ha='center', color='#95a5a6')
        else:
            # Numeric metrics
            ax.text(5, 5, f"{constraint['current']}", fontsize=28, ha='center', 
                    fontweight='bold', color=color)
            ax.text(5, 3.5, f"Target: {'<' if idx in [2,3] else ''}{constraint['target']}", 
                    fontsize=12, ha='center', color='#95a5a6')
        
        ax.text(5, 2.5, constraint['metric'], fontsize=11, ha='center', 
                color='#95a5a6', style='italic')
        
        # Chapter unlock
        ax.text(5, 1.2, f"Unlocked: {constraint['chapter']}", fontsize=10, ha='center', 
                color='#7f8c8d')
    
    # Bottom section: Timeline visualization
    timeline_ax = fig.add_subplot(gs[2, :])
    timeline_ax.axis('off')
    timeline_ax.set_xlim(0, 12)
    timeline_ax.set_ylim(0, 4)
    
    timeline_ax.text(6, 3.5, 'ProductionCV Capability Timeline', fontsize=16, ha='center', 
                    fontweight='bold', color='#ecf0f1')
    
    chapters = ['Ch.1-2\nArchitecture', 'Ch.3-4\nDetection', 'Ch.5\nSegmentation', 
                'Ch.6\nInstances', 'Ch.7-8\nSelf-Supervised', 'Ch.9-10\nOptimization']
    chapter_status = ['done', 'done', 'current', 'next', 'future', 'future']
    
    for i, (chapter, status) in enumerate(zip(chapters, chapter_status)):
        x = 1 + i * 2
        
        # Timeline node
        if status == 'done':
            color = '#2ecc71'
            marker_style = 'o'
        elif status == 'current':
            color = '#f39c12'
            marker_style = 'o'
        else:
            color = '#95a5a6'
            marker_style = 'o'
        
        timeline_ax.plot(x, 2, marker=marker_style, markersize=20, color=color)
        timeline_ax.text(x, 0.8, chapter, fontsize=9, ha='center', va='top', color=color)
        
        # Connecting line
        if i < len(chapters) - 1:
            next_status = chapter_status[i + 1]
            line_color = '#2ecc71' if status == 'done' and next_status == 'done' else '#95a5a6'
            timeline_ax.plot([x, x + 2], [2, 2], '-', linewidth=3, color=line_color, alpha=0.5)
    
    # Current chapter highlight
    timeline_ax.text(5, 3, '👉 YOU ARE HERE', fontsize=12, ha='center', 
                    fontweight='bold', color='#f39c12')
    
    plt.tight_layout()
    return fig

def save_dashboard():
    """Generate and save progress dashboard."""
    fig = create_progress_dashboard()
    
    output_path = '../img/ch05-progress-check.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print(f"✅ Saved progress dashboard to {output_path}")
    
    plt.close()

if __name__ == '__main__':
    save_dashboard()
    print("Progress dashboard generation complete!")
