"""
Generate ProductionCV victory dashboard for Ch.10.
Shows all 5 constraints achieved.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from pathlib import Path

plt.style.use('dark_background')
DARK_BG = '#1a1a2e'

def create_victory_dashboard():
    """Create final victory dashboard showing all constraints met."""
    fig = plt.figure(figsize=(16, 10), facecolor=DARK_BG)
    fig.patch.set_facecolor(DARK_BG)
    
    fig.suptitle('🎯 ProductionCV Grand Challenge: ALL 5 CONSTRAINTS ACHIEVED!',
                 fontsize=20, color='#10b981', weight='bold', y=0.98)
    
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    # Title
    ax.text(5, 9.2, 'Final Model: Pruned MobileNetV2 + Mixed Precision (AMP)',
            ha='center', fontsize=16, color='white', weight='bold')
    
    # Constraint boxes
    constraints = [
        {
            'num': '1', 'name': 'Detection Accuracy', 
            'target': 'mAP@0.5 ≥ 85%', 'actual': '87.3%', 
            'status': '✅', 'color': '#10b981', 'y': 7.5
        },
        {
            'num': '2', 'name': 'Segmentation Accuracy',
            'target': 'mIoU ≥ 70%', 'actual': '71.2%',
            'status': '✅', 'color': '#10b981', 'y': 6.2
        },
        {
            'num': '3', 'name': 'Inference Latency',
            'target': '< 50ms', 'actual': '35ms',
            'status': '✅', 'color': '#10b981', 'y': 4.9
        },
        {
            'num': '4', 'name': 'Model Size',
            'target': '< 100 MB', 'actual': '6.8 MB',
            'status': '✅', 'color': '#10b981', 'y': 3.6
        },
        {
            'num': '5', 'name': 'Data Efficiency',
            'target': '< 1,000 labels', 'actual': '850 labels',
            'status': '✅', 'color': '#10b981', 'y': 2.3
        }
    ]
    
    for c in constraints:
        # Main box
        box = FancyBboxPatch((0.5, c['y']-0.45), 9, 0.9, 
                            boxstyle="round,pad=0.05",
                            edgecolor=c['color'], facecolor='#2d3748', 
                            linewidth=3, alpha=0.9)
        ax.add_patch(box)
        
        # Constraint number circle
        circle = Circle((1.2, c['y']), 0.3, color=c['color'], ec='white', linewidth=2, zorder=10)
        ax.add_patch(circle)
        ax.text(1.2, c['y'], c['num'], ha='center', va='center', 
               fontsize=16, color='white', weight='bold', zorder=11)
        
        # Constraint name
        ax.text(2.0, c['y']+0.15, c['name'], ha='left', va='center',
               fontsize=13, color='white', weight='bold')
        
        # Target
        ax.text(2.0, c['y']-0.15, f"Target: {c['target']}", ha='left', va='center',
               fontsize=10, color='#aaa')
        
        # Status
        ax.text(8.5, c['y'], c['status'], ha='center', va='center',
               fontsize=24, color=c['color'], weight='bold')
        
        # Actual value
        ax.text(7.2, c['y'], c['actual'], ha='right', va='center',
               fontsize=16, color=c['color'], weight='bold')
    
    # Journey summary at bottom
    journey_text = (
        "Journey: ResNet-50 (97 MB, 78ms) → MobileNetV2 (14 MB, 35ms) → \n"
        "Distilled (10.7 MB, 83.2% mAP) → Pruned 80% (6.8 MB) → AMP (35ms) ✅"
    )
    ax.text(5, 1.0, journey_text, ha='center', fontsize=11, color='#aaa',
           bbox=dict(boxstyle='round', facecolor='#1e3a1a', alpha=0.9, 
                    edgecolor='#10b981', linewidth=2))
    
    # Deployment status
    ax.text(5, 0.3, '🚀 Ready for Production Deployment — 500 Retail Stores — Jetson Nano',
           ha='center', fontsize=14, color='#10b981', weight='bold')
    
    plt.tight_layout()
    
    # Save
    output_path = Path(__file__).parent.parent / 'img' / 'ch10-productioncv-victory.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    print(f"✅ Generated: {output_path}")
    plt.close()

if __name__ == '__main__':
    create_victory_dashboard()
