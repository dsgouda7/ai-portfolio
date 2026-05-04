"""
Generate Ch.4 CI/CD Pipeline Flow Diagram

Visualizes the complete CI/CD pipeline: commit → test → build → deploy

Output:
- ch04_cicd_pipeline.png (static diagram)
- ch04_cicd_pipeline.gif (animated flow)
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import io

# Configuration
OUTPUT_DIR = '../img'
FILENAME = 'ch04_cicd_pipeline'

# Colors
COLOR_TRIGGER = '#FF6B6B'  # Red
COLOR_TEST = '#4ECDC4'     # Teal
COLOR_BUILD = '#45B7D1'    # Blue
COLOR_DEPLOY = '#96CEB4'   # Green
COLOR_SUCCESS = '#5CB85C'  # Success green
COLOR_FAIL = '#D9534F'     # Error red
COLOR_ARROW = '#333333'
COLOR_BG = '#F8F9FA'

def create_stage_box(ax, x, y, width, height, label, color, stage_num):
    """Draw a pipeline stage box"""
    # Main box
    rect = patches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.05",
        edgecolor='#333',
        facecolor=color,
        linewidth=2
    )
    ax.add_patch(rect)
    
    # Stage number
    circle = plt.Circle((x + 0.3, y + height - 0.3), 0.15, 
                       color='white', zorder=10)
    ax.add_patch(circle)
    ax.text(x + 0.3, y + height - 0.3, str(stage_num),
           ha='center', va='center', fontsize=12, fontweight='bold',
           color='#333', zorder=11)
    
    # Label
    ax.text(x + width/2, y + height/2, label,
           ha='center', va='center', fontsize=11, fontweight='bold',
           color='white')

def draw_arrow(ax, x1, y1, x2, y2, label='', color=COLOR_ARROW):
    """Draw an arrow between stages"""
    ax.annotate('',
               xy=(x2, y2),
               xytext=(x1, y1),
               arrowprops=dict(
                   arrowstyle='->',
                   lw=2.5,
                   color=color,
                   connectionstyle='arc3,rad=0'
               ))
    
    if label:
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.2, label,
               ha='center', va='center', fontsize=9,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor='none', alpha=0.8))

def create_pipeline_diagram(highlight_stage=None, show_failure=False):
    """
    Create CI/CD pipeline diagram
    
    Args:
        highlight_stage: Which stage to highlight (0-4, or None for all)
        show_failure: If True, show failure path from test stage
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    
    # Title
    ax.text(6, 7.5, 'CI/CD Pipeline Flow', 
           ha='center', va='top', fontsize=16, fontweight='bold')
    
    # Stage positions
    stages = [
        {'x': 1, 'y': 5, 'w': 1.8, 'h': 1.2, 'label': 'Trigger\n(git push)', 
         'color': COLOR_TRIGGER, 'num': 1},
        {'x': 3.5, 'y': 5, 'w': 1.8, 'h': 1.2, 'label': 'Test\n(pytest)', 
         'color': COLOR_TEST, 'num': 2},
        {'x': 6, 'y': 5, 'w': 1.8, 'h': 1.2, 'label': 'Build\n(Docker)', 
         'color': COLOR_BUILD, 'num': 3},
        {'x': 8.5, 'y': 5, 'w': 1.8, 'h': 1.2, 'label': 'Deploy\n(K8s)', 
         'color': COLOR_DEPLOY, 'num': 4},
    ]
    
    # Draw stages
    for i, stage in enumerate(stages):
        # Determine if this stage should be highlighted
        if highlight_stage is not None:
            if i < highlight_stage:
                alpha = 0.3  # Completed stages (faded)
            elif i == highlight_stage:
                alpha = 1.0  # Current stage (bright)
            else:
                alpha = 0.15  # Future stages (very faded)
        else:
            alpha = 1.0  # All stages visible
        
        # Adjust color alpha
        color = stage['color']
        if alpha < 1.0:
            # Convert to RGB and add transparency
            from matplotlib.colors import to_rgba
            rgba = list(to_rgba(color))
            rgba[3] = alpha
            color = tuple(rgba)
        
        create_stage_box(ax, stage['x'], stage['y'], stage['w'], stage['h'],
                        stage['label'], color, stage['num'])
    
    # Draw arrows
    arrow_alpha = 1.0 if highlight_stage is None else 0.3
    
    # Trigger → Test
    draw_arrow(ax, 2.8, 5.6, 3.5, 5.6)
    
    # Test → Build (success path)
    if not show_failure:
        draw_arrow(ax, 5.3, 5.6, 6.0, 5.6, '✓ pass')
    else:
        # Show failure path
        draw_arrow(ax, 5.3, 5.6, 6.0, 5.6, '✓ pass', color=COLOR_SUCCESS)
        
        # Failure path
        ax.annotate('',
                   xy=(4.4, 4.2),
                   xytext=(4.4, 5.0),
                   arrowprops=dict(arrowstyle='->', lw=2.5, color=COLOR_FAIL))
        
        # Failure box
        fail_rect = patches.FancyBboxPatch(
            (3.5, 3.5), 1.8, 0.7,
            boxstyle="round,pad=0.05",
            edgecolor=COLOR_FAIL,
            facecolor='white',
            linewidth=2
        )
        ax.add_patch(fail_rect)
        ax.text(4.4, 3.85, '✗ Tests Failed\nStop Pipeline',
               ha='center', va='center', fontsize=9, color=COLOR_FAIL,
               fontweight='bold')
    
    # Build → Deploy
    draw_arrow(ax, 7.8, 5.6, 8.5, 5.6)
    
    # Success indicator
    success_y = 5.0 if not show_failure else 5.8
    ax.text(10.5, success_y, '✓', fontsize=30, color=COLOR_SUCCESS, 
           ha='center', va='center')
    ax.text(10.5, success_y - 0.5, 'Live!', fontsize=11, 
           ha='center', va='center', fontweight='bold')
    
    # Details boxes
    details = [
        {
            'y': 3.5,
            'items': [
                'Trigger: Every push to main branch',
                'Runner: GitHub-hosted Ubuntu VM',
                'Secrets: Docker Hub credentials',
                'Duration: ~3-5 minutes total'
            ]
        },
        {
            'y': 1.5,
            'items': [
                'Test: pytest + flake8 linter',
                'Build: docker build + push to registry',
                'Deploy: kubectl set image (rolling update)',
                'Rollback: Automatic if health checks fail'
            ]
        }
    ]
    
    for detail in details:
        y_pos = detail['y']
        for i, item in enumerate(detail['items']):
            ax.text(1, y_pos - i*0.3, f'• {item}',
                   ha='left', va='top', fontsize=9, 
                   color='#555', style='italic')
    
    plt.tight_layout()
    return fig

def create_animation():
    """Create animated GIF showing pipeline progression"""
    frames = []
    
    # Frame 1: Full pipeline
    fig = create_pipeline_diagram()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', 
               facecolor=COLOR_BG)
    buf.seek(0)
    frames.append(Image.open(buf).copy())
    plt.close(fig)
    
    # Frames 2-5: Highlight each stage
    for stage in range(5):
        fig = create_pipeline_diagram(highlight_stage=stage)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                   facecolor=COLOR_BG)
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        plt.close(fig)
    
    # Frame 6: Show failure path
    fig = create_pipeline_diagram(show_failure=True)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
               facecolor=COLOR_BG)
    buf.seek(0)
    frames.append(Image.open(buf).copy())
    plt.close(fig)
    
    # Frame 7: Back to success
    fig = create_pipeline_diagram()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
               facecolor=COLOR_BG)
    buf.seek(0)
    frames.append(Image.open(buf).copy())
    plt.close(fig)
    
    # Save GIF
    frames[0].save(
        f'{OUTPUT_DIR}/{FILENAME}.gif',
        save_all=True,
        append_images=frames[1:],
        duration=[1500, 800, 800, 800, 800, 1500, 1500],  # ms per frame
        loop=0
    )
    print(f"✅ Created {OUTPUT_DIR}/{FILENAME}.gif")

if __name__ == '__main__':
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Static diagram
    fig = create_pipeline_diagram()
    fig.savefig(f'{OUTPUT_DIR}/{FILENAME}.png', dpi=150, bbox_inches='tight',
               facecolor=COLOR_BG)
    plt.close(fig)
    print(f"✅ Created {OUTPUT_DIR}/{FILENAME}.png")
    
    # Animated GIF
    create_animation()
