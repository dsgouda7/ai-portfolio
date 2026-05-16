"""
Generate Ch.4 Workflow Triggers Diagram

Visualizes different trigger types: push, PR, schedule, manual

Output:
- ch04_workflow_triggers.png (static diagram)
- ch04_workflow_triggers.gif (animated demonstration)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import io

# Configuration
OUTPUT_DIR = '../img'
FILENAME = 'ch04_workflow_triggers'

# Colors
COLOR_PUSH = '#FF6B6B'
COLOR_PR = '#4ECDC4'
COLOR_SCHEDULE = '#FFD93D'
COLOR_MANUAL = '#A8E6CF'
COLOR_WORKFLOW = '#6C5CE7'
COLOR_BG = '#F8F9FA'

def create_trigger_icon(ax, x, y, trigger_type, color, scale=1.0):
    """Draw an icon for each trigger type"""
    if trigger_type == 'push':
        # Git commit icon (circle with arrow)
        circle = plt.Circle((x, y), 0.3 * scale, color=color, zorder=10)
        ax.add_patch(circle)
        ax.annotate('',
                   xy=(x + 0.5 * scale, y),
                   xytext=(x + 0.1 * scale, y),
                   arrowprops=dict(arrowstyle='->', lw=2, color='white'))
        
    elif trigger_type == 'pr':
        # Pull request icon (merge symbol)
        # Two circles connected by lines
        c1 = plt.Circle((x - 0.2 * scale, y + 0.15 * scale), 0.12 * scale, 
                       color=color, zorder=10)
        c2 = plt.Circle((x + 0.2 * scale, y + 0.15 * scale), 0.12 * scale, 
                       color=color, zorder=10)
        c3 = plt.Circle((x, y - 0.15 * scale), 0.12 * scale, 
                       color=color, zorder=10)
        ax.add_patch(c1)
        ax.add_patch(c2)
        ax.add_patch(c3)
        
        # Connecting lines
        ax.plot([x - 0.2 * scale, x], [y + 0.15 * scale, y - 0.15 * scale],
               color=color, lw=2, zorder=9)
        ax.plot([x + 0.2 * scale, x], [y + 0.15 * scale, y - 0.15 * scale],
               color=color, lw=2, zorder=9)
    
    elif trigger_type == 'schedule':
        # Clock icon
        circle = plt.Circle((x, y), 0.3 * scale, color=color, zorder=10)
        ax.add_patch(circle)
        # Clock hands
        ax.plot([x, x], [y, y + 0.2 * scale], color='white', lw=2, zorder=11)
        ax.plot([x, x + 0.15 * scale], [y, y], color='white', lw=2, zorder=11)
    
    elif trigger_type == 'manual':
        # Hand/button icon (rectangle with finger)
        rect = patches.Rectangle((x - 0.25 * scale, y - 0.2 * scale), 
                                0.5 * scale, 0.4 * scale,
                                facecolor=color, edgecolor='white', 
                                linewidth=2, zorder=10)
        ax.add_patch(rect)
        ax.text(x, y, '▶', color='white', fontsize=14 * scale,
               ha='center', va='center', zorder=11)

def create_trigger_box(ax, x, y, width, height, trigger_type, label, 
                       sublabel, color, highlight=False):
    """Draw a trigger box with icon and labels"""
    alpha = 1.0 if highlight else 0.4
    
    # Main box
    rect = patches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.1",
        edgecolor='#333' if highlight else '#999',
        facecolor='white',
        linewidth=3 if highlight else 1.5,
        alpha=alpha
    )
    ax.add_patch(rect)
    
    # Icon
    icon_x = x + 0.5
    icon_y = y + height - 0.6
    create_trigger_icon(ax, icon_x, icon_y, trigger_type, color, 
                       scale=1.0 if highlight else 0.8)
    
    # Labels
    label_alpha = 1.0 if highlight else 0.6
    ax.text(x + width/2, y + height - 1.3, label,
           ha='center', va='top', fontsize=12, fontweight='bold',
           color='#333', alpha=label_alpha)
    ax.text(x + width/2, y + 0.3, sublabel,
           ha='center', va='bottom', fontsize=8, color='#666',
           style='italic', alpha=label_alpha)

def create_triggers_diagram(highlight=None):
    """
    Create workflow triggers diagram
    
    Args:
        highlight: Which trigger to highlight ('push', 'pr', 'schedule', 'manual', or None)
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    
    # Title
    ax.text(6, 7.5, 'GitHub Actions Workflow Triggers', 
           ha='center', va='top', fontsize=16, fontweight='bold')
    
    # Trigger boxes
    triggers = [
        {
            'x': 0.5, 'y': 4.5, 'w': 2.5, 'h': 2,
            'type': 'push', 'label': 'Push', 
            'sublabel': 'on: push', 'color': COLOR_PUSH
        },
        {
            'x': 3.5, 'y': 4.5, 'w': 2.5, 'h': 2,
            'type': 'pr', 'label': 'Pull Request',
            'sublabel': 'on: pull_request', 'color': COLOR_PR
        },
        {
            'x': 6.5, 'y': 4.5, 'w': 2.5, 'h': 2,
            'type': 'schedule', 'label': 'Schedule',
            'sublabel': 'on: schedule', 'color': COLOR_SCHEDULE
        },
        {
            'x': 9.5, 'y': 4.5, 'w': 2.5, 'h': 2,
            'type': 'manual', 'label': 'Manual',
            'sublabel': 'on: workflow_dispatch', 'color': COLOR_MANUAL
        }
    ]
    
    # Draw triggers
    for trigger in triggers:
        is_highlighted = (highlight == trigger['type']) or (highlight is None)
        create_trigger_box(ax, trigger['x'], trigger['y'], trigger['w'], 
                          trigger['h'], trigger['type'], trigger['label'],
                          trigger['sublabel'], trigger['color'], is_highlighted)
    
    # Central workflow box
    workflow_y = 2.5
    workflow_rect = patches.FancyBboxPatch(
        (4, workflow_y), 4, 1.2,
        boxstyle="round,pad=0.1",
        edgecolor=COLOR_WORKFLOW,
        facecolor=COLOR_WORKFLOW,
        linewidth=2
    )
    ax.add_patch(workflow_rect)
    ax.text(6, workflow_y + 0.6, 'Workflow Execution',
           ha='center', va='center', fontsize=13, fontweight='bold',
           color='white')
    ax.text(6, workflow_y + 0.2, 'test → build → deploy',
           ha='center', va='center', fontsize=9, color='white',
           style='italic')
    
    # Arrows from triggers to workflow
    arrow_targets = [
        (1.75, 4.5),   # Push
        (4.75, 4.5),   # PR
        (7.75, 4.5),   # Schedule
        (10.75, 4.5)   # Manual
    ]
    
    for i, (tx, ty) in enumerate(arrow_targets):
        alpha = 1.0 if (highlight == triggers[i]['type'] or highlight is None) else 0.3
        ax.annotate('',
                   xy=(6, workflow_y + 1.2),
                   xytext=(tx, ty),
                   arrowprops=dict(
                       arrowstyle='->',
                       lw=2,
                       color=triggers[i]['color'],
                       alpha=alpha,
                       connectionstyle='arc3,rad=0.2'
                   ))
    
    # Details for each trigger type
    details_y = 1.3
    details = [
        {
            'title': 'Push Trigger',
            'desc': 'Runs on every commit pushed to specified branches',
            'yaml': 'on:\\n  push:\\n    branches: [main]',
            'use': 'CI for main branch'
        },
        {
            'title': 'Pull Request Trigger',
            'desc': 'Runs on PR open/update to validate before merge',
            'yaml': 'on:\\n  pull_request:\\n    branches: [main]',
            'use': 'Pre-merge validation'
        },
        {
            'title': 'Schedule Trigger',
            'desc': 'Runs on cron schedule (e.g., nightly builds)',
            'yaml': 'on:\\n  schedule:\\n    - cron: "0 0 * * *"',
            'use': 'Nightly regression tests'
        },
        {
            'title': 'Manual Trigger',
            'desc': 'Runs on button click in GitHub UI',
            'yaml': 'on:\\n  workflow_dispatch:\\n    inputs: {...}',
            'use': 'Production deployments'
        }
    ]
    
    if highlight is not None:
        # Show details for highlighted trigger
        idx = {'push': 0, 'pr': 1, 'schedule': 2, 'manual': 3}[highlight]
        detail = details[idx]
        
        # Detail box
        detail_rect = patches.FancyBboxPatch(
            (1, 0.2), 10, 1.0,
            boxstyle="round,pad=0.1",
            edgecolor='#333',
            facecolor='white',
            linewidth=2
        )
        ax.add_patch(detail_rect)
        
        ax.text(1.5, 0.95, f"📌 {detail['title']}",
               ha='left', va='top', fontsize=11, fontweight='bold')
        ax.text(1.5, 0.65, detail['desc'],
               ha='left', va='top', fontsize=9, color='#555')
        ax.text(6.5, 0.95, 'YAML:',
               ha='left', va='top', fontsize=9, fontweight='bold')
        ax.text(6.5, 0.75, detail['yaml'],
               ha='left', va='top', fontsize=8, family='monospace',
               color='#333')
        ax.text(9.5, 0.95, f"Use case:",
               ha='left', va='top', fontsize=9, fontweight='bold')
        ax.text(9.5, 0.65, detail['use'],
               ha='left', va='top', fontsize=9, color='#555')
    else:
        # Show summary when nothing highlighted
        ax.text(6, 0.7, 'All triggers execute the same workflow — only the event differs',
               ha='center', va='center', fontsize=10, style='italic',
               color='#666')
        ax.text(6, 0.3, 'Combine multiple triggers: on: [push, pull_request, schedule]',
               ha='center', va='center', fontsize=9, color='#888',
               family='monospace')
    
    plt.tight_layout()
    return fig

def create_animation():
    """Create animated GIF showing each trigger type"""
    frames = []
    
    # Frame 1: All triggers
    fig = create_triggers_diagram()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
               facecolor=COLOR_BG)
    buf.seek(0)
    frames.append(Image.open(buf).copy())
    plt.close(fig)
    
    # Frames 2-5: Highlight each trigger
    for trigger in ['push', 'pr', 'schedule', 'manual']:
        fig = create_triggers_diagram(highlight=trigger)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                   facecolor=COLOR_BG)
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        plt.close(fig)
    
    # Frame 6: Back to all
    fig = create_triggers_diagram()
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
        duration=[1500, 1200, 1200, 1200, 1200, 1500],  # ms per frame
        loop=0
    )
    print(f"✅ Created {OUTPUT_DIR}/{FILENAME}.gif")

if __name__ == '__main__':
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Static diagram
    fig = create_triggers_diagram()
    fig.savefig(f'{OUTPUT_DIR}/{FILENAME}.png', dpi=150, bbox_inches='tight',
               facecolor=COLOR_BG)
    plt.close(fig)
    print(f"✅ Created {OUTPUT_DIR}/{FILENAME}.png")
    
    # Animated GIF
    create_animation()
