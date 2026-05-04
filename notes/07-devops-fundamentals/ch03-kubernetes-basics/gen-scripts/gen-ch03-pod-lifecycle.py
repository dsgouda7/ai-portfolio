"""
Generate Pod lifecycle state machine diagram.

Usage:
    python gen_ch03_pod_lifecycle.py

Output:
    ../img/ch03-pod-lifecycle.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib
import numpy as np
matplotlib.use('Agg')  # Non-interactive backend

# Color palette
COLOR_PENDING = '#F5A623'    # Orange
COLOR_RUNNING = '#7ED321'    # Green
COLOR_SUCCEEDED = '#4A90E2'  # Blue
COLOR_FAILED = '#D0021B'     # Red
COLOR_UNKNOWN = '#9013FE'    # Purple
COLOR_TEXT = '#2C3E50'       # Dark text

def create_pod_lifecycle_diagram():
    """Create pod lifecycle state machine diagram."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    fig.patch.set_facecolor('white')
    
    # Title
    ax.text(7, 9.5, 'Pod Lifecycle States', 
            ha='center', va='top', fontsize=18, fontweight='bold', color=COLOR_TEXT)
    
    # ========== STATES ==========
    states = [
        ('Pending', 2, 7, COLOR_PENDING, 
         'Accepted by K8s but\ncontainer not yet started.\nWaiting for scheduling\nor image pull.'),
        ('Running', 7, 7, COLOR_RUNNING, 
         'Pod bound to node,\nall containers created.\nAt least one container\nstill running.'),
        ('Succeeded', 12, 7, COLOR_SUCCEEDED, 
         'All containers terminated\nsuccessfully (exit code 0).\nWon\'t be restarted.'),
        ('Failed', 9.5, 3.5, COLOR_FAILED, 
         'All containers terminated,\nat least one failed\n(non-zero exit code).'),
        ('Unknown', 4.5, 3.5, COLOR_UNKNOWN, 
         'State cannot be determined.\nUsually network issue\nwith node.'),
    ]
    
    state_boxes = {}
    for name, x, y, color, desc in states:
        # State box
        box = FancyBboxPatch(
            (x - 1.2, y - 0.8), 2.4, 1.6,
            boxstyle="round,pad=0.1",
            edgecolor=color,
            facecolor=color,
            alpha=0.3,
            linewidth=3
        )
        ax.add_patch(box)
        
        # State name
        ax.text(x, y + 0.4, name, ha='center', va='center', 
                fontsize=14, fontweight='bold', color=color)
        
        # Description
        ax.text(x, y - 0.2, desc, ha='center', va='top', 
                fontsize=8, color=COLOR_TEXT, style='italic')
        
        state_boxes[name] = (x, y)
    
    # ========== TRANSITIONS ==========
    transitions = [
        # (from_state, to_state, label, curvature)
        ('Pending', 'Running', 'Container images\npulled & started', 0),
        ('Running', 'Succeeded', 'All containers\nexit successfully', 0),
        ('Running', 'Failed', 'Container crashes\n(exit code ≠ 0)', -0.3),
        ('Pending', 'Failed', 'Image pull fails\nor invalid config', -0.4),
        ('Pending', 'Unknown', 'Node loses\nconnection', 0.3),
        ('Running', 'Unknown', 'Node\nunreachable', 0.3),
    ]
    
    for from_state, to_state, label, curve in transitions:
        x1, y1 = state_boxes[from_state]
        x2, y2 = state_boxes[to_state]
        
        # Adjust arrow start/end points to edge of boxes
        dx = x2 - x1
        dy = y2 - y1
        dist = (dx**2 + dy**2)**0.5
        
        # Normalize and scale
        offset = 1.3  # Distance from center to edge
        x1_adj = x1 + (dx / dist) * offset
        y1_adj = y1 + (dy / dist) * offset
        x2_adj = x2 - (dx / dist) * offset
        y2_adj = y2 - (dy / dist) * offset
        
        # Create curved arrow
        arrow = FancyArrowPatch(
            (x1_adj, y1_adj), (x2_adj, y2_adj),
            arrowstyle='->', mutation_scale=25,
            linewidth=2, color=COLOR_TEXT, alpha=0.7,
            connectionstyle=f"arc3,rad={curve}"
        )
        ax.add_patch(arrow)
        
        # Label
        mid_x = (x1_adj + x2_adj) / 2
        mid_y = (y1_adj + y2_adj) / 2
        
        # Offset label perpendicular to arrow
        perp_x = -(y2_adj - y1_adj) / dist * 0.5
        perp_y = (x2_adj - x1_adj) / dist * 0.5
        
        ax.text(mid_x + perp_x, mid_y + perp_y, label, 
                ha='center', va='center', fontsize=8, 
                color=COLOR_TEXT, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='lightgray', alpha=0.9))
    
    # ========== ENTRY POINT ==========
    # Start circle
    start_circle = Circle((0.8, 7), 0.3, color=COLOR_TEXT, alpha=0.5)
    ax.add_patch(start_circle)
    ax.text(0.8, 7, 'START', ha='center', va='center', 
            fontsize=8, color='white', fontweight='bold')
    
    # Arrow to Pending
    arrow = FancyArrowPatch(
        (1.1, 7), (state_boxes['Pending'][0] - 1.3, state_boxes['Pending'][1]),
        arrowstyle='->', mutation_scale=25,
        linewidth=2, color=COLOR_TEXT, alpha=0.7
    )
    ax.add_patch(arrow)
    
    # ========== RESTART BEHAVIOR ==========
    # Self-loop on Running
    running_x, running_y = state_boxes['Running']
    
    # Draw arc above Running state
    arc_points = []
    for i in range(20):
        angle = 3.14 + i * 3.14 / 19  # Half circle above
        r = 0.8
        x = running_x + r * np.cos(angle)
        y = running_y + 0.8 + r * np.sin(angle)
        arc_points.append((x, y))
    
    # Draw arc
    for i in range(len(arc_points) - 1):
        ax.plot([arc_points[i][0], arc_points[i+1][0]], 
               [arc_points[i][1], arc_points[i+1][1]], 
               color=COLOR_RUNNING, linewidth=2, alpha=0.7)
    
    # Arrow head
    arrow = FancyArrowPatch(
        arc_points[-2], arc_points[-1],
        arrowstyle='->', mutation_scale=20,
        linewidth=2, color=COLOR_RUNNING, alpha=0.7
    )
    ax.add_patch(arrow)
    
    ax.text(running_x, running_y + 1.8, 'restartPolicy=Always\n(pod restarts on failure)', 
            ha='center', va='bottom', fontsize=8, color=COLOR_RUNNING, 
            fontweight='bold', style='italic')
    
    # ========== COMMON SCENARIOS ==========
    ax.text(0.5, 2.2, 'Common Scenarios:', fontsize=11, 
            fontweight='bold', color=COLOR_TEXT)
    
    scenarios = [
        ('Pending → Failed', 'ImagePullBackOff (can\'t pull Docker image)'),
        ('Running → Failed', 'CrashLoopBackOff (app crashes immediately)'),
        ('Pending → Running → Succeeded', 'Batch job completes successfully'),
        ('Running → Unknown', 'Node loses network connection'),
    ]
    
    for i, (path, desc) in enumerate(scenarios):
        y = 1.8 - i * 0.3
        ax.text(0.7, y, f'• {path}:', fontsize=8, 
                fontweight='bold', color=COLOR_TEXT)
        ax.text(3.5, y, desc, fontsize=8, 
                color=COLOR_TEXT, style='italic')
    
    # ========== DEBUG COMMANDS ==========
    ax.text(8, 2.2, 'Debug Commands:', fontsize=11, 
            fontweight='bold', color=COLOR_TEXT)
    
    commands = [
        ('kubectl get pods', 'Check current status'),
        ('kubectl describe pod <name>', 'Detailed events & state'),
        ('kubectl logs <name>', 'Application stdout/stderr'),
        ('kubectl logs <name> --previous', 'Logs from crashed container'),
    ]
    
    for i, (cmd, desc) in enumerate(commands):
        y = 1.8 - i * 0.3
        ax.text(8.2, y, f'{cmd}', fontsize=8, 
                fontweight='bold', color='#E74C3C', family='monospace')
        ax.text(11.5, y, f'→ {desc}', fontsize=8, 
                color=COLOR_TEXT, style='italic')
    
    # ========== NOTES ==========
    note_box = FancyBboxPatch(
        (0.3, 0.1), 13.4, 0.5,
        boxstyle="round,pad=0.1",
        edgecolor='gray',
        facecolor='lightyellow',
        alpha=0.5,
        linewidth=1
    )
    ax.add_patch(note_box)
    
    ax.text(7, 0.35, 
            '💡 Deployment/ReplicaSet restart policy is "Always" — Failed pods automatically restart. '
            'Jobs use "OnFailure" or "Never".', 
            ha='center', va='center', fontsize=9, color=COLOR_TEXT, style='italic')
    
    plt.tight_layout()
    return fig

if __name__ == '__main__':
    import os
    print("Generating Pod lifecycle diagram...")
    fig = create_pod_lifecycle_diagram()
    
    # Get script directory and construct path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, '../img/ch03-pod-lifecycle.png')
    output_path = os.path.normpath(output_path)
    
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path}")
    
    plt.close()
