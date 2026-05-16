"""
Generate Kubernetes architecture diagram showing control plane and worker nodes.

Usage:
    python gen_ch03_k8s_architecture.py

Output:
    ../img/ch03-k8s-architecture.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Color palette
COLOR_CONTROL = '#4A90E2'  # Blue for control plane
COLOR_WORKER = '#7ED321'   # Green for worker nodes
COLOR_POD = '#F5A623'      # Orange for pods
COLOR_TEXT = '#2C3E50'     # Dark text
COLOR_BG = '#ECF0F1'       # Light gray background

def create_architecture_diagram():
    """Create K8s cluster architecture diagram."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    fig.patch.set_facecolor('white')
    
    # Title
    ax.text(7, 9.5, 'Kubernetes Cluster Architecture', 
            ha='center', va='top', fontsize=18, fontweight='bold', color=COLOR_TEXT)
    
    # ========== CONTROL PLANE ==========
    control_plane_box = FancyBboxPatch(
        (0.5, 6), 5, 2.8, 
        boxstyle="round,pad=0.1", 
        edgecolor=COLOR_CONTROL, 
        facecolor=COLOR_CONTROL, 
        alpha=0.2, 
        linewidth=2
    )
    ax.add_patch(control_plane_box)
    
    ax.text(3, 8.6, 'Control Plane (Master Node)', 
            ha='center', va='top', fontsize=12, fontweight='bold', color=COLOR_CONTROL)
    
    # Control plane components
    components = [
        ('API Server', 1.5, 7.8, 'Gateway for all\ncluster operations'),
        ('Scheduler', 3, 7.8, 'Assigns pods\nto nodes'),
        ('Controller\nManager', 4.5, 7.8, 'Maintains\ndesired state'),
        ('etcd', 1.5, 6.6, 'Distributed\nkey-value store'),
    ]
    
    for name, x, y, desc in components:
        # Component box
        comp_box = FancyBboxPatch(
            (x - 0.4, y - 0.25), 0.8, 0.5,
            boxstyle="round,pad=0.05",
            edgecolor=COLOR_CONTROL,
            facecolor='white',
            linewidth=1.5
        )
        ax.add_patch(comp_box)
        ax.text(x, y, name, ha='center', va='center', fontsize=9, 
                fontweight='bold', color=COLOR_CONTROL)
        ax.text(x, y - 0.6, desc, ha='center', va='top', fontsize=7, 
                color=COLOR_TEXT, style='italic')
    
    # ========== WORKER NODES ==========
    worker_nodes = [
        (1.5, 3.5, 'Worker Node 1'),
        (5.5, 3.5, 'Worker Node 2'),
        (9.5, 3.5, 'Worker Node 3'),
    ]
    
    for wx, wy, wname in worker_nodes:
        # Worker node box
        worker_box = FancyBboxPatch(
            (wx - 1.4, wy - 2.5), 2.8, 4.5,
            boxstyle="round,pad=0.1",
            edgecolor=COLOR_WORKER,
            facecolor=COLOR_WORKER,
            alpha=0.15,
            linewidth=2
        )
        ax.add_patch(worker_box)
        
        ax.text(wx, wy + 1.8, wname, 
                ha='center', va='top', fontsize=11, fontweight='bold', color=COLOR_WORKER)
        
        # Kubelet
        kubelet_box = FancyBboxPatch(
            (wx - 0.6, wy + 1), 1.2, 0.4,
            boxstyle="round,pad=0.05",
            edgecolor=COLOR_WORKER,
            facecolor='white',
            linewidth=1.5
        )
        ax.add_patch(kubelet_box)
        ax.text(wx, wy + 1.2, 'Kubelet', ha='center', va='center', 
                fontsize=9, fontweight='bold', color=COLOR_WORKER)
        ax.text(wx, wy + 0.7, 'Manages pods\non this node', ha='center', va='top', 
                fontsize=7, color=COLOR_TEXT, style='italic')
        
        # Kube-proxy
        proxy_box = FancyBboxPatch(
            (wx - 0.6, wy + 0.1), 1.2, 0.4,
            boxstyle="round,pad=0.05",
            edgecolor=COLOR_WORKER,
            facecolor='white',
            linewidth=1.5
        )
        ax.add_patch(proxy_box)
        ax.text(wx, wy + 0.3, 'Kube-proxy', ha='center', va='center', 
                fontsize=9, fontweight='bold', color=COLOR_WORKER)
        ax.text(wx, wy - 0.15, 'Network rules\nfor services', ha='center', va='top', 
                fontsize=7, color=COLOR_TEXT, style='italic')
        
        # Pods (example)
        pod_positions = [
            (wx - 0.7, wy - 1.1, 'Pod'),
            (wx + 0.3, wy - 1.1, 'Pod'),
            (wx - 0.2, wy - 1.9, 'Pod'),
        ]
        
        for px, py, pname in pod_positions:
            pod_box = FancyBboxPatch(
                (px - 0.2, py - 0.15), 0.4, 0.3,
                boxstyle="round,pad=0.03",
                edgecolor=COLOR_POD,
                facecolor=COLOR_POD,
                alpha=0.6,
                linewidth=1
            )
            ax.add_patch(pod_box)
            ax.text(px, py, pname, ha='center', va='center', 
                    fontsize=7, color='white', fontweight='bold')
    
    # ========== CONNECTIONS ==========
    # Control plane to worker nodes
    for wx, wy, _ in worker_nodes:
        arrow = FancyArrowPatch(
            (3, 6), (wx, wy + 2),
            arrowstyle='->', mutation_scale=20, 
            linewidth=1.5, color=COLOR_CONTROL, alpha=0.5,
            linestyle='--'
        )
        ax.add_patch(arrow)
    
    # ========== LEGEND ==========
    legend_y = 0.8
    ax.text(0.7, legend_y + 0.3, 'Key Concepts:', fontsize=10, 
            fontweight='bold', color=COLOR_TEXT)
    
    legend_items = [
        (COLOR_CONTROL, 'Control Plane: Brain of the cluster (scheduling, state management)'),
        (COLOR_WORKER, 'Worker Nodes: Run application containers (pods)'),
        (COLOR_POD, 'Pods: Smallest deployable units (1+ containers)'),
    ]
    
    for i, (color, text) in enumerate(legend_items):
        y = legend_y - i * 0.3
        legend_box = patches.Rectangle((0.5, y - 0.1), 0.15, 0.15, 
                                       facecolor=color, edgecolor=color, alpha=0.6)
        ax.add_patch(legend_box)
        ax.text(0.75, y, text, fontsize=8, va='center', color=COLOR_TEXT)
    
    # ========== ANNOTATIONS ==========
    ax.text(7, 5.5, '• API Server is the central hub — all components communicate through it', 
            ha='left', va='top', fontsize=9, color=COLOR_TEXT, style='italic')
    ax.text(7, 5.1, '• etcd stores cluster state (which pods run on which nodes)', 
            ha='left', va='top', fontsize=9, color=COLOR_TEXT, style='italic')
    ax.text(7, 4.7, '• Scheduler watches for unassigned pods and picks nodes', 
            ha='left', va='top', fontsize=9, color=COLOR_TEXT, style='italic')
    ax.text(7, 4.3, '• Kubelet ensures containers are running in pods', 
            ha='left', va='top', fontsize=9, color=COLOR_TEXT, style='italic')
    ax.text(7, 3.9, '• If a pod crashes, Controller Manager restarts it', 
            ha='left', va='top', fontsize=9, color=COLOR_TEXT, style='italic')
    
    plt.tight_layout()
    return fig

if __name__ == '__main__':
    import os
    print("Generating Kubernetes architecture diagram...")
    fig = create_architecture_diagram()
    
    # Get script directory and construct path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, '../img/ch03-k8s-architecture.png')
    output_path = os.path.normpath(output_path)
    
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path}")
    
    plt.close()
