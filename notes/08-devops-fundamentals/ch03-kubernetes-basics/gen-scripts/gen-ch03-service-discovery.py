"""
Generate Service Discovery diagram comparing ClusterIP, NodePort, and LoadBalancer.

Usage:
    python gen_ch03_service_discovery.py

Output:
    ../img/ch03-service-discovery.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Color palette
COLOR_CLUSTERIP = '#4A90E2'    # Blue
COLOR_NODEPORT = '#7ED321'     # Green
COLOR_LOADBALANCER = '#F5A623' # Orange
COLOR_POD = '#9013FE'          # Purple
COLOR_TEXT = '#2C3E50'         # Dark text
COLOR_CLIENT = '#E74C3C'       # Red

def create_service_discovery_diagram():
    """Create diagram showing three service types."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    fig.patch.set_facecolor('white')
    
    # Title
    ax.text(8, 9.7, 'Kubernetes Service Types', 
            ha='center', va='top', fontsize=18, fontweight='bold', color=COLOR_TEXT)
    ax.text(8, 9.3, 'How to expose pods to clients', 
            ha='center', va='top', fontsize=12, color=COLOR_TEXT, style='italic')
    
    # ========== CLUSTERIP (Default) ==========
    clusterip_x = 2.5
    
    # Section box
    section_box = FancyBboxPatch(
        (0.3, 4.5), 4.5, 4,
        boxstyle="round,pad=0.1",
        edgecolor=COLOR_CLUSTERIP,
        facecolor=COLOR_CLUSTERIP,
        alpha=0.1,
        linewidth=2
    )
    ax.add_patch(section_box)
    
    ax.text(clusterip_x, 8.3, 'ClusterIP (Default)', 
            ha='center', va='top', fontsize=13, fontweight='bold', color=COLOR_CLUSTERIP)
    ax.text(clusterip_x, 7.9, 'Internal only — not accessible outside cluster', 
            ha='center', va='top', fontsize=9, color=COLOR_TEXT, style='italic')
    
    # Service
    service_box = FancyBboxPatch(
        (clusterip_x - 0.8, 6.8), 1.6, 0.5,
        boxstyle="round,pad=0.05",
        edgecolor=COLOR_CLUSTERIP,
        facecolor=COLOR_CLUSTERIP,
        alpha=0.6,
        linewidth=2
    )
    ax.add_patch(service_box)
    ax.text(clusterip_x, 7.05, 'Service\n(ClusterIP: 10.96.0.1)', 
            ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    # Pods
    pod_positions = [(1.5, 5.8), (2.5, 5.8), (3.5, 5.8)]
    for px, py in pod_positions:
        pod_box = FancyBboxPatch(
            (px - 0.3, py - 0.25), 0.6, 0.5,
            boxstyle="round,pad=0.05",
            edgecolor=COLOR_POD,
            facecolor=COLOR_POD,
            alpha=0.6,
            linewidth=1.5
        )
        ax.add_patch(pod_box)
        ax.text(px, py, 'Pod', ha='center', va='center', 
                fontsize=8, color='white', fontweight='bold')
    
    # Arrows: Service → Pods
    for px, py in pod_positions:
        arrow = FancyArrowPatch(
            (clusterip_x, 6.8), (px, py + 0.25),
            arrowstyle='->', mutation_scale=15,
            linewidth=1.5, color=COLOR_CLUSTERIP, alpha=0.7
        )
        ax.add_patch(arrow)
    
    # Client (inside cluster)
    client_box = FancyBboxPatch(
        (clusterip_x - 0.5, 7.8), 1, 0.35,
        boxstyle="round,pad=0.05",
        edgecolor=COLOR_CLIENT,
        facecolor=COLOR_CLIENT,
        alpha=0.5,
        linewidth=1.5
    )
    ax.add_patch(client_box)
    ax.text(clusterip_x, 7.98, 'Client Pod', ha='center', va='center', 
            fontsize=8, color='white', fontweight='bold')
    
    # Arrow: Client → Service
    arrow = FancyArrowPatch(
        (clusterip_x, 7.8), (clusterip_x, 7.3),
        arrowstyle='->', mutation_scale=15,
        linewidth=2, color=COLOR_CLIENT, alpha=0.7
    )
    ax.add_patch(arrow)
    
    # Use case
    ax.text(clusterip_x, 5.2, 'Use case:', ha='center', va='top', 
            fontsize=9, fontweight='bold', color=COLOR_TEXT)
    ax.text(clusterip_x, 4.9, 'Internal microservices\n(database, cache, API)', 
            ha='center', va='top', fontsize=8, color=COLOR_TEXT, style='italic')
    
    # ========== NODEPORT ==========
    nodeport_x = 8
    
    # Section box
    section_box = FancyBboxPatch(
        (5.8, 4.5), 4.5, 4,
        boxstyle="round,pad=0.1",
        edgecolor=COLOR_NODEPORT,
        facecolor=COLOR_NODEPORT,
        alpha=0.1,
        linewidth=2
    )
    ax.add_patch(section_box)
    
    ax.text(nodeport_x, 8.3, 'NodePort', 
            ha='center', va='top', fontsize=13, fontweight='bold', color=COLOR_NODEPORT)
    ax.text(nodeport_x, 7.9, 'Accessible via node IP:port (30000-32767)', 
            ha='center', va='top', fontsize=9, color=COLOR_TEXT, style='italic')
    
    # Node box
    node_box = FancyBboxPatch(
        (6.5, 5.4), 3, 2,
        boxstyle="round,pad=0.05",
        edgecolor='gray',
        facecolor='lightgray',
        alpha=0.3,
        linewidth=1.5
    )
    ax.add_patch(node_box)
    ax.text(8, 7.3, 'Node (192.168.1.10)', ha='center', va='top', 
            fontsize=8, color='gray', fontweight='bold')
    
    # Service
    service_box = FancyBboxPatch(
        (nodeport_x - 0.8, 6.5), 1.6, 0.5,
        boxstyle="round,pad=0.05",
        edgecolor=COLOR_NODEPORT,
        facecolor=COLOR_NODEPORT,
        alpha=0.6,
        linewidth=2
    )
    ax.add_patch(service_box)
    ax.text(nodeport_x, 6.75, 'Service\n(NodePort: 30080)', 
            ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    # Pods
    pod_positions = [(7, 5.8), (8, 5.8), (9, 5.8)]
    for px, py in pod_positions:
        pod_box = FancyBboxPatch(
            (px - 0.3, py - 0.25), 0.6, 0.5,
            boxstyle="round,pad=0.05",
            edgecolor=COLOR_POD,
            facecolor=COLOR_POD,
            alpha=0.6,
            linewidth=1.5
        )
        ax.add_patch(pod_box)
        ax.text(px, py, 'Pod', ha='center', va='center', 
                fontsize=8, color='white', fontweight='bold')
    
    # Arrows: Service → Pods
    for px, py in pod_positions:
        arrow = FancyArrowPatch(
            (nodeport_x, 6.5), (px, py + 0.25),
            arrowstyle='->', mutation_scale=15,
            linewidth=1.5, color=COLOR_NODEPORT, alpha=0.7
        )
        ax.add_patch(arrow)
    
    # External client
    client_box = FancyBboxPatch(
        (nodeport_x - 0.6, 8), 1.2, 0.35,
        boxstyle="round,pad=0.05",
        edgecolor=COLOR_CLIENT,
        facecolor=COLOR_CLIENT,
        alpha=0.5,
        linewidth=1.5
    )
    ax.add_patch(client_box)
    ax.text(nodeport_x, 8.18, 'External Client', ha='center', va='center', 
            fontsize=8, color='white', fontweight='bold')
    
    # Arrow: Client → Node
    arrow = FancyArrowPatch(
        (nodeport_x, 8), (nodeport_x, 7.4),
        arrowstyle='->', mutation_scale=15,
        linewidth=2, color=COLOR_CLIENT, alpha=0.7
    )
    ax.add_patch(arrow)
    ax.text(nodeport_x + 0.8, 7.7, 'http://192.168.1.10:30080', 
            ha='left', va='center', fontsize=7, color=COLOR_CLIENT, 
            fontweight='bold', family='monospace')
    
    # Use case
    ax.text(nodeport_x, 5.2, 'Use case:', ha='center', va='top', 
            fontsize=9, fontweight='bold', color=COLOR_TEXT)
    ax.text(nodeport_x, 4.9, 'Development/testing\n(Kind, Minikube)', 
            ha='center', va='top', fontsize=8, color=COLOR_TEXT, style='italic')
    
    # ========== LOADBALANCER ==========
    lb_x = 13.5
    
    # Section box
    section_box = FancyBboxPatch(
        (11.3, 4.5), 4.5, 4,
        boxstyle="round,pad=0.1",
        edgecolor=COLOR_LOADBALANCER,
        facecolor=COLOR_LOADBALANCER,
        alpha=0.1,
        linewidth=2
    )
    ax.add_patch(section_box)
    
    ax.text(lb_x, 8.3, 'LoadBalancer', 
            ha='center', va='top', fontsize=13, fontweight='bold', color=COLOR_LOADBALANCER)
    ax.text(lb_x, 7.9, 'Cloud provider assigns external IP', 
            ha='center', va='top', fontsize=9, color=COLOR_TEXT, style='italic')
    
    # Cloud Load Balancer
    lb_box = FancyBboxPatch(
        (lb_x - 1, 7.2), 2, 0.5,
        boxstyle="round,pad=0.05",
        edgecolor=COLOR_LOADBALANCER,
        facecolor=COLOR_LOADBALANCER,
        alpha=0.7,
        linewidth=2
    )
    ax.add_patch(lb_box)
    ax.text(lb_x, 7.45, 'Cloud Load Balancer\n(IP: 203.0.113.42)', 
            ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    # Service
    service_box = FancyBboxPatch(
        (lb_x - 0.8, 6.3), 1.6, 0.5,
        boxstyle="round,pad=0.05",
        edgecolor=COLOR_LOADBALANCER,
        facecolor=COLOR_LOADBALANCER,
        alpha=0.5,
        linewidth=2
    )
    ax.add_patch(service_box)
    ax.text(lb_x, 6.55, 'Service\n(LoadBalancer)', 
            ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    # Pods (across multiple nodes)
    ax.text(12.3, 5.9, 'Node 1', ha='center', va='top', 
            fontsize=7, color='gray', style='italic')
    ax.text(13.5, 5.9, 'Node 2', ha='center', va='top', 
            fontsize=7, color='gray', style='italic')
    ax.text(14.7, 5.9, 'Node 3', ha='center', va='top', 
            fontsize=7, color='gray', style='italic')
    
    pod_positions = [(12.3, 5.3), (13.5, 5.3), (14.7, 5.3)]
    for px, py in pod_positions:
        pod_box = FancyBboxPatch(
            (px - 0.3, py - 0.25), 0.6, 0.5,
            boxstyle="round,pad=0.05",
            edgecolor=COLOR_POD,
            facecolor=COLOR_POD,
            alpha=0.6,
            linewidth=1.5
        )
        ax.add_patch(pod_box)
        ax.text(px, py, 'Pod', ha='center', va='center', 
                fontsize=8, color='white', fontweight='bold')
    
    # Arrows: LB → Service → Pods
    arrow = FancyArrowPatch(
        (lb_x, 7.2), (lb_x, 6.8),
        arrowstyle='->', mutation_scale=15,
        linewidth=2, color=COLOR_LOADBALANCER, alpha=0.7
    )
    ax.add_patch(arrow)
    
    for px, py in pod_positions:
        arrow = FancyArrowPatch(
            (lb_x, 6.3), (px, py + 0.25),
            arrowstyle='->', mutation_scale=15,
            linewidth=1.5, color=COLOR_LOADBALANCER, alpha=0.7
        )
        ax.add_patch(arrow)
    
    # External client
    client_box = FancyBboxPatch(
        (lb_x - 0.6, 8.1), 1.2, 0.35,
        boxstyle="round,pad=0.05",
        edgecolor=COLOR_CLIENT,
        facecolor=COLOR_CLIENT,
        alpha=0.5,
        linewidth=1.5
    )
    ax.add_patch(client_box)
    ax.text(lb_x, 8.28, 'Internet Client', ha='center', va='center', 
            fontsize=8, color='white', fontweight='bold')
    
    # Arrow: Client → LB
    arrow = FancyArrowPatch(
        (lb_x, 8.1), (lb_x, 7.7),
        arrowstyle='->', mutation_scale=15,
        linewidth=2, color=COLOR_CLIENT, alpha=0.7
    )
    ax.add_patch(arrow)
    ax.text(lb_x - 1.8, 7.9, 'http://203.0.113.42', 
            ha='right', va='center', fontsize=7, color=COLOR_CLIENT, 
            fontweight='bold', family='monospace')
    
    # Use case
    ax.text(lb_x, 4.9, 'Use case:', ha='center', va='top', 
            fontsize=9, fontweight='bold', color=COLOR_TEXT)
    ax.text(lb_x, 4.6, 'Production web apps\n(AWS ELB, Azure LB, GCP LB)', 
            ha='center', va='top', fontsize=8, color=COLOR_TEXT, style='italic')
    
    # ========== COMPARISON TABLE ==========
    table_y = 3.5
    ax.text(8, table_y + 0.3, 'Quick Comparison', ha='center', va='top', 
            fontsize=12, fontweight='bold', color=COLOR_TEXT)
    
    # Table headers
    headers = ['Feature', 'ClusterIP', 'NodePort', 'LoadBalancer']
    header_x = [1, 4.5, 8, 11.5]
    
    for i, (hx, header) in enumerate(zip(header_x, headers)):
        ax.text(hx, table_y - 0.2, header, ha='left', va='top', 
                fontsize=9, fontweight='bold', color=COLOR_TEXT)
    
    # Table rows
    rows = [
        ('External access', '❌ No', '✅ Yes (node IP)', '✅ Yes (cloud IP)'),
        ('Use in production', '✅ Internal only', '⚠️ Rare', '✅ Common'),
        ('Cloud provider needed', '❌ No', '❌ No', '✅ Yes (AWS/Azure/GCP)'),
        ('Port range', 'Any', '30000-32767', 'Any (80, 443 typical)'),
        ('Cost', 'Free', 'Free', '💰 Hourly charge'),
    ]
    
    for i, (feature, clip, nport, lb) in enumerate(rows):
        y = table_y - 0.5 - i * 0.3
        ax.text(header_x[0], y, feature, ha='left', va='top', 
                fontsize=8, color=COLOR_TEXT, style='italic')
        ax.text(header_x[1], y, clip, ha='left', va='top', 
                fontsize=8, color=COLOR_TEXT)
        ax.text(header_x[2], y, nport, ha='left', va='top', 
                fontsize=8, color=COLOR_TEXT)
        ax.text(header_x[3], y, lb, ha='left', va='top', 
                fontsize=8, color=COLOR_TEXT)
    
    # ========== BOTTOM NOTE ==========
    note_box = FancyBboxPatch(
        (0.3, 0.1), 15.4, 0.6,
        boxstyle="round,pad=0.1",
        edgecolor='gray',
        facecolor='lightyellow',
        alpha=0.5,
        linewidth=1
    )
    ax.add_patch(note_box)
    
    ax.text(8, 0.45, 
            '💡 All service types include load balancing across pods. '
            'ClusterIP is the default — add NodePort/LoadBalancer only when external access is needed. '
            'LoadBalancer automatically creates NodePort + ClusterIP internally.', 
            ha='center', va='center', fontsize=9, color=COLOR_TEXT, style='italic', wrap=True)
    
    plt.tight_layout()
    return fig

if __name__ == '__main__':
    import os
    print("Generating Service Discovery diagram...")
    fig = create_service_discovery_diagram()
    
    # Get script directory and construct path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, '../img/ch03-service-discovery.png')
    output_path = os.path.normpath(output_path)
    
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path}")
    
    plt.close()
