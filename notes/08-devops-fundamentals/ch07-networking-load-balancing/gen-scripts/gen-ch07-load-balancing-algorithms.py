#!/usr/bin/env python3
"""
Generate Diagram: Load Balancing Algorithms
Shows: Round-robin, least-conn, IP hash
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib
matplotlib.use('Agg')

def draw_algorithm(ax, title, y_base, algorithm_type):
    """Draw one load balancing algorithm"""
    
    # Title
    ax.text(7, y_base + 2.8, title, ha='center', va='center',
            fontsize=14, fontweight='bold', color='#1f2937',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f3f4f6', 
                     edgecolor='#6b7280', linewidth=2))
    
    # Nginx proxy
    proxy_box = FancyBboxPatch((5.5, y_base + 1.5), 3, 1,
                              boxstyle="round,pad=0.1",
                              facecolor='#d1fae5', edgecolor='#059669', linewidth=3)
    ax.add_patch(proxy_box)
    ax.text(7, y_base + 2, 'Nginx', ha='center', va='center',
            fontsize=12, fontweight='bold', color='#065f46')
    
    # Backend servers
    backends = [
        {"name": "B1", "x": 2, "load": 5},
        {"name": "B2", "x": 7, "load": 12},
        {"name": "B3", "x": 12, "load": 8}
    ]
    
    for backend in backends:
        # Backend box
        box_color = '#e0e7ff' if backend["load"] < 10 else '#fed7aa'
        box = FancyBboxPatch((backend["x"] - 0.8, y_base - 0.4), 1.6, 0.8,
                            boxstyle="round,pad=0.05",
                            facecolor=box_color, edgecolor='#4f46e5', linewidth=2)
        ax.add_patch(box)
        ax.text(backend["x"], y_base, backend["name"], ha='center', va='center',
               fontsize=11, fontweight='bold', color='#312e81')
    
    # Draw request routing based on algorithm
    if algorithm_type == "round-robin":
        # Sequential distribution
        colors = ['#3b82f6', '#10b981', '#f59e0b']
        for i, (backend, color) in enumerate(zip(backends, colors)):
            arrow = FancyArrowPatch((7, y_base + 1.5), (backend["x"], y_base + 0.4),
                                   arrowstyle='->', mutation_scale=20,
                                   lw=2, color=color)
            ax.add_patch(arrow)
            # Request labels
            ax.text(backend["x"], y_base - 0.8, f'Req {i+1}, {i+4}, {i+7}...',
                   ha='center', va='center', fontsize=8, color='#374151')
        
        # Description
        desc = "Cycles through backends in order\nEven distribution (33% each)"
        ax.text(7, y_base - 1.5, desc, ha='center', va='center',
               fontsize=9, color='#4b5563', style='italic')
    
    elif algorithm_type == "least-conn":
        # Route to backend with fewest connections
        # B1: 5 connections (chosen)
        # B2: 12 connections (busy)
        # B3: 8 connections
        
        # Show connection counts
        for backend in backends:
            ax.text(backend["x"], y_base + 0.8, f'{backend["load"]} active',
                   ha='center', va='center', fontsize=8, 
                   color='#dc2626' if backend["load"] > 10 else '#059669')
        
        # Arrow to backend with fewest connections (B1)
        arrow = FancyArrowPatch((7, y_base + 1.5), (backends[0]["x"], y_base + 0.4),
                               arrowstyle='->', mutation_scale=25,
                               lw=3, color='#10b981')
        ax.add_patch(arrow)
        ax.text(4, y_base + 1, 'New request', ha='center', fontsize=9, 
               color='#065f46', fontweight='bold')
        
        # Description
        desc = "Routes to backend with fewest active connections\nIdeal for long-lived connections (WebSocket, SSE)"
        ax.text(7, y_base - 1.5, desc, ha='center', va='center',
               fontsize=9, color='#4b5563', style='italic')
    
    elif algorithm_type == "ip-hash":
        # Same client IP → same backend
        clients = [
            {"ip": "192.168.1.50", "backend": 0, "color": '#3b82f6'},
            {"ip": "192.168.1.75", "backend": 2, "color": '#f59e0b'},
            {"ip": "192.168.1.100", "backend": 1, "color": '#10b981'}
        ]
        
        # Client indicators
        for i, client in enumerate(clients):
            client_x = 0.5 + i * 1.5
            # Client box
            circle = Circle((client_x, y_base + 2), 0.3, 
                          facecolor=client["color"], edgecolor='#1f2937', linewidth=2)
            ax.add_patch(circle)
            ax.text(client_x, y_base + 2.7, client["ip"], ha='center', fontsize=7,
                   color='#374151')
            
            # Arrow: Client → Nginx → Backend
            arrow1 = FancyArrowPatch((client_x + 0.3, y_base + 2), (5.5, y_base + 2),
                                    arrowstyle='->', mutation_scale=15,
                                    lw=1.5, color=client["color"], linestyle='--')
            ax.add_patch(arrow1)
            
            target_backend = backends[client["backend"]]
            arrow2 = FancyArrowPatch((7, y_base + 1.5), 
                                    (target_backend["x"], y_base + 0.4),
                                    arrowstyle='->', mutation_scale=20,
                                    lw=2, color=client["color"])
            ax.add_patch(arrow2)
        
        # Description
        desc = "hash(client_IP) % N → backend index\nSticky sessions (same client → same backend)"
        ax.text(7, y_base - 1.5, desc, ha='center', va='center',
               fontsize=9, color='#4b5563', style='italic')

def main():
    """Generate load balancing algorithms diagram"""
    print("Generating load balancing algorithms diagram...")
    
    fig, ax = plt.subplots(figsize=(14, 12), facecolor='#ffffff')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Main title
    ax.text(7, 11.5, 'Load Balancing Algorithms', ha='center', va='top',
            fontsize=20, fontweight='bold', color='#1f2937')
    
    # Draw three algorithms
    draw_algorithm(ax, "1. Round-Robin (Default)", 8, "round-robin")
    draw_algorithm(ax, "2. Least Connections", 5, "least-conn")
    draw_algorithm(ax, "3. IP Hash (Sticky Sessions)", 2, "ip-hash")
    
    # Comparison table
    table_data = [
        ["Algorithm", "Use Case", "Pros", "Cons"],
        ["Round-robin", "Identical backends", "Simple, even", "Ignores load"],
        ["Least-conn", "Long connections", "Respects load", "Overhead"],
        ["IP-hash", "Stateful apps", "Sticky sessions", "Uneven distribution"]
    ]
    
    # Note at bottom
    note = "Note: Choose algorithm based on application requirements. Round-robin works for most stateless apps."
    ax.text(7, 0.3, note, ha='center', va='center',
            fontsize=9, color='#6b7280', style='italic',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#fef3c7', 
                     edgecolor='#f59e0b', linewidth=1))
    
    # Save
    plt.tight_layout()
    plt.savefig('../img/gen_ch07_load_balancing_algorithms.png', dpi=150, bbox_inches='tight', facecolor='#ffffff')
    plt.close()
    
    print("✓ Diagram saved: gen_ch07_load_balancing_algorithms.png")

if __name__ == "__main__":
    main()
