#!/usr/bin/env python3
"""
Generate Diagram: Reverse Proxy Architecture
Shows: Client → Nginx → Backend pool
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib
matplotlib.use('Agg')

def main():
    """Generate reverse proxy architecture diagram"""
    print("Generating reverse proxy diagram...")
    
    fig, ax = plt.subplots(figsize=(14, 10), facecolor='#ffffff')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, 'Reverse Proxy Architecture', ha='center', va='top',
            fontsize=22, fontweight='bold', color='#1f2937')
    
    # Internet cloud
    cloud_text = "Internet\n(Clients)"
    cloud_box = FancyBboxPatch((0.5, 7), 2.5, 1.5,
                              boxstyle="round,pad=0.15",
                              facecolor='#dbeafe', edgecolor='#3b82f6', linewidth=3)
    ax.add_patch(cloud_box)
    ax.text(1.75, 7.75, cloud_text, ha='center', va='center',
            fontsize=13, fontweight='bold', color='#1e40af')
    
    # Firewall
    firewall_box = FancyBboxPatch((4, 7.25), 1.5, 1,
                                 boxstyle="round,pad=0.1",
                                 facecolor='#fef3c7', edgecolor='#f59e0b', linewidth=3)
    ax.add_patch(firewall_box)
    ax.text(4.75, 7.75, 'Firewall\nPort 80/443', ha='center', va='center',
            fontsize=11, fontweight='bold', color='#92400e')
    
    # Arrow: Internet → Firewall
    arrow1 = FancyArrowPatch((3, 7.75), (4, 7.75),
                            arrowstyle='->', mutation_scale=30,
                            lw=3, color='#3b82f6')
    ax.add_patch(arrow1)
    ax.text(3.5, 8.1, 'HTTPS', ha='center', fontsize=10, color='#1e40af')
    
    # Nginx reverse proxy
    nginx_box = FancyBboxPatch((7, 6.75), 3, 1.5,
                              boxstyle="round,pad=0.15",
                              facecolor='#d1fae5', edgecolor='#059669', linewidth=4)
    ax.add_patch(nginx_box)
    ax.text(8.5, 7.75, 'Nginx Reverse Proxy', ha='center', va='top',
            fontsize=14, fontweight='bold', color='#065f46')
    ax.text(8.5, 7.3, '• SSL Termination\n• Load Balancing\n• Health Checks', 
            ha='center', va='center', fontsize=9, color='#065f46')
    
    # Arrow: Firewall → Nginx
    arrow2 = FancyArrowPatch((5.5, 7.75), (7, 7.5),
                            arrowstyle='->', mutation_scale=30,
                            lw=3, color='#3b82f6')
    ax.add_patch(arrow2)
    ax.text(6.25, 8, 'Public IP', ha='center', fontsize=10, color='#1e40af')
    
    # Backend pool label
    ax.text(8.5, 5.5, 'Backend Pool (Private Network)', ha='center', va='center',
            fontsize=13, fontweight='bold', color='#1f2937',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f3f4f6', edgecolor='#9ca3af', linewidth=2))
    
    # Backend servers
    backends = [
        {"name": "Backend1", "x": 3, "y": 3, "ip": "10.0.1.4:5000"},
        {"name": "Backend2", "x": 8.5, "y": 3, "ip": "10.0.1.5:5000"},
        {"name": "Backend3", "x": 14, "y": 3, "ip": "10.0.1.6:5000"}
    ]
    
    for i, backend in enumerate(backends):
        # Backend box
        box = FancyBboxPatch((backend["x"] - 1.75, backend["y"] - 0.75), 3.5, 1.5,
                            boxstyle="round,pad=0.1",
                            facecolor='#e0e7ff', edgecolor='#4f46e5', linewidth=3)
        ax.add_patch(box)
        
        # Backend label
        ax.text(backend["x"], backend["y"] + 0.3, backend["name"], 
               ha='center', va='center',
               fontsize=12, fontweight='bold', color='#312e81')
        ax.text(backend["x"], backend["y"] - 0.2, backend["ip"],
               ha='center', va='center',
               fontsize=9, color='#4338ca')
        
        # Health indicator
        health_circle = plt.Circle((backend["x"] + 1.4, backend["y"] + 0.5), 0.15,
                                  facecolor='#10b981', edgecolor='#065f46', linewidth=2)
        ax.add_patch(health_circle)
        ax.text(backend["x"] + 1.4, backend["y"] + 0.5, '✓', 
               ha='center', va='center',
               fontsize=10, fontweight='bold', color='#ffffff')
        
        # Arrow: Nginx → Backend
        arrow_start = (8.5, 6.75)
        arrow_end = (backend["x"], backend["y"] + 0.75)
        arrow = FancyArrowPatch(arrow_start, arrow_end,
                               arrowstyle='->', mutation_scale=25,
                               lw=2, color='#059669', linestyle='--')
        ax.add_patch(arrow)
    
    # Arrow label: Round-robin
    ax.text(8.5, 5, 'Round-robin\nload balancing', ha='center', va='center',
            fontsize=10, color='#065f46', style='italic')
    
    # Database (shared by all backends)
    db_box = FancyBboxPatch((5.75, 0.5), 2.5, 1,
                           boxstyle="round,pad=0.1",
                           facecolor='#fce7f3', edgecolor='#ec4899', linewidth=3)
    ax.add_patch(db_box)
    ax.text(7, 1, 'Database\nPostgreSQL', ha='center', va='center',
            fontsize=11, fontweight='bold', color='#9f1239')
    
    # Arrows: Backends → Database
    for backend in backends:
        arrow_start = (backend["x"], backend["y"] - 0.75)
        arrow_end = (7, 1.5)
        arrow = FancyArrowPatch(arrow_start, arrow_end,
                               arrowstyle='<->', mutation_scale=20,
                               lw=1.5, color='#ec4899', linestyle=':')
        ax.add_patch(arrow)
    
    # Key benefits box
    benefits = [
        "✓ Single point of entry (clients only know proxy IP)",
        "✓ SSL termination (decrypt once, not per backend)",
        "✓ Horizontal scaling (add/remove backends dynamically)",
        "✓ Failover (route around unhealthy backends)",
        "✓ Caching & rate limiting (protect backends)"
    ]
    
    benefit_y = 9
    for i, benefit in enumerate(benefits):
        ax.text(11.5, benefit_y - i * 0.4, benefit,
               ha='left', va='center', fontsize=9, color='#374151')
    
    # Save
    plt.tight_layout()
    plt.savefig('../img/gen_ch07_reverse_proxy.png', dpi=150, bbox_inches='tight', facecolor='#ffffff')
    plt.close()
    
    print("✓ Diagram saved: gen_ch07_reverse_proxy.png")

if __name__ == "__main__":
    main()
