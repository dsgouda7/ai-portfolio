#!/usr/bin/env python3
"""
Generate Chapter Animation — Networking & Load Balancing Needle

Shows load distribution across backends with one backend failing
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import matplotlib
matplotlib.use('Agg')
from PIL import Image
import io

def generate_frame(stage):
    """Generate a single frame of the load balancing animation
    
    Stage 0-5: Normal operation (load distributed evenly)
    Stage 6-10: Backend2 fails (load shifts to backend1 & backend3)
    Stage 11-15: Backend2 recovers (load redistributes)
    """
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(6, 7.5, 'Load Balancer — Backend Health', ha='center', va='top',
            fontsize=20, fontweight='bold', color='#ffffff')
    
    # Client
    client_box = FancyBboxPatch((0.5, 3.5), 1.5, 1,
                               boxstyle="round,pad=0.1", 
                               facecolor='#3b82f6', edgecolor='#60a5fa', linewidth=3)
    ax.add_patch(client_box)
    ax.text(1.25, 4, 'Client', ha='center', va='center',
            fontsize=12, fontweight='bold', color='#ffffff')
    
    # Nginx proxy
    proxy_box = FancyBboxPatch((4, 3.5), 2, 1,
                              boxstyle="round,pad=0.1",
                              facecolor='#059669', edgecolor='#10b981', linewidth=3)
    ax.add_patch(proxy_box)
    ax.text(5, 4, 'Nginx\nProxy', ha='center', va='center',
            fontsize=12, fontweight='bold', color='#ffffff')
    
    # Arrow: Client → Nginx
    ax.annotate('', xy=(4, 4), xytext=(2, 4),
                arrowprops=dict(arrowstyle='->', lw=2, color='#ffffff'))
    ax.text(3, 4.3, 'HTTP', ha='center', fontsize=9, color='#ffffff')
    
    # Backend positions
    backends = [
        {"name": "Backend1", "y": 6, "color": "#15803d"},
        {"name": "Backend2", "y": 4, "color": "#15803d"},
        {"name": "Backend3", "y": 2, "color": "#15803d"}
    ]
    
    # Determine backend health based on stage
    if 6 <= stage <= 10:
        backends[1]["color"] = "#b91c1c"  # Backend2 failed (red)
        backends[1]["status"] = "DOWN"
    else:
        backends[1]["status"] = "UP"
    
    backends[0]["status"] = "UP"
    backends[2]["status"] = "UP"
    
    # Draw backends
    for i, backend in enumerate(backends):
        # Backend box
        box = FancyBboxPatch((8.5, backend["y"] - 0.5), 2.5, 1,
                            boxstyle="round,pad=0.1",
                            facecolor=backend["color"], 
                            edgecolor=backend["color"] if backend["status"] == "UP" else "#ef4444",
                            linewidth=3)
        ax.add_patch(box)
        
        # Backend label
        status_text = f"{backend['name']}\n{backend['status']}"
        ax.text(9.75, backend["y"], status_text, ha='center', va='center',
                fontsize=11, fontweight='bold', color='#ffffff')
        
        # Health indicator
        health_color = '#10b981' if backend["status"] == "UP" else '#ef4444'
        health_circle = Circle((11.2, backend["y"]), 0.15, 
                              facecolor=health_color, edgecolor='#ffffff', linewidth=2)
        ax.add_patch(health_circle)
        
        # Arrow: Nginx → Backend
        if backend["status"] == "UP":
            arrow_color = '#10b981'
            arrow_width = 2
        else:
            arrow_color = '#4b5563'
            arrow_width = 1
        
        ax.annotate('', xy=(8.5, backend["y"]), xytext=(6, 4),
                   arrowprops=dict(arrowstyle='->', lw=arrow_width, 
                                 color=arrow_color, alpha=0.7))
    
    # Request distribution bars (bottom panel)
    bar_y_base = 0.5
    bar_width = 0.4
    
    # Calculate request counts based on stage
    if stage < 6:
        # Normal: even distribution (33% each)
        req_counts = [10 + stage * 2, 10 + stage * 2, 10 + stage * 2]
    elif stage < 11:
        # Backend2 down: traffic shifts to backend1 & backend3
        base = 10 + 5 * 2
        shift = (stage - 6) * 3
        req_counts = [base + shift, base, base + shift]  # Backend2 stays at last count before failure
    else:
        # Backend2 recovers: redistribute
        base = 10 + 5 * 2
        shift = 15
        recovery = (stage - 11) * 2
        req_counts = [base + shift - recovery, base + recovery, base + shift - recovery]
    
    ax.text(1, 1.5, 'Request Count:', ha='left', va='center',
            fontsize=11, fontweight='bold', color='#ffffff')
    
    for i, (backend, count) in enumerate(zip(backends, req_counts)):
        x_pos = 3 + i * 3
        
        # Bar
        bar_color = backend["color"] if backend["status"] == "UP" else "#4b5563"
        bar = Rectangle((x_pos, bar_y_base), bar_width, count * 0.02,
                       facecolor=bar_color, edgecolor='#ffffff', linewidth=2)
        ax.add_patch(bar)
        
        # Label
        ax.text(x_pos + bar_width / 2, bar_y_base - 0.2, 
               backend["name"][-1],  # Show just the number
               ha='center', va='top', fontsize=10, color='#ffffff')
        
        # Count
        ax.text(x_pos + bar_width / 2, bar_y_base + count * 0.02 + 0.1,
               str(count), ha='center', va='bottom', 
               fontsize=10, fontweight='bold', color='#ffffff')
    
    # Stage indicator
    if stage < 6:
        stage_text = "Normal Operation"
        stage_color = '#10b981'
    elif stage < 11:
        stage_text = "Backend2 Failed — Failover Active"
        stage_color = '#ef4444'
    else:
        stage_text = "Backend2 Recovered — Load Redistributing"
        stage_color = '#f59e0b'
    
    stage_box = FancyBboxPatch((1, 6.5), 4, 0.6,
                              boxstyle="round,pad=0.05",
                              facecolor=stage_color, edgecolor='#ffffff', linewidth=2, alpha=0.3)
    ax.add_patch(stage_box)
    ax.text(3, 6.8, stage_text, ha='center', va='center',
            fontsize=11, fontweight='bold', color='#ffffff')
    
    return fig

def fig_to_image(fig):
    """Convert matplotlib figure to PIL Image"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#1a1a2e')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img

def main():
    """Generate animation frames and save as GIF"""
    print("Generating load balancing animation...")
    
    frames = []
    
    # Generate frames
    stages = list(range(0, 6)) + [5] * 3 + \
             list(range(6, 11)) + [10] * 3 + \
             list(range(11, 16)) + [15] * 3
    
    for i, stage in enumerate(stages):
        print(f"  Frame {i+1}/{len(stages)}: Stage {stage}", end="\r")
        fig = generate_frame(stage)
        img = fig_to_image(fig)
        frames.append(img)
    
    print("\nSaving animation...")
    
    # Save animated GIF
    frames[0].save(
        '../img/ch07-networking-load-balancing-needle.gif',
        save_all=True,
        append_images=frames[1:],
        duration=200,  # 200ms per frame
        loop=0
    )
    
    # Save static PNG (final frame)
    frames[-1].save('../img/ch07-networking-load-balancing-needle.png')
    
    print("✓ Animation saved:")
    print("  - ch07-networking-load-balancing-needle.gif")
    print("  - ch07-networking-load-balancing-needle.png")

if __name__ == "__main__":
    main()
