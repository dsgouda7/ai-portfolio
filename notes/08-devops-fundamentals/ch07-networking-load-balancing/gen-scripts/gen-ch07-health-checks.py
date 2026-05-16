#!/usr/bin/env python3
"""
Generate Diagram: Health Checks
Shows: Active probing vs passive failure detection
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import matplotlib
matplotlib.use('Agg')

def draw_passive_health_check(ax):
    """Draw passive health check flow"""
    y_base = 6
    
    # Title
    ax.text(7, y_base + 3.5, 'Passive Health Check (Default Nginx)', ha='center', va='center',
            fontsize=14, fontweight='bold', color='#1f2937',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#dbeafe', 
                     edgecolor='#3b82f6', linewidth=2))
    
    # Timeline
    timeline_events = [
        {"time": "t=0s", "x": 1, "event": "Request 1\n✓ Success", "color": '#10b981'},
        {"time": "t=5s", "x": 3.5, "event": "Request 2\n✗ Timeout", "color": '#f59e0b'},
        {"time": "t=10s", "x": 6, "event": "Request 3\n✗ Timeout", "color": '#ef4444'},
        {"time": "t=15s", "x": 8.5, "event": "Backend\nMarked DOWN", "color": '#7f1d1d'},
        {"time": "t=20s", "x": 11, "event": "Traffic\nRerouted", "color": '#3b82f6'},
    ]
    
    # Draw timeline
    timeline_y = y_base + 1.5
    ax.plot([0.5, 12], [timeline_y, timeline_y], 'k-', linewidth=2)
    
    for event in timeline_events:
        # Event marker
        circle = Circle((event["x"], timeline_y), 0.2,
                       facecolor=event["color"], edgecolor='#1f2937', linewidth=2)
        ax.add_patch(circle)
        
        # Time label
        ax.text(event["x"], timeline_y - 0.5, event["time"],
               ha='center', va='top', fontsize=9, color='#374151')
        
        # Event label
        ax.text(event["x"], timeline_y + 0.5, event["event"],
               ha='center', va='bottom', fontsize=9, color=event["color"],
               fontweight='bold')
    
    # Configuration box
    config = "max_fails=2 fail_timeout=30s\n\nDetection: 10s (2 failures @ 5s interval)"
    config_box = FancyBboxPatch((0.5, y_base - 0.5), 4, 1.2,
                               boxstyle="round,pad=0.1",
                               facecolor='#fef3c7', edgecolor='#f59e0b', linewidth=2)
    ax.add_patch(config_box)
    ax.text(2.5, y_base + 0.1, config, ha='center', va='center',
           fontsize=9, color='#78350f')
    
    # Pros/Cons
    pros_cons = [
        ("✓ Pros", "• No extra traffic\n• Zero overhead", '#10b981'),
        ("✗ Cons", "• Slow detection\n• Real requests fail", '#ef4444')
    ]
    
    for i, (title, text, color) in enumerate(pros_cons):
        x_pos = 7 + i * 4
        box = FancyBboxPatch((x_pos - 1.5, y_base - 0.5), 3, 1.2,
                            boxstyle="round,pad=0.1",
                            facecolor='#f3f4f6', edgecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x_pos, y_base + 0.4, title, ha='center', va='top',
               fontsize=10, fontweight='bold', color=color)
        ax.text(x_pos, y_base - 0.1, text, ha='center', va='center',
               fontsize=8, color='#374151')

def draw_active_health_check(ax):
    """Draw active health check flow"""
    y_base = 2
    
    # Title
    ax.text(7, y_base + 3.5, 'Active Health Check (Nginx Plus / Module)', ha='center', va='center',
            fontsize=14, fontweight='bold', color='#1f2937',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#d1fae5', 
                     edgecolor='#059669', linewidth=2))
    
    # Timeline
    timeline_events = [
        {"time": "t=0s", "x": 1, "event": "Probe /health\n✓ 200 OK", "color": '#10b981'},
        {"time": "t=5s", "x": 3.5, "event": "Probe /health\n✗ Timeout", "color": '#f59e0b'},
        {"time": "t=10s", "x": 6, "event": "Probe /health\n✗ Timeout", "color": '#ef4444'},
        {"time": "t=15s", "x": 8.5, "event": "Backend\nMarked DOWN", "color": '#7f1d1d'},
        {"time": "t=20s", "x": 11, "event": "Traffic\nRerouted", "color": '#3b82f6'},
    ]
    
    # Draw timeline
    timeline_y = y_base + 1.5
    ax.plot([0.5, 12], [timeline_y, timeline_y], 'k-', linewidth=2)
    
    for event in timeline_events:
        # Event marker
        circle = Circle((event["x"], timeline_y), 0.2,
                       facecolor=event["color"], edgecolor='#1f2937', linewidth=2)
        ax.add_patch(circle)
        
        # Time label
        ax.text(event["x"], timeline_y - 0.5, event["time"],
               ha='center', va='top', fontsize=9, color='#374151')
        
        # Event label
        ax.text(event["x"], timeline_y + 0.5, event["event"],
               ha='center', va='bottom', fontsize=9, color=event["color"],
               fontweight='bold')
    
    # Configuration box
    config = "interval=5s fails=2 uri=/health\n\nDetection: 10s (2 probe failures)"
    config_box = FancyBboxPatch((0.5, y_base - 0.5), 4, 1.2,
                               boxstyle="round,pad=0.1",
                               facecolor='#fef3c7', edgecolor='#f59e0b', linewidth=2)
    ax.add_patch(config_box)
    ax.text(2.5, y_base + 0.1, config, ha='center', va='center',
           fontsize=9, color='#78350f')
    
    # Pros/Cons
    pros_cons = [
        ("✓ Pros", "• Fast detection (5-10s)\n• No user impact", '#10b981'),
        ("✗ Cons", "• Extra probe traffic\n• Requires Nginx Plus", '#ef4444')
    ]
    
    for i, (title, text, color) in enumerate(pros_cons):
        x_pos = 7 + i * 4
        box = FancyBboxPatch((x_pos - 1.5, y_base - 0.5), 3, 1.2,
                            boxstyle="round,pad=0.1",
                            facecolor='#f3f4f6', edgecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x_pos, y_base + 0.4, title, ha='center', va='top',
               fontsize=10, fontweight='bold', color=color)
        ax.text(x_pos, y_base - 0.1, text, ha='center', va='center',
               fontsize=8, color='#374151')

def main():
    """Generate health checks diagram"""
    print("Generating health checks diagram...")
    
    fig, ax = plt.subplots(figsize=(14, 12), facecolor='#ffffff')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Main title
    ax.text(7, 11.5, 'Health Check Strategies', ha='center', va='top',
            fontsize=20, fontweight='bold', color='#1f2937')
    
    # Draw both strategies
    draw_passive_health_check(ax)
    draw_active_health_check(ax)
    
    # Comparison table at bottom
    table_y = 0.8
    ax.text(7, table_y, 'Recommendation: Use active health checks in production (faster failure detection)', 
            ha='center', va='center',
            fontsize=10, color='#065f46', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#d1fae5', 
                     edgecolor='#059669', linewidth=2))
    
    # Health check endpoint best practices
    practices = [
        "Health Check Endpoint Best Practices:",
        "• Return 200 OK only when service is ready (DB connected, dependencies available)",
        "• Keep endpoint lightweight (<100ms response time)",
        "• Don't just return static 'OK' — verify critical dependencies",
        "• Log health check failures for debugging"
    ]
    
    practices_y = 0.2
    for i, practice in enumerate(practices):
        font_weight = 'bold' if i == 0 else 'normal'
        ax.text(7, practices_y - i * 0.15, practice, ha='center', va='center',
               fontsize=8, color='#374151', fontweight=font_weight)
    
    # Save
    plt.tight_layout()
    plt.savefig('../img/gen_ch07_health_checks.png', dpi=150, bbox_inches='tight', facecolor='#ffffff')
    plt.close()
    
    print("✓ Diagram saved: gen_ch07_health_checks.png")

if __name__ == "__main__":
    main()
