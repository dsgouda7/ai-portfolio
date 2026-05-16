"""
Generate network isolation diagram showing internal vs external networks.

Illustrates:
- External network (host access)
- Internal app-network (service-to-service DNS)
- Port exposure vs internal-only services

Output: img/ch02_network_isolation.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from pathlib import Path

# Color palette
COLOR_PRIMARY = "#1e3a8a"
COLOR_SUCCESS = "#15803d"
COLOR_INFO = "#1d4ed8"
COLOR_CAUTION = "#b45309"
COLOR_DANGER = "#b91c1c"
COLOR_BG = "#1a1a2e"

def draw_network_zone(ax, x, y, width, height, label, color, alpha=0.2):
    """Draw a network zone with label."""
    zone = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.15",
        facecolor=color,
        edgecolor=color,
        linewidth=3,
        alpha=alpha,
        zorder=0
    )
    ax.add_patch(zone)
    
    # Zone label
    ax.text(x + width/2, y + height - 0.3,
            label,
            ha='center', va='top',
            fontsize=14, fontweight='bold',
            color=color,
            bbox=dict(boxstyle='round,pad=0.4', facecolor=COLOR_BG,
                     edgecolor=color, linewidth=2))

def draw_container(ax, x, y, width, height, name, port_info, color, exposed=False):
    """Draw a container with optional port exposure indicator."""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.08",
        facecolor=color,
        edgecolor='white',
        linewidth=2,
        alpha=0.9
    )
    ax.add_patch(box)
    
    # Container name
    ax.text(x + width/2, y + height/2 + 0.2,
            name,
            ha='center', va='center',
            fontsize=12, fontweight='bold',
            color='white')
    
    # Port info
    ax.text(x + width/2, y + height/2 - 0.2,
            port_info,
            ha='center', va='center',
            fontsize=9,
            color='#cccccc')
    
    # Exposure indicator
    if exposed:
        ax.plot(x + width - 0.2, y + height - 0.2, 'o',
                color='#FFD700', markersize=12,
                markeredgecolor='white', markeredgewidth=2, zorder=10)
        ax.text(x + width - 0.2, y + height - 0.2, '🌐',
                ha='center', va='center', fontsize=8)

def draw_connection(ax, x1, y1, x2, y2, label="", style='-', color='white'):
    """Draw a connection line between points."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->,head_width=0.3,head_length=0.6',
        color=color,
        linewidth=2,
        linestyle=style,
        alpha=0.8,
        zorder=5
    )
    ax.add_patch(arrow)
    
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y, label,
                ha='center', va='bottom',
                fontsize=8, color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=COLOR_BG,
                         edgecolor='white', alpha=0.9))

def main():
    fig, ax = plt.subplots(figsize=(14, 10), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, 'Docker Compose Network Isolation',
            ha='center', va='top',
            fontsize=18, fontweight='bold', color='white')
    
    ax.text(7, 9, 'Internal vs. External Network Access',
            ha='center', va='top',
            fontsize=12, color='#888888')
    
    # Draw network zones
    # External network (host)
    draw_network_zone(ax, 0.5, 6.5, 13, 2,
                     'Host Network (0.0.0.0)', COLOR_DANGER, alpha=0.15)
    
    # Internal app network
    draw_network_zone(ax, 1, 1, 12, 5,
                     'app-network (bridge driver)', COLOR_SUCCESS, alpha=0.2)
    
    # Draw containers
    # Web container (exposed)
    draw_container(ax, 2, 3.5, 2.5, 1.5,
                  'web', ':5000 → :5000', COLOR_PRIMARY, exposed=True)
    
    # Database container (internal only)
    draw_container(ax, 6, 3.5, 2.5, 1.5,
                  'db', ':5432 (internal)', COLOR_INFO, exposed=False)
    
    # Cache container (internal only)
    draw_container(ax, 10, 3.5, 2.5, 1.5,
                  'cache', ':6379 (internal)', COLOR_CAUTION, exposed=False)
    
    # External client
    client = Circle((7, 7.5), 0.4, facecolor='none',
                   edgecolor='white', linewidth=2, linestyle='--')
    ax.add_patch(client)
    ax.text(7, 7.5, '🌍', ha='center', va='center', fontsize=20)
    ax.text(7, 8.2, 'External Client', ha='center', va='bottom',
            fontsize=11, color='white')
    
    # Draw connections
    # External client to web (public access)
    draw_connection(ax, 7, 7.1, 3.25, 5, 
                   'http://localhost:5000', color='#FFD700')
    
    # Web to database (internal DNS)
    draw_connection(ax, 4.5, 4.25, 6, 4.25,
                   'postgresql://db:5432', style='-', color=COLOR_INFO)
    
    # Web to cache (internal DNS)
    draw_connection(ax, 4.5, 4.25, 10, 4.25,
                   'redis://cache:6379', style='-', color=COLOR_CAUTION)
    
    # Blocked access indicators
    # Client cannot reach db
    blocked_x1, blocked_y1 = 6.5, 7.1
    blocked_x2, blocked_y2 = 7.25, 5
    ax.plot([blocked_x1, blocked_x2], [blocked_y1, blocked_y2],
            'r--', linewidth=2, alpha=0.6)
    ax.text((blocked_x1 + blocked_x2)/2 - 0.3, (blocked_y1 + blocked_y2)/2,
            '🚫', fontsize=16, color=COLOR_DANGER)
    
    # Client cannot reach cache
    blocked_x1, blocked_y1 = 7.5, 7.1
    blocked_x2, blocked_y2 = 11.25, 5
    ax.plot([blocked_x1, blocked_x2], [blocked_y1, blocked_y2],
            'r--', linewidth=2, alpha=0.6)
    ax.text((blocked_x1 + blocked_x2)/2 + 0.3, (blocked_y1 + blocked_y2)/2,
            '🚫', fontsize=16, color=COLOR_DANGER)
    
    # DNS resolution box
    dns_text = (
        "Internal DNS Resolution:\n"
        "• 'db' → 172.18.0.3:5432\n"
        "• 'cache' → 172.18.0.4:6379\n"
        "• Services access each other by name"
    )
    ax.text(1.5, 2, dns_text,
            ha='left', va='top',
            fontsize=9, color='white',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_BG,
                     edgecolor=COLOR_SUCCESS, linewidth=2, alpha=0.9))
    
    # Port mapping explanation
    port_text = (
        "Port Mapping:\n"
        "ports:\n"
        "  - '5000:5000'  # Host:Container\n"
        "\n"
        "✅ Exposes container port 5000\n"
        "   to host port 5000"
    )
    ax.text(12.5, 2, port_text,
            ha='right', va='top',
            fontsize=9, color='white',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_BG,
                     edgecolor='#FFD700', linewidth=2, alpha=0.9))
    
    # Security note
    security_text = (
        "🔒 Security Best Practice:\n"
        "Only expose ports that need external access.\n"
        "Database and cache remain internal-only."
    )
    ax.text(7, 0.5, security_text,
            ha='center', va='bottom',
            fontsize=9, color='white',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_BG,
                     edgecolor=COLOR_SUCCESS, linewidth=2, alpha=0.9))
    
    # Legend
    legend_elements = [
        ('🌐', 'Exposed to host'),
        ('🚫', 'Blocked (no route)'),
        ('━━', 'Allowed connection')
    ]
    
    legend_x = 0.8
    for i, (symbol, label) in enumerate(legend_elements):
        ax.text(legend_x + i * 3, 6.2, f'{symbol} {label}',
                ha='left', va='top',
                fontsize=9, color='white')
    
    # Save
    output_dir = Path(__file__).parent.parent / 'img'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'ch02_network_isolation.png'
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=COLOR_BG,
                bbox_inches='tight', pad_inches=0.3)
    print(f"✅ Generated: {output_path}")

if __name__ == '__main__':
    main()
