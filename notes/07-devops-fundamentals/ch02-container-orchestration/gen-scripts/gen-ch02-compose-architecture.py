"""
Generate service dependency graph for Docker Compose architecture.

Shows the relationship between Flask (web), PostgreSQL (db), and Redis (cache)
with dependency arrows and health check indicators.

Output: img/ch02_compose_architecture.png
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

# Color palette (consistent with repo conventions)
COLOR_PRIMARY = "#1e3a8a"  # Deep blue
COLOR_SUCCESS = "#15803d"  # Green
COLOR_INFO = "#1d4ed8"     # Blue
COLOR_CAUTION = "#b45309"  # Orange
COLOR_BG = "#1a1a2e"       # Dark background

def draw_service_box(ax, x, y, width, height, label, color, has_health_check=False):
    """Draw a service box with optional health check indicator."""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.1",
        facecolor=color,
        edgecolor="white",
        linewidth=2,
        alpha=0.9
    )
    ax.add_patch(box)
    
    # Service label
    ax.text(
        x + width/2, y + height/2,
        label,
        ha='center', va='center',
        fontsize=14, fontweight='bold',
        color='white'
    )
    
    # Health check indicator
    if has_health_check:
        check_x = x + width - 0.3
        check_y = y + height - 0.3
        ax.plot(check_x, check_y, 'o', color=COLOR_SUCCESS, markersize=10, 
                markeredgecolor='white', markeredgewidth=2)
        ax.text(check_x + 0.15, check_y, '✓', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')

def draw_dependency_arrow(ax, x1, y1, x2, y2, label=""):
    """Draw a dependency arrow with optional label."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->,head_width=0.4,head_length=0.8',
        color='white',
        linewidth=2,
        alpha=0.8,
        zorder=1
    )
    ax.add_patch(arrow)
    
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.3, label,
                ha='center', va='bottom',
                fontsize=9, color='white',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=COLOR_BG, 
                         edgecolor='white', linewidth=1, alpha=0.8))

def main():
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(6, 7.5, 'Docker Compose Service Architecture',
            ha='center', va='top',
            fontsize=18, fontweight='bold', color='white')
    
    ax.text(6, 7, 'Flask + PostgreSQL + Redis (3-tier stack)',
            ha='center', va='top',
            fontsize=12, color='#888888')
    
    # Draw services
    # Web tier (Flask)
    draw_service_box(ax, 4.5, 5, 3, 1.2, 'web\n(Flask)', COLOR_PRIMARY, has_health_check=True)
    
    # Data tier (PostgreSQL)
    draw_service_box(ax, 1, 2.5, 3, 1.2, 'db\n(PostgreSQL)', COLOR_INFO, has_health_check=True)
    
    # Cache tier (Redis)
    draw_service_box(ax, 8, 2.5, 3, 1.2, 'cache\n(Redis)', COLOR_CAUTION)
    
    # External client
    client_box = FancyBboxPatch(
        (4.5, 7), 3, 0.8,
        boxstyle="round,pad=0.05",
        facecolor='none',
        edgecolor='white',
        linewidth=2,
        linestyle='--',
        alpha=0.6
    )
    ax.add_patch(client_box)
    ax.text(6, 7.4, 'Client\n(Browser)', ha='center', va='center',
            fontsize=11, color='white')
    
    # Draw dependency arrows
    # Client → Web
    draw_dependency_arrow(ax, 6, 7, 6, 6.2, "HTTP :5000")
    
    # Web → Database
    draw_dependency_arrow(ax, 5.2, 5, 3.5, 3.7, "depends_on:\n  service_healthy")
    
    # Web → Cache
    draw_dependency_arrow(ax, 6.8, 5, 8.5, 3.7, "depends_on:\n  service_started")
    
    # Network isolation box
    network_box = FancyBboxPatch(
        (0.5, 2), 11, 4.5,
        boxstyle="round,pad=0.2",
        facecolor='none',
        edgecolor=COLOR_SUCCESS,
        linewidth=2,
        linestyle=':',
        alpha=0.5,
        zorder=0
    )
    ax.add_patch(network_box)
    ax.text(11.3, 4.2, 'app-network\n(bridge)', ha='left', va='center',
            fontsize=10, color=COLOR_SUCCESS, style='italic')
    
    # Volume indicators
    ax.text(2.5, 1.8, '💾 postgres-data', ha='center', va='top',
            fontsize=9, color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLOR_BG,
                     edgecolor=COLOR_INFO, linewidth=1.5))
    
    ax.text(9.5, 1.8, '💾 redis-data', ha='center', va='top',
            fontsize=9, color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLOR_BG,
                     edgecolor=COLOR_CAUTION, linewidth=1.5))
    
    # Legend
    legend_y = 0.8
    ax.text(1, legend_y, '✓ = Health check enabled', ha='left', va='center',
            fontsize=9, color='white')
    ax.text(6, legend_y, '💾 = Persistent volume', ha='left', va='center',
            fontsize=9, color='white')
    
    # Startup order annotation
    startup_text = (
        "Startup order:\n"
        "1. cache (no dependencies)\n"
        "2. db (waits for health check)\n"
        "3. web (waits for db healthy + cache started)"
    )
    ax.text(11.5, 0.5, startup_text, ha='right', va='bottom',
            fontsize=8, color='#888888',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_BG,
                     edgecolor='white', linewidth=1, alpha=0.8))
    
    # Save
    output_dir = Path(__file__).parent.parent / 'img'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'ch02_compose_architecture.png'
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=COLOR_BG, 
                bbox_inches='tight', pad_inches=0.3)
    print(f"✅ Generated: {output_path}")

if __name__ == '__main__':
    main()
