"""
Generate volume persistence diagram showing named volumes across container restarts.

Illustrates:
- Container lifecycle (up → down → up)
- Named volumes persisting data
- Anonymous volumes being deleted
- Data flow between container and volume

Output: img/ch02_volume_persistence.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from pathlib import Path

# Color palette
COLOR_PRIMARY = "#1e3a8a"
COLOR_SUCCESS = "#15803d"
COLOR_INFO = "#1d4ed8"
COLOR_CAUTION = "#b45309"
COLOR_DANGER = "#b91c1c"
COLOR_BG = "#1a1a2e"

def draw_container_state(ax, x, y, name, state, color):
    """Draw a container in a specific state."""
    if state == 'running':
        box = FancyBboxPatch(
            (x, y), 2, 1,
            boxstyle="round,pad=0.08",
            facecolor=color,
            edgecolor='white',
            linewidth=2,
            alpha=0.9
        )
        status_color = COLOR_SUCCESS
        status_text = '✓ Running'
    elif state == 'stopped':
        box = FancyBboxPatch(
            (x, y), 2, 1,
            boxstyle="round,pad=0.08",
            facecolor='#333333',
            edgecolor='#666666',
            linewidth=2,
            linestyle='--',
            alpha=0.5
        )
        status_color = '#888888'
        status_text = '✗ Stopped'
    else:  # removed
        box = Rectangle(
            (x, y), 2, 1,
            facecolor='none',
            edgecolor='#444444',
            linewidth=1,
            linestyle=':',
            alpha=0.3
        )
        status_color = COLOR_DANGER
        status_text = '⚠ Removed'
    
    ax.add_patch(box)
    
    # Container name
    ax.text(x + 1, y + 0.6, name,
            ha='center', va='center',
            fontsize=11, fontweight='bold',
            color='white' if state == 'running' else '#888888')
    
    # Status indicator
    ax.text(x + 1, y + 0.3, status_text,
            ha='center', va='center',
            fontsize=8, color=status_color)

def draw_volume(ax, x, y, name, color, persistent=True):
    """Draw a volume (cylinder shape)."""
    # Cylinder body
    cylinder = FancyBboxPatch(
        (x, y), 1.5, 1.2,
        boxstyle="round,pad=0.05",
        facecolor=color,
        edgecolor='white',
        linewidth=2,
        alpha=0.8
    )
    ax.add_patch(cylinder)
    
    # Volume icon
    ax.text(x + 0.75, y + 0.8, '💾',
            ha='center', va='center', fontsize=16)
    
    # Volume name
    ax.text(x + 0.75, y + 0.4, name,
            ha='center', va='center',
            fontsize=9, fontweight='bold', color='white')
    
    # Persistence indicator
    if persistent:
        ax.text(x + 0.75, y + 0.1, 'Named',
                ha='center', va='center',
                fontsize=7, color=COLOR_SUCCESS,
                bbox=dict(boxstyle='round,pad=0.2', facecolor=COLOR_BG,
                         edgecolor=COLOR_SUCCESS, linewidth=1))

def draw_data_flow(ax, x1, y1, x2, y2, bidirectional=True):
    """Draw data flow arrow between container and volume."""
    if bidirectional:
        # Forward arrow
        arrow1 = FancyArrowPatch(
            (x1, y1 + 0.1), (x2, y2 + 0.1),
            arrowstyle='->,head_width=0.3,head_length=0.5',
            color=COLOR_SUCCESS,
            linewidth=2,
            alpha=0.7
        )
        ax.add_patch(arrow1)
        
        # Backward arrow
        arrow2 = FancyArrowPatch(
            (x2, y2 - 0.1), (x1, y1 - 0.1),
            arrowstyle='->,head_width=0.3,head_length=0.5',
            color=COLOR_INFO,
            linewidth=2,
            alpha=0.7
        )
        ax.add_patch(arrow2)
    else:
        # Single dashed arrow (disconnected)
        ax.plot([x1, x2], [y1, y2], '--',
                color='#666666', linewidth=2, alpha=0.4)

def main():
    fig, ax = plt.subplots(figsize=(14, 10), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, 'Docker Compose Volume Persistence',
            ha='center', va='top',
            fontsize=18, fontweight='bold', color='white')
    
    ax.text(7, 9, 'Named Volumes Survive Container Restarts',
            ha='center', va='top',
            fontsize=12, color='#888888')
    
    # Timeline labels
    timeline_y = 7.5
    phases = [
        ('T=0', 'docker compose up', 1.5),
        ('T=1', 'docker compose down', 5.5),
        ('T=2', 'docker compose up', 10.5)
    ]
    
    for label, cmd, x in phases:
        ax.text(x, timeline_y + 0.8, label,
                ha='center', va='bottom',
                fontsize=11, fontweight='bold', color='white')
        ax.text(x, timeline_y + 0.5, cmd,
                ha='center', va='top',
                fontsize=9, color='#888888',
                style='italic')
    
    # Phase 1: Initial startup
    phase1_x = 0.5
    ax.text(1.5, 7, 'Phase 1: Services Running',
            ha='center', va='bottom',
            fontsize=12, fontweight='bold', color=COLOR_SUCCESS)
    
    # Container and volume
    draw_container_state(ax, phase1_x, 5, 'db\nPostgreSQL', 'running', COLOR_INFO)
    draw_volume(ax, phase1_x + 0.25, 3, 'postgres-data', COLOR_INFO, persistent=True)
    draw_data_flow(ax, phase1_x + 1, 5, phase1_x + 1, 4.2)
    
    # Data write indicator
    ax.text(phase1_x + 1, 4.6, 'Write:\nUSERS table',
            ha='center', va='center',
            fontsize=8, color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLOR_BG,
                     edgecolor=COLOR_SUCCESS, linewidth=1))
    
    # Phase 2: Stopped and removed
    phase2_x = 4.5
    ax.text(5.5, 7, 'Phase 2: Containers Removed',
            ha='center', va='bottom',
            fontsize=12, fontweight='bold', color='#888888')
    
    draw_container_state(ax, phase2_x, 5, 'db\n(removed)', 'removed', COLOR_INFO)
    draw_volume(ax, phase2_x + 0.25, 3, 'postgres-data', COLOR_INFO, persistent=True)
    draw_data_flow(ax, phase2_x + 1, 5, phase2_x + 1, 4.2, bidirectional=False)
    
    # Persistence note
    ax.text(phase2_x + 1, 2.5, '✅ Data preserved!\nVolume NOT deleted',
            ha='center', va='top',
            fontsize=9, color=COLOR_SUCCESS,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLOR_BG,
                     edgecolor=COLOR_SUCCESS, linewidth=2))
    
    # Phase 3: Restart
    phase3_x = 9.5
    ax.text(10.5, 7, 'Phase 3: Services Restarted',
            ha='center', va='bottom',
            fontsize=12, fontweight='bold', color=COLOR_SUCCESS)
    
    draw_container_state(ax, phase3_x, 5, 'db\nPostgreSQL', 'running', COLOR_INFO)
    draw_volume(ax, phase3_x + 0.25, 3, 'postgres-data', COLOR_INFO, persistent=True)
    draw_data_flow(ax, phase3_x + 1, 5, phase3_x + 1, 4.2)
    
    # Data read indicator
    ax.text(phase3_x + 1, 4.6, 'Read:\nUSERS table\n(same data!)',
            ha='center', va='center',
            fontsize=8, color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLOR_BG,
                     edgecolor=COLOR_INFO, linewidth=1))
    
    # Timeline arrows
    for i in range(len(phases) - 1):
        x1 = phases[i][2] + 1.5
        x2 = phases[i+1][2] - 1.5
        arrow = FancyArrowPatch(
            (x1, timeline_y), (x2, timeline_y),
            arrowstyle='->,head_width=0.4,head_length=0.8',
            color='white',
            linewidth=2,
            alpha=0.5
        )
        ax.add_patch(arrow)
    
    # Comparison: Anonymous volume (gets deleted)
    compare_y = 1.5
    ax.text(7, compare_y + 0.5, 'Comparison: Anonymous Volume (No Persistence)',
            ha='center', va='top',
            fontsize=11, fontweight='bold', color=COLOR_DANGER)
    
    # Before
    ax.text(3, compare_y - 0.5, 'Before\n`docker compose down`',
            ha='center', va='top',
            fontsize=9, color='white')
    anon_vol1 = Circle((3, compare_y - 1.5), 0.3,
                      facecolor=COLOR_CAUTION, edgecolor='white', linewidth=2)
    ax.add_patch(anon_vol1)
    ax.text(3, compare_y - 1.5, '📦', ha='center', va='center', fontsize=12)
    ax.text(3, compare_y - 2, 'Anonymous\nvolume',
            ha='center', va='top', fontsize=8, color='white')
    
    # Arrow
    arrow = FancyArrowPatch(
        (4, compare_y - 1.5), (6, compare_y - 1.5),
        arrowstyle='->,head_width=0.4,head_length=0.8',
        color=COLOR_DANGER,
        linewidth=2
    )
    ax.add_patch(arrow)
    ax.text(5, compare_y - 1.2, 'compose down', ha='center', va='bottom',
            fontsize=8, color=COLOR_DANGER)
    
    # After (deleted)
    ax.text(8, compare_y - 0.5, 'After\n(data lost)',
            ha='center', va='top',
            fontsize=9, color=COLOR_DANGER)
    ax.text(8, compare_y - 1.5, '🗑️', ha='center', va='center', fontsize=24,
            color=COLOR_DANGER)
    ax.text(8, compare_y - 2, 'Volume\ndeleted',
            ha='center', va='top', fontsize=8, color=COLOR_DANGER)
    
    # Best practice box
    best_practice = (
        "✅ Best Practice:\n"
        "Always use named volumes for data that must persist:\n"
        "\n"
        "volumes:\n"
        "  postgres-data:  # Survives 'down'\n"
        "\n"
        "services:\n"
        "  db:\n"
        "    volumes:\n"
        "      - postgres-data:/var/lib/postgresql/data\n"
        "\n"
        "To delete volumes: docker compose down --volumes"
    )
    ax.text(11, compare_y - 0.5, best_practice,
            ha='left', va='top',
            fontsize=8, color='white', family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_BG,
                     edgecolor=COLOR_SUCCESS, linewidth=2, alpha=0.9))
    
    # Legend
    legend_x = 0.5
    legend_y = 0.2
    ax.text(legend_x, legend_y, '💾 Named volume (persistent)   |   📦 Anonymous volume (temporary)',
            ha='left', va='center',
            fontsize=9, color='white')
    
    # Save
    output_dir = Path(__file__).parent.parent / 'img'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'ch02_volume_persistence.png'
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=COLOR_BG,
                bbox_inches='tight', pad_inches=0.3)
    print(f"✅ Generated: {output_path}")

if __name__ == '__main__':
    main()
