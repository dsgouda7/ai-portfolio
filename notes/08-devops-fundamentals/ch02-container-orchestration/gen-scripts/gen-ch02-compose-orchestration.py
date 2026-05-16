"""
Generate Docker Compose startup sequence animation showing health-check-based 
dependency ordering.

Visual flow (5 frames):
1. Compose reads docker-compose.yml (YAML icon → parser)
2. Creates network bridge (network topology: isolated bridge created)
3. Starts `cache` service (no dependencies, starts immediately)
4. Starts `db` service (waits for health check)
5. Starts `web` service (waits for both cache+db ready)

Output: img/ch02-compose-orchestration.gif
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
import numpy as np

# Color palette (consistent with DevOps conventions)
COLOR_BG = "#1a1a2e"           # Dark background
COLOR_NETWORK = "#2563eb"      # Blue for network
COLOR_HEALTHY = "#15803d"      # Green for healthy services
COLOR_WAITING = "#eab308"      # Yellow for waiting/health checks
COLOR_TEXT = "#ffffff"         # White text
COLOR_YAML = "#60a5fa"         # Light blue for YAML
COLOR_GRAY = "#6b7280"         # Gray for inactive

def draw_yaml_icon(ax, x, y, size=0.5, active=True):
    """Draw a YAML file icon."""
    color = COLOR_YAML if active else COLOR_GRAY
    rect = FancyBboxPatch(
        (x - size/2, y - size/2), size, size * 1.3,
        boxstyle="round,pad=0.02",
        facecolor=color,
        edgecolor=COLOR_TEXT,
        linewidth=2,
        alpha=0.9 if active else 0.3
    )
    ax.add_patch(rect)
    # YAML text lines
    if active:
        for i in range(3):
            line_y = y + 0.15 - i * 0.12
            ax.plot([x - 0.15, x + 0.15], [line_y, line_y], 
                   color=COLOR_BG, linewidth=2)
    ax.text(x, y - size/2 - 0.15, "docker-\ncompose.yml",
           ha='center', va='top', fontsize=9, color=COLOR_TEXT)

def draw_network_bridge(ax, x, y, width=2.5, height=1.8, active=True):
    """Draw network bridge representation."""
    color = COLOR_NETWORK if active else COLOR_GRAY
    alpha = 0.6 if active else 0.2
    
    # Bridge box
    rect = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.1",
        facecolor='none',
        edgecolor=color,
        linewidth=2.5,
        linestyle='--' if active else ':',
        alpha=alpha
    )
    ax.add_patch(rect)
    
    ax.text(x, y + height/2 + 0.15, "app-network",
           ha='center', va='bottom', fontsize=10, 
           color=color if active else COLOR_GRAY,
           fontweight='bold' if active else 'normal')

def draw_service_box(ax, x, y, service_name, icon, status='inactive', 
                     has_health_check=False):
    """
    Draw a service container box.
    status: 'inactive' (gray), 'starting' (yellow), 'healthy' (green)
    """
    colors = {
        'inactive': COLOR_GRAY,
        'starting': COLOR_WAITING,
        'healthy': COLOR_HEALTHY
    }
    color = colors.get(status, COLOR_GRAY)
    alpha = 0.3 if status == 'inactive' else 0.9
    
    box = FancyBboxPatch(
        (x - 0.35, y - 0.25), 0.7, 0.5,
        boxstyle="round,pad=0.05",
        facecolor=color,
        edgecolor=COLOR_TEXT,
        linewidth=2,
        alpha=alpha
    )
    ax.add_patch(box)
    
    # Service icon/label
    ax.text(x, y + 0.05, icon, ha='center', va='center',
           fontsize=16, color=COLOR_TEXT)
    ax.text(x, y - 0.15, service_name, ha='center', va='center',
           fontsize=8, color=COLOR_TEXT, fontweight='bold')
    
    # Health check indicator
    if has_health_check and status == 'healthy':
        check_circle = Circle((x + 0.25, y + 0.15), 0.08,
                             facecolor=COLOR_HEALTHY,
                             edgecolor=COLOR_TEXT,
                             linewidth=1.5)
        ax.add_patch(check_circle)
        ax.text(x + 0.25, y + 0.15, '✓', ha='center', va='center',
               fontsize=8, fontweight='bold', color=COLOR_TEXT)
    elif has_health_check and status == 'starting':
        check_circle = Circle((x + 0.25, y + 0.15), 0.08,
                             facecolor=COLOR_WAITING,
                             edgecolor=COLOR_TEXT,
                             linewidth=1.5)
        ax.add_patch(check_circle)
        ax.text(x + 0.25, y + 0.15, '?', ha='center', va='center',
               fontsize=8, fontweight='bold', color=COLOR_TEXT)

def draw_arrow(ax, x1, y1, x2, y2, color=COLOR_TEXT, label=""):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->,head_width=0.3,head_length=0.5',
        color=color,
        linewidth=2.5,
        alpha=0.9
    )
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.15, label, ha='center', va='bottom',
               fontsize=8, color=color,
               bbox=dict(boxstyle='round,pad=0.2', facecolor=COLOR_BG,
                        edgecolor=color, linewidth=1))

def create_frame(frame_num):
    """Create a single frame of the animation."""
    fig, ax = plt.subplots(figsize=(10, 7), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.5, 4.5)
    ax.axis('off')
    
    # Title
    ax.text(2.5, 4.3, 'Docker Compose Startup Sequence',
           ha='center', va='top', fontsize=16, fontweight='bold',
           color=COLOR_TEXT)
    
    # Frame-specific content
    if frame_num == 0:
        # Frame 1: Compose reads YAML
        draw_yaml_icon(ax, 2.5, 3.2, active=True)
        ax.text(2.5, 2.3, "Compose engine reads\nconfiguration", 
               ha='center', va='top', fontsize=11, color=COLOR_TEXT)
        
        caption = ("Compose parses docker-compose.yml to understand\n"
                  "service definitions, dependencies, and startup order")
        
    elif frame_num == 1:
        # Frame 2: Creates network
        draw_yaml_icon(ax, 1.2, 3.5, size=0.4, active=False)
        draw_network_bridge(ax, 2.5, 2.0, active=True)
        draw_arrow(ax, 1.5, 3.3, 2.0, 2.6, color=COLOR_NETWORK)
        
        ax.text(2.5, 0.8, "Creates isolated bridge network\nfor service communication",
               ha='center', va='top', fontsize=11, color=COLOR_TEXT)
        
        caption = ("Network isolation ensures services communicate\n"
                  "securely without exposing internal ports to host")
        
    elif frame_num == 2:
        # Frame 3: Starts cache (no dependencies)
        draw_yaml_icon(ax, 0.8, 3.8, size=0.35, active=False)
        draw_network_bridge(ax, 2.5, 2.0, active=True)
        draw_service_box(ax, 2.5, 2.0, "cache", "🔴", status='healthy')
        
        ax.text(2.5, 0.8, "`cache` starts immediately\n(no dependencies declared)",
               ha='center', va='top', fontsize=11, color=COLOR_TEXT)
        
        caption = ("Redis cache starts first - depends_on not specified,\n"
                  "so Compose launches it as soon as network is ready")
        
    elif frame_num == 3:
        # Frame 4: Starts db (waits for health check)
        draw_yaml_icon(ax, 0.8, 3.8, size=0.35, active=False)
        draw_network_bridge(ax, 2.5, 2.0, active=True)
        draw_service_box(ax, 1.5, 2.0, "cache", "🔴", status='healthy')
        draw_service_box(ax, 3.5, 2.0, "db", "🐘", status='healthy',
                        has_health_check=True)
        
        ax.text(2.5, 0.8, "`db` starts and passes health check\n"
                          "(PostgreSQL ready to accept connections)",
               ha='center', va='top', fontsize=11, color=COLOR_TEXT)
        
        caption = ("Health check validates that PostgreSQL is actually ready,\n"
                  "not just that the container started (critical for depends_on)")
        
    else:  # frame_num == 4
        # Frame 5: Starts web (waits for both)
        draw_yaml_icon(ax, 0.8, 3.8, size=0.35, active=False)
        draw_network_bridge(ax, 2.5, 2.0, active=True)
        draw_service_box(ax, 1.2, 2.0, "cache", "🔴", status='healthy')
        draw_service_box(ax, 2.5, 2.0, "db", "🐘", status='healthy',
                        has_health_check=True)
        draw_service_box(ax, 3.8, 2.0, "web", "🌐", status='healthy',
                        has_health_check=True)
        
        # Connection arrows
        draw_arrow(ax, 3.5, 2.0, 2.7, 2.0, color=COLOR_HEALTHY)
        draw_arrow(ax, 3.5, 1.9, 1.5, 1.9, color=COLOR_HEALTHY)
        
        ax.text(2.5, 0.8, "`web` starts after dependencies satisfied\n"
                          "(Flask connects to both db and cache)",
               ha='center', va='top', fontsize=11, color=COLOR_TEXT)
        
        caption = ("Declarative orchestration complete: Compose materialized\n"
                  "the desired state in correct order, preventing race conditions")
    
    # Caption box at bottom
    caption_box = FancyBboxPatch(
        (0.2, -0.4), 5.1, 0.35,
        boxstyle="round,pad=0.08",
        facecolor='#2a2a3e',
        edgecolor=COLOR_TEXT,
        linewidth=1.5,
        alpha=0.9
    )
    ax.add_patch(caption_box)
    ax.text(2.75, -0.22, caption, ha='center', va='center',
           fontsize=9, color=COLOR_TEXT, style='italic')
    
    return fig

def main():
    """Generate the animation GIF."""
    print("Generating Docker Compose orchestration animation...")
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / "img"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "ch02-compose-orchestration.gif"
    
    # Create animation
    fig = plt.figure(figsize=(10, 7))
    
    def animate(frame_num):
        plt.clf()
        return create_frame(frame_num)
    
    anim = FuncAnimation(
        fig, animate, frames=5, interval=800, repeat=True
    )
    
    # Save as GIF
    writer = PillowWriter(fps=1.25)  # 0.8s per frame = 1.25 fps
    anim.save(output_path, writer=writer)
    
    plt.close('all')
    
    print(f"✓ Animation saved to: {output_path}")
    print(f"  Frames: 5")
    print(f"  Duration: 4.0s (0.8s per frame)")
    print(f"  Theme: Dark (#1a1a2e background)")

if __name__ == "__main__":
    main()
