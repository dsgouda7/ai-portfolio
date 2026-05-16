"""
Generate Ch.4 Secrets Management Diagram

Visualizes secure flow: GitHub Secrets → Actions → Docker Hub

Output:
- ch04_secrets_management.png (static diagram)
- ch04_secrets_management.gif (animated flow)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image
import io

# Configuration
OUTPUT_DIR = '../img'
FILENAME = 'ch04_secrets_management'

# Colors
COLOR_SECRET = '#E74C3C'      # Red (sensitive data)
COLOR_GITHUB = '#24292E'      # GitHub dark
COLOR_ACTION = '#4ECDC4'      # Teal
COLOR_DOCKER = '#2496ED'      # Docker blue
COLOR_SECURE = '#27AE60'      # Green (secure)
COLOR_INSECURE = '#E74C3C'    # Red (insecure)
COLOR_BG = '#F8F9FA'

def draw_lock_icon(ax, x, y, locked=True, scale=1.0):
    """Draw a lock icon"""
    color = COLOR_SECURE if locked else COLOR_INSECURE
    
    # Lock body (rectangle)
    rect = patches.Rectangle(
        (x - 0.15 * scale, y - 0.25 * scale),
        0.3 * scale, 0.3 * scale,
        facecolor=color, edgecolor='white', linewidth=1.5
    )
    ax.add_patch(rect)
    
    # Lock shackle (arc)
    if locked:
        arc = patches.Arc(
            (x, y + 0.05 * scale), 0.25 * scale, 0.25 * scale,
            angle=0, theta1=0, theta2=180,
            color=color, linewidth=3
        )
        ax.add_patch(arc)
    else:
        # Open lock (shackle tilted)
        ax.plot([x - 0.125 * scale, x + 0.05 * scale], 
               [y + 0.05 * scale, y + 0.2 * scale],
               color=color, linewidth=3)

def draw_key_icon(ax, x, y, scale=1.0):
    """Draw a key icon"""
    # Key shaft
    ax.plot([x - 0.2 * scale, x + 0.1 * scale], [y, y],
           color=COLOR_SECRET, linewidth=3)
    
    # Key head (circle)
    circle = plt.Circle((x - 0.2 * scale, y), 0.08 * scale,
                       color=COLOR_SECRET, zorder=10)
    ax.add_patch(circle)
    
    # Key teeth
    teeth_x = [x + 0.05 * scale, x + 0.1 * scale]
    for tx in teeth_x:
        ax.plot([tx, tx], [y, y - 0.08 * scale],
               color=COLOR_SECRET, linewidth=2)

def create_component_box(ax, x, y, width, height, label, sublabel, color, 
                        show_lock=False, locked=True, highlight=False):
    """Draw a component box with optional lock icon"""
    alpha = 1.0 if highlight else 0.5
    
    # Main box
    rect = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.1",
        edgecolor='#333' if highlight else '#999',
        facecolor=color,
        linewidth=3 if highlight else 1.5,
        alpha=alpha
    )
    ax.add_patch(rect)
    
    # Lock icon
    if show_lock:
        draw_lock_icon(ax, x + width - 0.4, y + height - 0.4, locked)
    
    # Labels
    ax.text(x + width/2, y + height - 0.5, label,
           ha='center', va='center', fontsize=12, fontweight='bold',
           color='white', alpha=1.0 if highlight else 0.8)
    ax.text(x + width/2, y + 0.3, sublabel,
           ha='center', va='bottom', fontsize=8, color='white',
           style='italic', alpha=1.0 if highlight else 0.6)

def draw_secure_arrow(ax, x1, y1, x2, y2, label='', encrypted=True, 
                     highlight=False):
    """Draw an arrow with encryption indicator"""
    alpha = 1.0 if highlight else 0.4
    color = COLOR_SECURE if encrypted else COLOR_INSECURE
    
    # Arrow
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->', mutation_scale=20,
        linewidth=3 if highlight else 2,
        color=color, alpha=alpha,
        connectionstyle='arc3,rad=0.1'
    )
    ax.add_patch(arrow)
    
    # Label with lock icon
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    
    if label:
        # Background box
        bbox_color = 'white' if encrypted else '#FFE5E5'
        ax.text(mid_x, mid_y, label,
               ha='center', va='center', fontsize=9,
               bbox=dict(boxstyle='round,pad=0.4', facecolor=bbox_color,
                        edgecolor=color, linewidth=2, alpha=1.0))
        
        # Small lock icon next to label
        if encrypted:
            draw_lock_icon(ax, mid_x + 0.8, mid_y, locked=True, scale=0.5)
        else:
            draw_lock_icon(ax, mid_x + 0.8, mid_y, locked=False, scale=0.5)

def create_secrets_diagram(highlight_stage=None, show_insecure=False):
    """
    Create secrets management diagram
    
    Args:
        highlight_stage: Which stage to highlight (1-4, or None)
        show_insecure: If True, show insecure anti-pattern
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    
    # Title
    title = 'Secrets Management Flow' if not show_insecure else '❌ INSECURE: Never Commit Secrets!'
    title_color = '#333' if not show_insecure else COLOR_INSECURE
    ax.text(6, 7.5, title, 
           ha='center', va='top', fontsize=16, fontweight='bold',
           color=title_color)
    
    if not show_insecure:
        # Secure flow
        
        # Component boxes
        components = [
            {
                'x': 0.5, 'y': 5, 'w': 2.5, 'h': 1.5,
                'label': 'GitHub\nSecrets', 'sublabel': 'Encrypted vault',
                'color': COLOR_SECRET, 'lock': True, 'locked': True,
                'stage': 1
            },
            {
                'x': 3.5, 'y': 5, 'w': 2.5, 'h': 1.5,
                'label': 'Workflow', 'sublabel': 'Access via ${{ secrets }}',
                'color': COLOR_ACTION, 'lock': False, 'locked': True,
                'stage': 2
            },
            {
                'x': 6.5, 'y': 5, 'w': 2.5, 'h': 1.5,
                'label': 'Runner', 'sublabel': 'Temporary VM',
                'color': COLOR_GITHUB, 'lock': False, 'locked': True,
                'stage': 3
            },
            {
                'x': 9.5, 'y': 5, 'w': 2.5, 'h': 1.5,
                'label': 'Docker Hub', 'sublabel': 'Push image',
                'color': COLOR_DOCKER, 'lock': True, 'locked': True,
                'stage': 4
            }
        ]
        
        # Draw components
        for comp in components:
            is_highlighted = (highlight_stage == comp['stage']) or (highlight_stage is None)
            create_component_box(
                ax, comp['x'], comp['y'], comp['w'], comp['h'],
                comp['label'], comp['sublabel'], comp['color'],
                show_lock=comp['lock'], locked=comp['locked'],
                highlight=is_highlighted
            )
        
        # Draw arrows
        arrows = [
            {
                'x1': 3.0, 'y1': 5.75, 'x2': 3.5, 'y2': 5.75,
                'label': '1. Request', 'stage': 1
            },
            {
                'x1': 6.0, 'y1': 5.75, 'x2': 6.5, 'y2': 5.75,
                'label': '2. Inject', 'stage': 2
            },
            {
                'x1': 9.0, 'y1': 5.75, 'x2': 9.5, 'y2': 5.75,
                'label': '3. Authenticate', 'stage': 3
            }
        ]
        
        for arrow in arrows:
            is_highlighted = (highlight_stage == arrow['stage']) or (highlight_stage is None)
            draw_secure_arrow(
                ax, arrow['x1'], arrow['y1'], arrow['x2'], arrow['y2'],
                arrow['label'], encrypted=True, highlight=is_highlighted
            )
        
        # Security features box
        security_y = 3.5
        security_features = [
            '🔒 Secrets encrypted at rest (AES-256)',
            '🔒 Never exposed in logs or UI',
            '🔒 Scoped to repository/organization',
            '🔒 Automatically redacted in outputs',
            '🔒 Runner VM destroyed after job'
        ]
        
        for i, feature in enumerate(security_features):
            ax.text(1, security_y - i*0.35, feature,
                   ha='left', va='top', fontsize=9, color='#555')
        
        # Best practices box
        practices_y = 3.5
        practices = [
            '✓ Use secrets for all sensitive data',
            '✓ Rotate secrets regularly',
            '✓ Use fine-grained tokens (not passwords)',
            '✓ Separate dev/staging/prod secrets',
            '✓ Audit secret access via workflow logs'
        ]
        
        for i, practice in enumerate(practices):
            ax.text(7, practices_y - i*0.35, practice,
                   ha='left', va='top', fontsize=9, color='#555')
    
    else:
        # Insecure anti-pattern
        
        # Git repo with exposed secret
        repo_rect = FancyBboxPatch(
            (1, 5), 3, 1.5,
            boxstyle="round,pad=0.1",
            edgecolor=COLOR_INSECURE,
            facecolor='white',
            linewidth=3
        )
        ax.add_patch(repo_rect)
        ax.text(2.5, 6.2, 'Git Repository', ha='center', fontsize=12, 
               fontweight='bold', color=COLOR_INSECURE)
        ax.text(2.5, 5.7, '.env file committed', ha='center', fontsize=9,
               color=COLOR_INSECURE, style='italic')
        
        # Exposed secret
        secret_text = 'DOCKER_TOKEN=dckr_pat_abc123...'
        ax.text(2.5, 5.3, secret_text, ha='center', fontsize=8,
               family='monospace', color=COLOR_INSECURE,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFE5E5',
                        edgecolor=COLOR_INSECURE, linewidth=2))
        
        # Open lock icon
        draw_lock_icon(ax, 3.8, 6, locked=False, scale=1.5)
        
        # Danger indicators
        dangers = [
            ('5', '5.5', '❌ Visible in git history forever'),
            ('5', '4.5', '❌ Accessible to anyone who clones repo'),
            ('5', '3.5', '❌ Leaked in public forks'),
            ('5', '2.5', '❌ Compromised if repo is hacked'),
        ]
        
        for x, y, text in dangers:
            ax.text(float(x), float(y), text, ha='left', fontsize=10,
                   color=COLOR_INSECURE, fontweight='bold')
        
        # Correct approach
        ax.text(2.5, 1.5, '✓ INSTEAD: Use GitHub Secrets', ha='center',
               fontsize=12, fontweight='bold', color=COLOR_SECURE)
        correct_steps = [
            '1. Settings → Secrets → New secret',
            '2. Store: DOCKER_TOKEN = dckr_pat_abc123...',
            '3. Access in workflow: ${{ secrets.DOCKER_TOKEN }}',
            '4. Never commit secrets to git!'
        ]
        for i, step in enumerate(correct_steps):
            ax.text(2.5, 1.0 - i*0.25, step, ha='center', fontsize=9,
                   color='#555')
    
    plt.tight_layout()
    return fig

def create_animation():
    """Create animated GIF showing secrets flow"""
    frames = []
    
    # Frame 1: Full flow
    fig = create_secrets_diagram()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
               facecolor=COLOR_BG)
    buf.seek(0)
    frames.append(Image.open(buf).copy())
    plt.close(fig)
    
    # Frames 2-5: Highlight each stage
    for stage in range(1, 5):
        fig = create_secrets_diagram(highlight_stage=stage)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                   facecolor=COLOR_BG)
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        plt.close(fig)
    
    # Frame 6: Show insecure anti-pattern
    fig = create_secrets_diagram(show_insecure=True)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
               facecolor=COLOR_BG)
    buf.seek(0)
    frames.append(Image.open(buf).copy())
    plt.close(fig)
    
    # Frame 7: Back to secure flow
    fig = create_secrets_diagram()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
               facecolor=COLOR_BG)
    buf.seek(0)
    frames.append(Image.open(buf).copy())
    plt.close(fig)
    
    # Save GIF
    frames[0].save(
        f'{OUTPUT_DIR}/{FILENAME}.gif',
        save_all=True,
        append_images=frames[1:],
        duration=[1500, 800, 800, 800, 800, 1500, 1500],  # ms per frame
        loop=0
    )
    print(f"✅ Created {OUTPUT_DIR}/{FILENAME}.gif")

if __name__ == '__main__':
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Static diagram
    fig = create_secrets_diagram()
    fig.savefig(f'{OUTPUT_DIR}/{FILENAME}.png', dpi=150, bbox_inches='tight',
               facecolor=COLOR_BG)
    plt.close(fig)
    print(f"✅ Created {OUTPUT_DIR}/{FILENAME}.png")
    
    # Animated GIF
    create_animation()
