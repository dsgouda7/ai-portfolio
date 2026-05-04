"""
Generate throughput needle animation for Ch.6 Model Serving Frameworks.

Shows progression: 10 req/s → 150 req/s with vLLM
Animated GIF showing needle moving on speedometer gauge.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Wedge, Circle, FancyArrowPatch
import numpy as np
from PIL import Image
import io

def create_speedometer_frame(current_value, max_value=200, title="Throughput"):
    """Create a single speedometer frame"""
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.3, 1.2)
    ax.axis('off')
    
    # Draw gauge arc
    theta = np.linspace(0, np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    
    # Color segments
    segments = [
        (0, 50, '#b91c1c'),      # Red: 0-50 (slow)
        (50, 100, '#b45309'),    # Orange: 50-100 (medium)
        (100, 150, '#15803d'),   # Green: 100-150 (fast)
        (150, 200, '#1d4ed8'),   # Blue: 150-200 (very fast)
    ]
    
    for start, end, color in segments:
        start_angle = (1 - start / max_value) * 180
        end_angle = (1 - end / max_value) * 180
        wedge = Wedge((0, 0), 1.0, end_angle, start_angle,
                     width=0.15, facecolor=color, edgecolor='white', linewidth=2)
        ax.add_patch(wedge)
    
    # Draw tick marks
    for value in [0, 50, 100, 150, 200]:
        angle = (1 - value / max_value) * np.pi
        x_inner = 0.85 * np.cos(angle)
        y_inner = 0.85 * np.sin(angle)
        x_outer = 1.05 * np.cos(angle)
        y_outer = 1.05 * np.sin(angle)
        ax.plot([x_inner, x_outer], [y_inner, y_outer], 'white', linewidth=2)
        
        # Labels
        x_label = 1.15 * np.cos(angle)
        y_label = 1.15 * np.sin(angle)
        ax.text(x_label, y_label, str(value), ha='center', va='center',
               fontsize=12, fontweight='bold', color='white')
    
    # Draw needle
    needle_angle = (1 - current_value / max_value) * np.pi
    needle_x = 0.75 * np.cos(needle_angle)
    needle_y = 0.75 * np.sin(needle_angle)
    
    ax.plot([0, needle_x], [0, needle_y], color='white', linewidth=4)
    ax.plot([0, needle_x], [0, needle_y], color='#fbbf24', linewidth=2.5)
    
    # Center circle
    center = Circle((0, 0), 0.08, facecolor='white', edgecolor='black', linewidth=2, zorder=10)
    ax.add_patch(center)
    
    # Display current value
    ax.text(0, -0.15, f'{current_value:.0f} req/s', ha='center', va='top',
           fontsize=24, fontweight='bold', color='white',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#0f172a', 
                    edgecolor='white', linewidth=2))
    
    # Title
    ax.text(0, 1.1, title, ha='center', va='bottom',
           fontsize=18, fontweight='bold', color='white')
    
    # Labels
    ax.text(-0.95, -0.05, 'HuggingFace\nBaseline', ha='center', va='center',
           fontsize=9, color='#b91c1c', fontweight='bold')
    ax.text(0.95, -0.05, 'vLLM with\nPagedAttention', ha='center', va='center',
           fontsize=9, color='#1d4ed8', fontweight='bold')
    
    return fig

def generate_animation():
    """Generate animated GIF"""
    # Animation sequence
    frames = []
    
    # Phase 1: Baseline (10 req/s) - hold for 1 second
    for _ in range(10):
        fig = create_speedometer_frame(10, title="Model Serving Throughput")
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#1a1a2e')
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()
        plt.close(fig)
    
    # Phase 2: Transition to vLLM (10 → 150 req/s) - smooth acceleration
    transition_values = np.concatenate([
        np.linspace(10, 50, 15),   # Accelerate
        np.linspace(50, 150, 25),  # Continue accelerating
    ])
    
    for value in transition_values:
        fig = create_speedometer_frame(value, title="Model Serving Throughput")
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#1a1a2e')
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()
        plt.close(fig)
    
    # Phase 3: vLLM steady state (150 req/s) - hold for 1.5 seconds
    for _ in range(15):
        fig = create_speedometer_frame(150, title="Model Serving Throughput")
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#1a1a2e')
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()
        plt.close(fig)
    
    # Save as animated GIF
    frames[0].save(
        '../img/ch06-serving-throughput-needle.gif',
        save_all=True,
        append_images=frames[1:],
        duration=100,  # 100ms per frame
        loop=0
    )
    
    print("✅ Generated: ../img/ch06-serving-throughput-needle.gif")
    print(f"   Frames: {len(frames)}")
    print(f"   Duration: {len(frames) * 0.1:.1f}s")

if __name__ == "__main__":
    # Check if PIL is available
    try:
        generate_animation()
    except ImportError:
        print("⚠️  Pillow (PIL) not installed. Installing...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'pillow'])
        generate_animation()
