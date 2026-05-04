"""
Generate contrastive learning positive/negative pairs animation.
Shows how two augmented views of the same image are pulled together,
while views from different images are pushed apart in embedding space.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.patches as mpatches

# Dark theme setup
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#1a1a2e'
plt.rcParams['axes.facecolor'] = '#1a1a2e'

# Animation parameters
num_frames = 60
num_epochs = 5

# Initialize embedding space (2D projection of 128-d space)
np.random.seed(42)

# 4 images, each with 2 augmented views
# Start with random positions
image_colors = ['#3b82f6', '#15803d', '#b45309', '#b91c1c']  # Blue, Green, Orange, Red
num_images = 4

# Initial positions (random)
positions = np.random.randn(num_images, 2, 2) * 3  # [num_images, 2_views, 2_dims]

# Target positions (positive pairs close, negatives far)
targets = np.array([
    [[-3, 2], [-2.8, 2.1]],   # Image 1: blue (top-left cluster)
    [[2, 2.5], [2.2, 2.3]],    # Image 2: green (top-right cluster)
    [[-2.5, -2], [-2.7, -1.9]], # Image 3: orange (bottom-left cluster)
    [[2.5, -2.2], [2.3, -2.4]]  # Image 4: red (bottom-right cluster)
])

def create_frame(epoch_frac):
    """Create a single frame of the animation"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Interpolate positions
    t = epoch_frac / num_epochs
    t_smooth = 1 - (1 - t) ** 3  # Ease-out cubic
    current_pos = positions + (targets - positions) * t_smooth
    
    # Plot embeddings
    for img_idx in range(num_images):
        color = image_colors[img_idx]
        
        # Plot two views
        view1 = current_pos[img_idx, 0]
        view2 = current_pos[img_idx, 1]
        
        # Draw views as circles
        circle1 = Circle(view1, 0.15, color=color, alpha=0.8, zorder=3)
        circle2 = Circle(view2, 0.15, color=color, alpha=0.8, zorder=3)
        ax.add_patch(circle1)
        ax.add_patch(circle2)
        
        # Draw positive pair connection (green line)
        if t_smooth > 0.2:
            ax.plot([view1[0], view2[0]], [view1[1], view2[1]], 
                   color='#15803d', linewidth=2, alpha=0.6, linestyle='--', zorder=1)
        
        # Add labels
        ax.text(view1[0], view1[1] + 0.4, f'Img{img_idx+1}_View1', 
               ha='center', fontsize=9, color='white', weight='bold')
    
    # Draw negative pair repulsions (red dashed lines between different images)
    if t_smooth > 0.5:
        # Just show a few representative negative connections to avoid clutter
        negative_pairs = [(0, 1), (1, 3), (0, 2), (2, 3)]
        for i, j in negative_pairs:
            pos_i = current_pos[i, 0]
            pos_j = current_pos[j, 0]
            
            # Red repulsion line
            mid_x = (pos_i[0] + pos_j[0]) / 2
            mid_y = (pos_i[1] + pos_j[1]) / 2
            
            ax.plot([pos_i[0], mid_x], [pos_i[1], mid_y], 
                   color='#b91c1c', linewidth=1, alpha=0.3, linestyle=':', zorder=0)
    
    # Title and annotations
    epoch_label = f"Training Progress: Epoch {epoch_frac:.1f}/{num_epochs}"
    ax.text(0, 4.2, epoch_label, ha='center', fontsize=14, 
           weight='bold', color='white')
    
    # Loss annotation
    current_loss = 2.5 * np.exp(-epoch_frac * 0.8)  # Decay from ~2.5 to ~0.5
    ax.text(0, -4.5, f"Contrastive Loss: {current_loss:.3f}", 
           ha='center', fontsize=12, color='#3b82f6')
    
    # Legend
    positive_patch = mpatches.Patch(color='#15803d', label='Positive Pairs (pulled together)')
    negative_patch = mpatches.Patch(color='#b91c1c', label='Negative Pairs (pushed apart)')
    ax.legend(handles=[positive_patch, negative_patch], 
             loc='upper right', fontsize=10, framealpha=0.9)
    
    # Formatting
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel('Embedding Dimension 1', fontsize=11)
    ax.set_ylabel('Embedding Dimension 2', fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.set_title('SimCLR Contrastive Learning: Embedding Space Evolution', 
                fontsize=15, weight='bold', pad=20)
    
    return fig, ax

# Generate animation
print("Generating contrastive pairs animation...")

frames = []
for i in range(num_frames):
    epoch_frac = (i / num_frames) * num_epochs
    fig, ax = create_frame(epoch_frac)
    
    # Save frame
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:,:,:3]  # RGBA to RGB
    frames.append(frame)
    plt.close(fig)
    
    if (i + 1) % 10 == 0:
        print(f"  Frame {i+1}/{num_frames}")

# Save as GIF
print("Saving animation...")
fig_anim = plt.figure(figsize=(12, 10))
im = plt.imshow(frames[0])
plt.axis('off')

def update(frame_idx):
    im.set_array(frames[frame_idx])
    return [im]

anim = FuncAnimation(fig_anim, update, frames=len(frames), interval=100, blit=True)
anim.save('../img/ch07-contrastive-pairs.gif', writer=PillowWriter(fps=10))
plt.close()

# Save final frame as static PNG
fig, ax = create_frame(num_epochs)
plt.savefig('../img/ch07-contrastive-pairs-final.png', dpi=150, 
           bbox_inches='tight', facecolor='#1a1a2e')
plt.close()

print("Animation saved: ch07-contrastive-pairs.gif")
print("Final frame saved: ch07-contrastive-pairs-final.png")
