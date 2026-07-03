"""
Generate DINO attention map visualization animation.
Shows how attention heads emerge to focus on object boundaries.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.patches as mpatches

# Dark theme setup
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#1a1a2e'
plt.rcParams['axes.facecolor'] = '#1a1a2e'

# Animation parameters
num_frames = 60
grid_size = 14  # 14×14 patches for 224×224 image

# Simulate attention map evolution during training
np.random.seed(42)

def generate_attention_map(epoch_frac):
    """Generate attention map that progressively focuses on object centers"""
    # Create base attention (uniform at start, peaked at end)
    x = np.linspace(-3, 3, grid_size)
    y = np.linspace(-3, 3, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Three object centers (simulate 3 products on shelf)
    centers = [(-1.5, 0.5), (0.5, 0.8), (1.8, -0.3)]
    
    attention = np.zeros((grid_size, grid_size))
    
    for cx, cy in centers:
        # Gaussian centered on object
        sigma = 3.0 - 2.5 * epoch_frac  # Start wide, end narrow
        gaussian = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
        attention += gaussian
    
    # Add noise (decreases over training)
    noise_level = 0.5 * (1 - epoch_frac)
    attention += np.random.rand(grid_size, grid_size) * noise_level
    
    # Normalize
    attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
    
    return attention

def create_frame(epoch_frac):
    """Create a single frame showing DINO attention evolution"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Left: Simulated input image (retail shelf)
    ax = axes[0]
    # Create synthetic shelf image (3 products)
    shelf_img = np.ones((224, 224, 3)) * 0.2  # Dark background
    
    # Draw 3 products (rectangles with colors)
    products = [
        (40, 80, 80, 120, [0.2, 0.5, 0.8]),   # Blue product
        (90, 130, 80, 120, [0.8, 0.3, 0.2]),  # Red product
        (140, 180, 80, 120, [0.3, 0.8, 0.3])  # Green product
    ]
    
    for x1, x2, y1, y2, color in products:
        shelf_img[y1:y2, x1:x2] = color
    
    ax.imshow(shelf_img)
    ax.set_title('Input: Retail Shelf (3 Products)', fontsize=13, weight='bold')
    ax.axis('off')
    
    # Middle: Attention map evolution
    ax = axes[1]
    attention = generate_attention_map(epoch_frac)
    
    im = ax.imshow(attention, cmap='hot', interpolation='bilinear')
    ax.set_title(f'DINO Attention Map (Epoch {epoch_frac*800:.0f}/800)', 
                fontsize=13, weight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Right: Overlay on input
    ax = axes[2]
    ax.imshow(shelf_img)
    
    # Resize attention to match image size
    from scipy.ndimage import zoom
    attention_resized = zoom(attention, (224/grid_size, 224/grid_size), order=1)
    
    overlay = ax.imshow(attention_resized, cmap='hot', alpha=0.6, interpolation='bilinear')
    ax.set_title('Attention Overlay: Emergent Object Focus', fontsize=13, weight='bold')
    ax.axis('off')
    
    # Add epoch and insight text
    fig.suptitle('DINO Self-Supervised Learning: Attention Maps Emerge Automatically', 
                fontsize=16, weight='bold', y=0.98)
    
    insight_text = f"Training Progress: {epoch_frac*100:.0f}%\n"
    if epoch_frac < 0.3:
        insight_text += "Early: Attention is diffuse (model exploring)"
    elif epoch_frac < 0.7:
        insight_text += "Mid: Attention focusing on high-contrast regions"
    else:
        insight_text += "Late: Attention peaks on object centers (no labels!)"
    
    fig.text(0.5, 0.02, insight_text, ha='center', fontsize=12, 
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#1e3a8a', alpha=0.9),
            color='white', weight='bold')
    
    return fig

# Generate animation
print("Generating DINO attention animation...")

frames = []
for i in range(num_frames):
    epoch_frac = i / (num_frames - 1)
    fig = create_frame(epoch_frac)
    
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
fig_anim = plt.figure(figsize=(18, 6))
im = plt.imshow(frames[0])
plt.axis('off')

def update(frame_idx):
    im.set_array(frames[frame_idx])
    return [im]

anim = FuncAnimation(fig_anim, update, frames=len(frames), interval=100, blit=True)
anim.save('../img/ch08-dino-attention.gif', writer=PillowWriter(fps=10))
plt.close()

# Save final frame
fig = create_frame(1.0)
plt.savefig('../img/ch08-dino-attention-final.png', dpi=150, 
           bbox_inches='tight', facecolor='#1a1a2e')
plt.close()

print("Animation saved: ch08-dino-attention.gif")
print("Final frame saved: ch08-dino-attention-final.png")
