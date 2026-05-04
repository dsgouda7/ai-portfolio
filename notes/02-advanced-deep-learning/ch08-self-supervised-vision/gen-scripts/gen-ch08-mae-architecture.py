"""
Generate MAE architecture diagram showing masking and reconstruction.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Dark theme setup
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#1a1a2e'
plt.rcParams['axes.facecolor'] = '#1a1a2e'

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Patch grid parameters
grid_size = 14  # 14×14 patches
patch_size = 0.8

# Color scheme
visible_color = '#15803d'  # Green
masked_color = '#1a1a2e'   # Dark (masked)
reconstructed_color = '#3b82f6'  # Blue

# Generate random mask (75% masked)
np.random.seed(42)
num_patches = grid_size * grid_size
num_masked = int(0.75 * num_patches)
mask = np.zeros(num_patches, dtype=bool)
mask[np.random.choice(num_patches, num_masked, replace=False)] = True
mask = mask.reshape(grid_size, grid_size)

# Panel 1: Original image (as patches)
ax = axes[0]
for i in range(grid_size):
    for j in range(grid_size):
        color = plt.cm.viridis(np.random.rand())  # Random color per patch
        rect = mpatches.Rectangle((j, grid_size - i - 1), patch_size, patch_size,
                                  facecolor=color, edgecolor='white', linewidth=0.5)
        ax.add_patch(rect)

ax.set_xlim(0, grid_size)
ax.set_ylim(0, grid_size)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('1. Input Image\n(196 patches, 14×14)', fontsize=13, weight='bold')

# Panel 2: Masked input (75% masked)
ax = axes[1]
for i in range(grid_size):
    for j in range(grid_size):
        if mask[i, j]:
            color = masked_color  # Masked (black)
        else:
            color = visible_color  # Visible (green)
        
        rect = mpatches.Rectangle((j, grid_size - i - 1), patch_size, patch_size,
                                  facecolor=color, edgecolor='white', linewidth=0.5)
        ax.add_patch(rect)

ax.set_xlim(0, grid_size)
ax.set_ylim(0, grid_size)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('2. Masked Input (75% masked)\nEncoder sees only green (49 patches)', 
            fontsize=13, weight='bold')

# Panel 3: Encoder output + mask tokens
ax = axes[2]
# Draw visible patches (encoded)
visible_patches = []
for i in range(grid_size):
    for j in range(grid_size):
        if not mask[i, j]:
            visible_patches.append((i, j))

# Draw only first 10 visible patches for clarity
for idx, (i, j) in enumerate(visible_patches[:10]):
    rect = mpatches.Rectangle((j, grid_size - i - 1), patch_size, patch_size,
                              facecolor=visible_color, edgecolor='white', linewidth=0.5)
    ax.add_patch(rect)

# Add text: "Encoder → 49 latent tokens"
ax.text(grid_size/2, grid_size/2, 'Encoder\n(ViT, 12 layers)\n↓\n49 latent tokens',
       ha='center', va='center', fontsize=12, weight='bold',
       bbox=dict(boxstyle='round,pad=1', facecolor='#1e3a8a', alpha=0.9),
       color='white')

ax.set_xlim(0, grid_size)
ax.set_ylim(0, grid_size)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('3. Encoder Output\n(Lightweight: only 25% of patches)', 
            fontsize=13, weight='bold')

# Panel 4: Reconstructed output
ax = axes[3]
for i in range(grid_size):
    for j in range(grid_size):
        if mask[i, j]:
            color = reconstructed_color  # Reconstructed (blue)
        else:
            color = visible_color  # Original visible (green)
        
        rect = mpatches.Rectangle((j, grid_size - i - 1), patch_size, patch_size,
                                  facecolor=color, edgecolor='white', linewidth=0.5)
        ax.add_patch(rect)

ax.set_xlim(0, grid_size)
ax.set_ylim(0, grid_size)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('4. Decoder Reconstruction\nBlue = predicted, Green = original', 
            fontsize=13, weight='bold')

# Overall title
fig.suptitle('MAE (Masked Autoencoder): 75% Masking Ratio', 
            fontsize=16, weight='bold')

# Legend
legend_elements = [
    mpatches.Patch(facecolor=visible_color, label='Visible patches (25%)'),
    mpatches.Patch(facecolor=masked_color, label='Masked patches (75%)'),
    mpatches.Patch(facecolor=reconstructed_color, label='Reconstructed patches')
]
fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
          fontsize=12, framealpha=0.9, bbox_to_anchor=(0.5, -0.05))

# Add architecture flow text
flow_text = ("MAE Architecture: Patchify → Random Mask (75%) → "
            "Encoder (49 patches) → Decoder (all 196) → Reconstruct → Loss (masked only)")
fig.text(0.5, 0.02, flow_text, ha='center', fontsize=11,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e', alpha=0.8),
        color='white')

plt.tight_layout()
plt.savefig('../img/ch08-mae-architecture.png', dpi=150, 
           bbox_inches='tight', facecolor='#1a1a2e')
print("Saved: ch08-mae-architecture.png")
plt.close()
