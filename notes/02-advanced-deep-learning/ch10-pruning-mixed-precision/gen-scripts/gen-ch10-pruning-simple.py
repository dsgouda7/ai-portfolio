"""
Generate simplified pruning visualization for Ch.10.
Shows before/after pruning with weight distribution.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

plt.style.use('dark_background')
DARK_BG = '#1a1a2e'

def create_pruning_visualization():
    """Create a single static image showing pruning process."""
    fig = plt.figure(figsize=(16, 8), facecolor=DARK_BG)
    fig.patch.set_facecolor(DARK_BG)
    
    # Create 3 subplots: Dense -> Pruned -> Fine-tuned
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)
    
    fig.suptitle('Iterative Magnitude Pruning: 80% Sparsity', 
                 fontsize=18, color='white', weight='bold', y=0.98)
    
    # Generate synthetic weight matrices
    np.random.seed(42)
    matrix_size = 32
    original_weights = np.random.randn(matrix_size, matrix_size) * 0.3
    
    # Stage 1: Dense network
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(original_weights, cmap='RdYlGn', vmin=-0.5, vmax=0.5, aspect='auto')
    ax1.set_title('1. Dense Network\n10.7 MB, 83.2% mAP', fontsize=12, color='white', weight='bold')
    ax1.set_xlabel('0% sparsity', fontsize=10, color='#aaa')
    ax1.set_xticks([])
    ax1.set_yticks([])
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Stage 2: Pruned (80% zeros)
    ax2 = fig.add_subplot(gs[0, 1])
    threshold = np.percentile(np.abs(original_weights), 80)
    mask = np.abs(original_weights) > threshold
    pruned_weights = original_weights * mask
    im2 = ax2.imshow(pruned_weights, cmap='RdYlGn', vmin=-0.5, vmax=0.5, aspect='auto')
    ax2.set_title('2. After Pruning\n7.2 MB, 80.8% mAP ⚠️', fontsize=12, color='white', weight='bold')
    ax2.set_xlabel('80% sparsity (before fine-tuning)', fontsize=10, color='#aaa')
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Stage 3: Fine-tuned
    ax3 = fig.add_subplot(gs[0, 2])
    # Slightly adjust remaining weights (simulate fine-tuning)
    finetuned_weights = pruned_weights + (np.random.randn(matrix_size, matrix_size) * 0.05 * mask)
    im3 = ax3.imshow(finetuned_weights, cmap='RdYlGn', vmin=-0.5, vmax=0.5, aspect='auto')
    ax3.set_title('3. After Fine-Tuning ✅\n6.8 MB, 82.1% mAP', fontsize=12, color='white', weight='bold')
    ax3.set_xlabel('80% sparsity (recovered accuracy)', fontsize=10, color='#aaa')
    ax3.set_xticks([])
    ax3.set_yticks([])
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # Bottom row: Weight magnitude histograms
    ax4 = fig.add_subplot(gs[1, :])
    
    # Plot weight distributions
    weights_flat = original_weights.flatten()
    pruned_flat = pruned_weights.flatten()
    finetuned_flat = finetuned_weights.flatten()
    
    ax4.hist(weights_flat, bins=50, alpha=0.6, label='Dense', color='#3b82f6', edgecolor='white', linewidth=0.5)
    ax4.hist(pruned_flat[pruned_flat != 0], bins=30, alpha=0.7, label='Pruned (non-zero)', color='#f59e0b', edgecolor='white', linewidth=0.5)
    ax4.hist(finetuned_flat[finetuned_flat != 0], bins=30, alpha=0.7, label='Fine-tuned (non-zero)', color='#10b981', edgecolor='white', linewidth=0.5)
    
    ax4.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Pruning threshold: {threshold:.3f}')
    ax4.axvline(-threshold, color='red', linestyle='--', linewidth=2)
    
    ax4.set_xlabel('Weight Magnitude', fontsize=12, color='white')
    ax4.set_ylabel('Frequency', fontsize=12, color='white')
    ax4.set_title('Weight Distribution: Magnitude-Based Pruning', fontsize=12, color='white', weight='bold')
    ax4.legend(fontsize=10, loc='upper right')
    ax4.grid(alpha=0.2)
    
    # Add annotation
    ax4.text(0.05, 0.95, '80% of weights with smallest magnitude → 0\nRemaining 20% → fine-tuned',
             transform=ax4.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='#2d3748', alpha=0.8, edgecolor='white', linewidth=1),
             color='white')
    
    plt.tight_layout()
    
    # Save
    output_path = Path(__file__).parent.parent / 'img' / 'ch10-pruning-process.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    print(f"✅ Generated: {output_path}")
    plt.close()

if __name__ == '__main__':
    create_pruning_visualization()
