"""
Generate pruning process animation for Ch.10 - Pruning & Mixed Precision Training.

Visualizes:
1. Weight magnitude distribution before/after pruning
2. Iterative pruning process (gradual removal of weights)
3. Fine-tuning recovery of accuracy
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import io

# Dark theme
plt.style.use('dark_background')
DARK_BG = '#1a1a2e'

def create_pruning_frames(num_frames=80):
    """Generate frames showing iterative pruning and fine-tuning."""
    frames = []
    
    # Pruning schedule: 0% → 40% → 60% → 80% (with fine-tuning between)
    # Frames 0-20: Prune to 40%
    # Frames 21-40: Fine-tune (accuracy recovery)
    # Frames 41-50: Prune to 60%
    # Frames 51-60: Fine-tune
    # Frames 61-70: Prune to 80%
    # Frames 71-80: Final fine-tune
    
    for frame_idx in range(num_frames):
        fig = plt.figure(figsize=(16, 10), facecolor=DARK_BG)
        fig.patch.set_facecolor(DARK_BG)
        
        # Determine current phase
        if frame_idx <= 20:
            # Phase 1: Initial pruning to 40%
            phase = "Phase 1: Pruning to 40%"
            sparsity = 40 * (frame_idx / 20)
            map_val = 83.2 - 1.2 * (frame_idx / 20)  # Drops during pruning
            iou_val = 68.9 - 1.5 * (frame_idx / 20)
        elif frame_idx <= 40:
            # Phase 2: Fine-tune (recover accuracy)
            phase = "Phase 2: Fine-Tuning"
            sparsity = 40
            progress = (frame_idx - 20) / 20
            map_val = 82.0 + 0.5 * progress  # Recover some accuracy
            iou_val = 67.4 + 1.2 * progress
        elif frame_idx <= 50:
            # Phase 3: Prune to 60%
            phase = "Phase 3: Pruning to 60%"
            progress = (frame_idx - 40) / 10
            sparsity = 40 + 20 * progress
            map_val = 82.5 - 0.8 * progress
            iou_val = 68.6 - 1.0 * progress
        elif frame_idx <= 60:
            # Phase 4: Fine-tune
            phase = "Phase 4: Fine-Tuning"
            sparsity = 60
            progress = (frame_idx - 50) / 10
            map_val = 81.7 + 0.6 * progress
            iou_val = 67.6 + 2.0 * progress
        elif frame_idx <= 70:
            # Phase 5: Prune to 80%
            phase = "Phase 5: Pruning to 80% (Final)"
            progress = (frame_idx - 60) / 10
            sparsity = 60 + 20 * progress
            map_val = 82.3 - 0.7 * progress
            iou_val = 69.6 - 0.5 * progress
        else:
            # Phase 6: Final fine-tune
            phase = "Phase 6: Final Fine-Tuning"
            sparsity = 80
            progress = (frame_idx - 70) / 10
            map_val = 81.6 + 0.5 * progress
            iou_val = 69.1 + 2.1 * progress
        
        # Create layout
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
        
        # Title
        fig.suptitle(
        f'Iterative Pruning Process — {phase}\nSparsity: {sparsity:.1f}%',
        fontsize=18, color='white', weight='bold', y=0.98
        )
        
        # Plot 1: Weight matrix visualization (sparse pattern)
        ax1 = fig.add_subplot(gs[0, :2])
        
        # Generate synthetic weight matrix (32x32)
        matrix_size = 32
        weight_matrix = np.random.randn(matrix_size, matrix_size) * 0.2
        
        # Apply sparsity mask (prune smallest magnitude weights)
        threshold = np.percentile(np.abs(weight_matrix), sparsity)
        mask = np.abs(weight_matrix) > threshold
        pruned_weights = weight_matrix * mask
        
        im = ax1.imshow(pruned_weights, cmap='RdYlGn', vmin=-0.5, vmax=0.5, 
                    interpolation='nearest', aspect='auto')
        ax1.set_title(f'Weight Matrix (Sample 32×32 Layer)\nGreen=Active, Black=Pruned',
                     fontsize=14, color='white', weight='bold')
        ax1.set_xlabel('Output Channel', color='white')
        ax1.set_ylabel('Input Channel', color='white')
        ax1.tick_params(colors='white')
        plt.colorbar(im, ax=ax1, label='Weight Value')
        
        # Add sparsity text
        actual_sparsity = 100 * (1 - mask.sum() / mask.size)
        ax1.text(0.02, 0.98, f'Sparsity: {actual_sparsity:.1f}%',
                transform=ax1.transAxes, fontsize=12, color='white', weight='bold',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        # Plot 2: Sparsity progress bar
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        
        # Vertical progress bar
        bar_height = 0.8
        bar_width = 0.4
        bar_x = 0.3
        bar_y = 0.1
        
        # Background (full)
        ax2.add_patch(Rectangle((bar_x, bar_y), bar_width, bar_height,
                                facecolor='#2d2d44', edgecolor='white', linewidth=2))
        
        # Progress (pruned)
        progress_height = bar_height * (sparsity / 100)
        color = '#15803d' if sparsity >= 80 else '#b45309' if sparsity >= 40 else '#1e3a8a'
        ax2.add_patch(Rectangle((bar_x, bar_y), bar_width, progress_height,
                                facecolor=color, edgecolor='white', linewidth=2, alpha=0.9))
        
        # Labels
        ax2.text(0.5, 0.95, 'Sparsity\nProgress', ha='center', va='top', 
                fontsize=12, color='white', weight='bold')
        ax2.text(0.5, bar_y + progress_height/2, f'{sparsity:.0f}%',
                ha='center', va='center', fontsize=16, color='white', weight='bold')
        ax2.text(0.5, 0.05, 'Target: 80%', ha='center', va='bottom',
                fontsize=10, color='white')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        
        # Plot 3: Accuracy curves (mAP and IoU)
        ax3 = fig.add_subplot(gs[1, :])
        
        # Simulated history
        history_length = frame_idx + 1
        map_history = []
        iou_history = []
        sparsity_history = []
        
        for i in range(history_length):
            if i <= 20:
                s = 40 * (i / 20)
                m = 83.2 - 1.2 * (i / 20)
                io = 68.9 - 1.5 * (i / 20)
            elif i <= 40:
                s = 40
                p = (i - 20) / 20
                m = 82.0 + 0.5 * p
                io = 67.4 + 1.2 * p
            elif i <= 50:
                p = (i - 40) / 10
                s = 40 + 20 * p
                m = 82.5 - 0.8 * p
                io = 68.6 - 1.0 * p
            elif i <= 60:
                s = 60
                p = (i - 50) / 10
                m = 81.7 + 0.6 * p
                io = 67.6 + 2.0 * p
            elif i <= 70:
                p = (i - 60) / 10
                s = 60 + 20 * p
                m = 82.3 - 0.7 * p
                io = 69.6 - 0.5 * p
            else:
                s = 80
                p = (i - 70) / 10
                m = 81.6 + 0.5 * p
                io = 69.1 + 2.1 * p
            
            map_history.append(m)
            iou_history.append(io)
            sparsity_history.append(s)
        
        x_axis = range(history_length)
        
        ax3_twin = ax3.twinx()
        
        line1 = ax3.plot(x_axis, map_history, color='#1e3a8a', linewidth=2, label='mAP@0.5 (%)')
        line2 = ax3.plot(x_axis, iou_history, color='#15803d', linewidth=2, label='IoU (%)')
        line3 = ax3_twin.plot(x_axis, sparsity_history, color='#b45309', linewidth=2, 
                              linestyle='--', label='Sparsity (%)')
        
        ax3.axhline(y=85, color='red', linestyle='--', linewidth=1, alpha=0.5, label='mAP Target (85%)')
        ax3.axhline(y=70, color='green', linestyle='--', linewidth=1, alpha=0.5, label='IoU Target (70%)')
        
        ax3.set_title('Accuracy vs Sparsity During Pruning', fontsize=14, color='white', weight='bold')
        ax3.set_xlabel('Iteration', color='white', fontsize=12)
        ax3.set_ylabel('Accuracy (%)', color='white', fontsize=12)
        ax3_twin.set_ylabel('Sparsity (%)', color='white', fontsize=12)
        ax3.set_xlim(0, 80)
        ax3.set_ylim(65, 90)
        ax3_twin.set_ylim(0, 100)
        ax3.tick_params(colors='white')
        ax3_twin.tick_params(colors='white')
        
        # Combined legend
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='lower left', fontsize=10)
        ax3.grid(alpha=0.3)
        
        # Plot 4: Model size reduction
        ax4 = fig.add_subplot(gs[2, 0])
        
        sizes = [
        ('Teacher\n(ResNet-50)', 97.0, '#1e3a8a'),
        ('Ch.9\n(Distilled)', 10.7, '#b45309'),
        ('Ch.10\n(Pruned)', 10.7 * (1 - sparsity/100 * 0.6), '#15803d'),  # Approx compression
        ]
        
        names = [s[0] for s in sizes]
        vals = [s[1] for s in sizes]
        cols = [s[2] for s in sizes]
        
        bars = ax4.bar(names, vals, color=cols, alpha=0.9, edgecolor='white', linewidth=2)
        ax4.axhline(y=100, color='red', linestyle='--', linewidth=2, label='100 MB Target')
        ax4.set_title('Model Size Evolution', fontsize=12, color='white', weight='bold')
        ax4.set_ylabel('Size (MB)', color='white')
        ax4.set_ylim(0, 110)
        ax4.tick_params(colors='white')
        ax4.legend(fontsize=9)
        ax4.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, vals):
            ax4.text(bar.get_x() + bar.get_width()/2., val + 2, f'{val:.1f}MB',
                        ha='center', va='bottom', color='white', fontsize=10, weight='bold')
        
        # Plot 5: Current metrics
        ax5 = fig.add_subplot(gs[2, 1:])
        ax5.axis('off')
        
        metrics_text = f"""
        CURRENT METRICS
        {'='*40}
        
        Sparsity:         {sparsity:.1f}%
        mAP@0.5:          {map_val:.1f}% {'✅' if map_val >= 82 else '⚠️'}
        IoU:              {iou_val:.1f}% {'✅' if iou_val >= 70 else '⚠️'}
        Model Size:       {10.7 * (1 - sparsity/100 * 0.6):.1f} MB
        
        Phase: {phase}
        
        {'='*40}
        Next: {'Complete!' if frame_idx >= 79 else 'Continue pruning/fine-tuning'}
        """
        
        ax5.text(0.1, 0.5, metrics_text, fontsize=11, color='white', family='monospace',
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='#2d2d44', alpha=0.8,
                         edgecolor='white', linewidth=2))
        
        # Save frame
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, facecolor=DARK_BG, bbox_inches='tight')
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()
        plt.close(fig)
        
        return frames
    
    if __name__ == '__main__':
        print("Generating pruning process animation...")
        frames = create_pruning_frames(num_frames=80)
        
        # Save as GIF
        frames[0].save(
        '../img/ch10-pruning-process.gif',
        save_all=True,
        append_images=frames[1:],
        duration=100,  # ms per frame
        loop=0
        )
        
        # Save key frames
        frames[0].save('../img/ch10-pruning-start.png')       # 0% sparsity
        frames[40].save('../img/ch10-pruning-phase2.png')     # 40% sparsity
        frames[60].save('../img/ch10-pruning-phase4.png')     # 60% sparsity
        frames[-1].save('../img/ch10-pruning-final.png')      # 80% sparsity
        
        print("✅ Animation saved: ch10-pruning-process.gif")
        print("✅ Key frames saved: start, phase2, phase4, final .png")
    
