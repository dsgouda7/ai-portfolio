"""
Generate mixed precision training comparison animation for Ch.10.

Visualizes:
1. FP32 vs FP16 data flow
2. Training speedup
3. Numerical stability (gradient scaling)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from PIL import Image
import io

# Dark theme
plt.style.use('dark_background')
DARK_BG = '#1a1a2e'

def create_mixed_precision_frames(num_frames=60):
    """Generate frames comparing FP32 and FP16 training."""
    frames = []
    
    for frame_idx in range(num_frames):
        fig = plt.figure(figsize=(16, 10), facecolor=DARK_BG)
        fig.patch.set_facecolor(DARK_BG)
        
        # Progress (show both training paradigms side by side, then compare)
        progress = frame_idx / (num_frames - 1)
        
        # Create layout
        gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
        
        # Title
        fig.suptitle(
                'Mixed Precision Training (FP16 + FP32) — 2× Speedup',
                fontsize=20, color='white', weight='bold', y=0.98
        )
        
        # Plot 1: FP32 Training Flow (left)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis('off')
        ax1.set_title('Standard Training (FP32)', fontsize=14, color='white', weight='bold', pad=10)
        
        # Draw FP32 pipeline
        y_pos = 0.9
        step_height = 0.15
        
        steps_fp32 = [
                ('Input (FP32)', '#1e3a8a'),
                ('Forward Pass (FP32)', '#1e3a8a'),
                ('Loss (FP32)', '#1e3a8a'),
                ('Backward Pass (FP32)', '#1e3a8a'),
                ('Update Weights (FP32)', '#1e3a8a'),
        ]
        
        for i, (label, color) in enumerate(steps_fp32):
                y = y_pos - i * step_height
                box = FancyBboxPatch((0.1, y - 0.05), 0.8, 0.08,
                                    boxstyle="round,pad=0.01", 
                                    facecolor=color, edgecolor='white', linewidth=2, alpha=0.8)
                ax1.add_patch(box)
                ax1.text(0.5, y, label, ha='center', va='center', 
                        fontsize=11, color='white', weight='bold')
                
                # Arrow to next step
                if i < len(steps_fp32) - 1:
                    ax1.arrow(0.5, y - 0.05, 0, -0.04, head_width=0.05, head_length=0.02,
                             fc='white', ec='white', linewidth=2)
        
        ax1.text(0.5, 0.05, 'All operations in 32-bit precision\n(Slower, more memory)',
                    ha='center', va='bottom', fontsize=10, color='white', style='italic')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        
        # Plot 2: FP16+FP32 Training Flow (right)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.axis('off')
        ax2.set_title('Mixed Precision (FP16 + FP32)', fontsize=14, color='white', weight='bold', pad=10)
        
        steps_amp = [
                ('Input (FP32)', '#1e3a8a'),
                ('Forward Pass (FP16)', '#15803d'),  # Green = FP16
                ('Loss (FP32)', '#1e3a8a'),
                ('Backward Pass (FP16, scaled)', '#b45309'),  # Orange = scaled
                ('Update Weights (FP32)', '#1e3a8a'),
        ]
        
        for i, (label, color) in enumerate(steps_amp):
                y = y_pos - i * step_height
                box = FancyBboxPatch((0.1, y - 0.05), 0.8, 0.08,
                                    boxstyle="round,pad=0.01",
                                    facecolor=color, edgecolor='white', linewidth=2, alpha=0.8)
                ax2.add_patch(box)
                ax2.text(0.5, y, label, ha='center', va='center',
                        fontsize=11, color='white', weight='bold')
                
                if i < len(steps_amp) - 1:
                    ax2.arrow(0.5, y - 0.05, 0, -0.04, head_width=0.05, head_length=0.02,
                             fc='white', ec='white', linewidth=2)
        
        ax2.text(0.5, 0.05, 'Mix of 16-bit (fast) and 32-bit (stable)\n2× speedup on Tensor Cores',
                    ha='center', va='bottom', fontsize=10, color='white', style='italic')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        
        # Plot 3: Training speed comparison
        ax3 = fig.add_subplot(gs[1, :])
        
        # Simulated training progress
        epochs = np.arange(int(progress * 10) + 1)
        
        # FP32: 4 hours for 10 epochs (24 min/epoch)
        time_fp32 = epochs * 24  # minutes
        
        # FP16: 2 hours for 10 epochs (12 min/epoch) — 2× faster
        time_amp = epochs * 12  # minutes
        
        ax3.plot(epochs, time_fp32, marker='o', color='#1e3a8a', linewidth=3, 
                    markersize=8, label='FP32 (Standard)', linestyle='--')
        ax3.plot(epochs, time_amp, marker='s', color='#15803d', linewidth=3,
                    markersize=8, label='FP16 + FP32 (Mixed Precision)')
        
        ax3.set_title('Training Time Comparison (10 Epochs)', fontsize=14, color='white', weight='bold')
        ax3.set_xlabel('Epoch', color='white', fontsize=12)
        ax3.set_ylabel('Cumulative Time (minutes)', color='white', fontsize=12)
        ax3.set_xlim(0, 10)
        ax3.set_ylim(0, 250)
        ax3.tick_params(colors='white')
        ax3.legend(loc='upper left', fontsize=11)
        ax3.grid(alpha=0.3)
        
        # Add speedup annotation
        if len(epochs) > 5:
                speedup = time_fp32[-1] / time_amp[-1] if time_amp[-1] > 0 else 0
                ax3.text(0.7, 0.8, f'Speedup: {speedup:.1f}×', transform=ax3.transAxes,
                        fontsize=14, color='#15803d', weight='bold',
                        bbox=dict(boxstyle='round', facecolor='#15803d', alpha=0.2, edgecolor='white', linewidth=2))
        
        # Plot 4: Gradient scaling visualization
        ax4 = fig.add_subplot(gs[2, 0])
        
        # Show gradient underflow problem and scaling solution
        gradients = np.array([1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3])  # Small gradients
        scale_factor = 1024
        
        # FP16 range
        fp16_min = 6e-5
        
        x = np.arange(len(gradients))
        width = 0.35
        
        # Unscaled (many underflow)
        underflow_mask = gradients < fp16_min
        colors_unscaled = ['#b91c1c' if u else '#1e3a8a' for u in underflow_mask]
        bars1 = ax4.bar(x - width/2, gradients, width, label='Unscaled (FP16)',
                           color=colors_unscaled, alpha=0.8, edgecolor='white')
        
        # Scaled (preserved)
        scaled_grads = np.minimum(gradients * scale_factor, 1e-2)  # Cap for visualization
        bars2 = ax4.bar(x + width/2, scaled_grads, width, label=f'Scaled (×{scale_factor})',
                           color='#15803d', alpha=0.8, edgecolor='white')
        
        ax4.axhline(y=fp16_min, color='red', linestyle='--', linewidth=2, label='FP16 Underflow')
        ax4.set_title('Gradient Scaling Prevents Underflow', fontsize=12, color='white', weight='bold')
        ax4.set_ylabel('Gradient Magnitude', color='white')
        ax4.set_yscale('log')
        ax4.set_xticks(x)
        ax4.set_xticklabels([f'G{i+1}' for i in range(len(gradients))], color='white')
        ax4.tick_params(colors='white')
        ax4.legend(fontsize=9)
        ax4.grid(alpha=0.3, which='both')
        
        # Plot 5: Memory usage comparison
        ax5 = fig.add_subplot(gs[2, 1])
        
        memory_data = [
                ('Activations\n(FP32)', 8.0, '#1e3a8a'),
                ('Activations\n(FP16)', 4.0, '#15803d'),
                ('Gradients\n(FP32)', 8.0, '#1e3a8a'),
                ('Gradients\n(FP16)', 4.0, '#15803d'),
                ('Weights\n(FP32)', 8.0, '#1e3a8a'),
                ('Weights\n(FP32 + FP16)', 8.0, '#b45309'),  # Master + model copy
        ]
        
        mem_names = [d[0] for d in memory_data]
        mem_vals = [d[1] for d in memory_data]
        mem_colors = [d[2] for d in memory_data]
        
        bars5 = ax5.barh(mem_names, mem_vals, color=mem_colors, alpha=0.8, edgecolor='white', linewidth=2)
        ax5.set_title('Memory Usage (GB)', fontsize=12, color='white', weight='bold')
        ax5.set_xlabel('Memory (GB)', color='white')
        ax5.set_xlim(0, 10)
        ax5.tick_params(colors='white')
        ax5.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars5, mem_vals):
                ax5.text(val + 0.2, bar.get_y() + bar.get_height()/2., f'{val:.1f} GB',
                        va='center', ha='left', color='white', fontsize=10, weight='bold')
        
        # Summary text
        total_fp32 = 8.0 + 8.0 + 8.0  # Activations + Gradients + Weights
        total_amp = 4.0 + 4.0 + 8.0   # FP16 activations/gradients, FP32 weights
        savings = ((total_fp32 - total_amp) / total_fp32) * 100
        
        fig.text(0.5, 0.02, 
                    f'Mixed Precision: 2× faster training, {savings:.0f}% memory reduction, <0.5% accuracy impact',
                    ha='center', va='bottom', fontsize=12, color='#15803d', weight='bold',
                    bbox=dict(boxstyle='round', facecolor='#15803d', alpha=0.2, edgecolor='white', linewidth=2))
        
        # Save frame
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, facecolor=DARK_BG, bbox_inches='tight')
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()
        plt.close(fig)
        
        return frames
    
    if __name__ == '__main__':
        print("Generating mixed precision training animation...")
        frames = create_mixed_precision_frames(num_frames=60)
        
        # Save as GIF
        frames[0].save(
        '../img/ch10-mixed-precision.gif',
        save_all=True,
        append_images=frames[1:],
        duration=120,  # ms per frame
        loop=0
        )
        
        # Save key frames
        frames[0].save('../img/ch10-mixed-precision-start.png')
        frames[30].save('../img/ch10-mixed-precision-mid.png')
        frames[-1].save('../img/ch10-mixed-precision-final.png')
        
        print("✅ Animation saved: ch10-mixed-precision.gif")
        print("✅ Key frames saved: start, mid, final .png")
    
