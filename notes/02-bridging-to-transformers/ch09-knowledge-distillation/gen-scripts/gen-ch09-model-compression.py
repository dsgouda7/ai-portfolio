"""
Generate model compression comparison animation for Ch.9 - Knowledge Distillation.

Visualizes the compression achieved through distillation:
- Model size reduction
- Accuracy preservation
- Latency improvement
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from PIL import Image
import io

# Dark theme
plt.style.use('dark_background')
DARK_BG = '#1a1a2e'

def create_compression_frames(num_frames=60):
    """Generate frames showing progressive compression."""
    frames = []
    
    # Model specifications
    models = {
        'teacher': {'name': 'ResNet-50\n(Teacher)', 'size_mb': 97.0, 'map': 85.4, 'iou': 71.2, 'latency_ms': 78, 'color': '#1e3a8a'},
        'baseline': {'name': 'MobileNetV2\n(Baseline)', 'size_mb': 14.2, 'map': 78.1, 'iou': 64.8, 'latency_ms': 42, 'color': '#b45309'},
        'distilled': {'name': 'MobileNetV2\n(Distilled)', 'size_mb': 10.7, 'map': 83.2, 'iou': 68.9, 'latency_ms': 39, 'color': '#15803d'},
    }
    
    for frame_idx in range(num_frames):
        fig = plt.figure(figsize=(16, 10), facecolor=DARK_BG)
        fig.patch.set_facecolor(DARK_BG)
        
        # Progress (fade in distilled model)
        progress = min(frame_idx / 40, 1.0)
        
        # Create layout
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
        
        # Title
        fig.suptitle(
            'Knowledge Distillation: Compression Without Accuracy Loss',
            fontsize=20, color='white', weight='bold', y=0.98
        )
        
        # Plot 1: Model Size Comparison
        ax1 = fig.add_subplot(gs[0, 0])
        
        sizes = [models['teacher']['size_mb'], 
                models['baseline']['size_mb'], 
                models['distilled']['size_mb'] * progress]
        colors = [models['teacher']['color'], 
                 models['baseline']['color'], 
                 models['distilled']['color']]
        
        bars1 = ax1.barh(range(3), sizes, color=colors, alpha=0.9, edgecolor='white', linewidth=2)
        ax1.axvline(x=100, color='red', linestyle='--', linewidth=2, label='100 MB Target')
        ax1.set_title('Model Size (MB)', fontsize=14, color='white', weight='bold')
        ax1.set_xlabel('Size (MB)', color='white', fontsize=12)
        ax1.set_yticks(range(3))
        ax1.set_yticklabels(['Teacher', 'Baseline\nStudent', 'Distilled\nStudent'], fontsize=10, color='white')
        ax1.set_xlim(0, 110)
        ax1.tick_params(colors='white')
        ax1.legend(fontsize=9)
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars1, sizes)):
            if val > 0:
                ax1.text(val + 2, bar.get_y() + bar.get_height()/2., f'{val:.1f} MB',
                        va='center', ha='left', color='white', fontsize=11, weight='bold')
        
        # Plot 2: Detection Accuracy (mAP)
        ax2 = fig.add_subplot(gs[0, 1])
        
        maps = [models['teacher']['map'], 
               models['baseline']['map'], 
               models['baseline']['map'] + (models['distilled']['map'] - models['baseline']['map']) * progress]
        
        bars2 = ax2.barh(range(3), maps, color=colors, alpha=0.9, edgecolor='white', linewidth=2)
        ax2.axvline(x=85, color='red', linestyle='--', linewidth=2, label='85% Target')
        ax2.set_title('Detection Accuracy (mAP@0.5)', fontsize=14, color='white', weight='bold')
        ax2.set_xlabel('mAP (%)', color='white', fontsize=12)
        ax2.set_yticks(range(3))
        ax2.set_yticklabels(['Teacher', 'Baseline\nStudent', 'Distilled\nStudent'], fontsize=10, color='white')
        ax2.set_xlim(75, 90)
        ax2.tick_params(colors='white')
        ax2.legend(fontsize=9)
        ax2.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars2, maps)):
            ax2.text(val + 0.5, bar.get_y() + bar.get_height()/2., f'{val:.1f}%',
                    va='center', ha='left', color='white', fontsize=11, weight='bold')
        
        # Plot 3: Inference Latency
        ax3 = fig.add_subplot(gs[0, 2])
        
        latencies = [models['teacher']['latency_ms'], 
                    models['baseline']['latency_ms'], 
                    models['distilled']['latency_ms'] * progress if progress > 0 else models['baseline']['latency_ms']]
        
        bars3 = ax3.barh(range(3), latencies, color=colors, alpha=0.9, edgecolor='white', linewidth=2)
        ax3.axvline(x=50, color='red', linestyle='--', linewidth=2, label='50ms Target')
        ax3.set_title('Inference Latency (ms)', fontsize=14, color='white', weight='bold')
        ax3.set_xlabel('Latency (ms)', color='white', fontsize=12)
        ax3.set_yticks(range(3))
        ax3.set_yticklabels(['Teacher', 'Baseline\nStudent', 'Distilled\nStudent'], fontsize=10, color='white')
        ax3.set_xlim(0, 90)
        ax3.tick_params(colors='white')
        ax3.legend(fontsize=9)
        ax3.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars3, latencies)):
            if val > 0:
                ax3.text(val + 2, bar.get_y() + bar.get_height()/2., f'{val:.0f} ms',
                        va='center', ha='left', color='white', fontsize=11, weight='bold')
        
        # Plot 4: Accuracy vs Size Trade-off
        ax4 = fig.add_subplot(gs[1, :2])
        
        teacher_pt = (models['teacher']['size_mb'], models['teacher']['map'])
        baseline_pt = (models['baseline']['size_mb'], models['baseline']['map'])
        distilled_pt = (models['distilled']['size_mb'] * progress, 
                       models['baseline']['map'] + (models['distilled']['map'] - models['baseline']['map']) * progress)
        
        # Plot points
        ax4.scatter(*teacher_pt, s=300, color=models['teacher']['color'], 
                   edgecolor='white', linewidth=2, zorder=5, label='Teacher', alpha=0.9)
        ax4.scatter(*baseline_pt, s=300, color=models['baseline']['color'], 
                   edgecolor='white', linewidth=2, zorder=5, label='Baseline Student', alpha=0.9)
        
        if progress > 0.1:
            ax4.scatter(*distilled_pt, s=300, color=models['distilled']['color'], 
                       edgecolor='white', linewidth=2, zorder=5, label='Distilled Student', alpha=0.9)
            
            # Draw arrow from baseline to distilled
            if progress > 0.5:
                ax4.annotate('', xy=distilled_pt, xytext=baseline_pt,
                           arrowprops=dict(arrowstyle='->', color='white', lw=2, alpha=0.7))
        
        # Ideal region (top-left: small size, high accuracy)
        ax4.add_patch(Rectangle((0, 85), 15, 5, facecolor='#15803d', alpha=0.1, 
                                edgecolor='#15803d', linestyle='--', linewidth=2))
        ax4.text(7.5, 87, 'Ideal Zone\n(Small & Accurate)', ha='center', va='center',
                color='#15803d', fontsize=11, weight='bold')
        
        ax4.set_title('Accuracy vs Size Trade-off', fontsize=14, color='white', weight='bold')
        ax4.set_xlabel('Model Size (MB)', color='white', fontsize=12)
        ax4.set_ylabel('mAP@0.5 (%)', color='white', fontsize=12)
        ax4.set_xlim(0, 105)
        ax4.set_ylim(76, 88)
        ax4.tick_params(colors='white')
        ax4.legend(loc='lower right', fontsize=10)
        ax4.grid(alpha=0.3)
        
        # Plot 5: Compression metrics
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        
        # Calculate compression ratios
        size_compression = models['teacher']['size_mb'] / models['distilled']['size_mb']
        accuracy_retention = (models['distilled']['map'] / models['teacher']['map']) * 100
        speedup = models['teacher']['latency_ms'] / models['distilled']['latency_ms']
        
        metrics_text = f"""
        COMPRESSION METRICS
        {'='*30}
        
        Size Compression:    {size_compression:.1f}×
        (97 MB → 10.7 MB)
        
        Accuracy Retention:  {accuracy_retention:.1f}%
        (85.4% → 83.2% mAP)
        
        Speedup:             {speedup:.1f}×
        (78 ms → 39 ms)
        
        Accuracy Loss:       {models['teacher']['map'] - models['distilled']['map']:.1f}%
        (Only 2.2% drop!)
        
        vs Baseline:         +{models['distilled']['map'] - models['baseline']['map']:.1f}%
        (Distillation recovers 5.1%)
        """
        
        ax5.text(0.1, 0.5, metrics_text, fontsize=11, color='white', family='monospace',
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='#2d2d44', alpha=0.8, 
                         edgecolor='white', linewidth=2))
        
        # Plot 6: Progressive constraint satisfaction
        ax6 = fig.add_subplot(gs[2, :])
        
        constraint_names = ['#1\nAccuracy\n≥85% mAP', '#2\nSegmentation\n≥70% IoU', 
                           '#3\nLatency\n<50ms', '#4\nSize\n<100 MB', '#5\nData\n<1k labels']
        
        # Status for each constraint (teacher, baseline, distilled)
        teacher_scores = [1.0, 1.0, 0.0, 0.03, 1.0]  # Normalized 0-1
        baseline_scores = [0.4, 0.0, 1.0, 1.0, 1.0]
        distilled_scores = [0.9 * progress, 0.7 * progress, 1.0 * progress, 1.0 * progress, 1.0 * progress]
        
        x = np.arange(len(constraint_names))
        width = 0.25
        
        bars_t = ax6.bar(x - width, teacher_scores, width, label='Teacher', 
                        color=models['teacher']['color'], alpha=0.8, edgecolor='white')
        bars_b = ax6.bar(x, baseline_scores, width, label='Baseline Student', 
                        color=models['baseline']['color'], alpha=0.8, edgecolor='white')
        bars_d = ax6.bar(x + width, distilled_scores, width, label='Distilled Student', 
                        color=models['distilled']['color'], alpha=0.8, edgecolor='white')
        
        ax6.set_title('ProductionCV Constraint Satisfaction', fontsize=16, color='white', weight='bold')
        ax6.set_ylabel('Satisfaction Level', color='white', fontsize=12)
        ax6.set_xticks(x)
        ax6.set_xticklabels(constraint_names, fontsize=10, color='white')
        ax6.set_ylim(0, 1.2)
        ax6.tick_params(colors='white')
        ax6.legend(loc='upper right', fontsize=11)
        ax6.grid(axis='y', alpha=0.3)
        ax6.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.5)
        
        # Save frame
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, facecolor=DARK_BG, bbox_inches='tight')
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()
        plt.close(fig)
    
    return frames

if __name__ == '__main__':
    print("Generating model compression animation...")
    frames = create_compression_frames(num_frames=60)
    
    # Save as GIF
    frames[0].save(
        '../img/ch09-model-compression.gif',
        save_all=True,
        append_images=frames[1:],
        duration=100,  # ms per frame
        loop=0
    )
    
    # Save key frames
    frames[0].save('../img/ch09-compression-start.png')
    frames[30].save('../img/ch09-compression-mid.png')
    frames[-1].save('../img/ch09-compression-end.png')
    
    print("✅ Animation saved: ch09-model-compression.gif")
    print("✅ Key frames saved: start, mid, end .png")
