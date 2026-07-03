"""
Generate distillation process animation for Ch.9 - Knowledge Distillation.

Visualizes:
1. Teacher model producing soft targets (temperature-scaled probabilities)
2. Student model learning from both soft targets and hard labels
3. Progressive convergence of student's predictions to teacher's knowledge
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image
import io

# Dark theme
plt.style.use('dark_background')
DARK_BG = '#1a1a2e'

def create_distillation_frames(num_frames=60):
    """Generate frames showing knowledge distillation process."""
    frames = []
    
    # Product classes
    classes = ['Soda\nCan', 'Water\nBottle', 'Juice\nBox', 'Cereal\nBox']
    
    # Teacher's soft targets (fixed)
    teacher_probs = np.array([0.65, 0.18, 0.12, 0.05])
    
    # Hard label (one-hot)
    hard_label = np.array([1.0, 0.0, 0.0, 0.0])
    
    for frame_idx in range(num_frames):
        fig = plt.figure(figsize=(14, 8), facecolor=DARK_BG)
        fig.patch.set_facecolor(DARK_BG)
        
        # Progress through training
        progress = frame_idx / (num_frames - 1)
        
        # Student starts with uniform distribution, converges to teacher
        initial_probs = np.array([0.25, 0.25, 0.25, 0.25])
        student_probs = initial_probs + progress * (teacher_probs - initial_probs)
        student_probs = student_probs / student_probs.sum()  # Normalize
        
        # Create subplots
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
        
        # Title
        fig.suptitle(
            f'Knowledge Distillation Process (Epoch {int(progress * 100)}/100)',
            fontsize=18, color='white', weight='bold', y=0.98
        )
        
        # Plot 1: Teacher soft targets
        ax1 = fig.add_subplot(gs[0, 0])
        bars1 = ax1.bar(range(4), teacher_probs, color='#1e3a8a', alpha=0.8, edgecolor='white')
        ax1.set_title('Teacher Model\n(Soft Targets, τ=5)', fontsize=12, color='white', weight='bold')
        ax1.set_ylabel('Probability', color='white')
        ax1.set_xticks(range(4))
        ax1.set_xticklabels(classes, fontsize=9, color='white')
        ax1.set_ylim(0, 1.0)
        ax1.tick_params(colors='white')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars1, teacher_probs)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', color='white', fontsize=9)
        
        # Plot 2: Hard labels
        ax2 = fig.add_subplot(gs[0, 1])
        bars2 = ax2.bar(range(4), hard_label, color='#b91c1c', alpha=0.8, edgecolor='white')
        ax2.set_title('Ground Truth\n(Hard Labels, τ=1)', fontsize=12, color='white', weight='bold')
        ax2.set_ylabel('Probability', color='white')
        ax2.set_xticks(range(4))
        ax2.set_xticklabels(classes, fontsize=9, color='white')
        ax2.set_ylim(0, 1.0)
        ax2.tick_params(colors='white')
        ax2.grid(axis='y', alpha=0.3)
        
        # Plot 3: Student predictions (evolving)
        ax3 = fig.add_subplot(gs[0, 2])
        bars3 = ax3.bar(range(4), student_probs, color='#15803d', alpha=0.8, edgecolor='white')
        ax3.set_title('Student Model\n(Learning)', fontsize=12, color='white', weight='bold')
        ax3.set_ylabel('Probability', color='white')
        ax3.set_xticks(range(4))
        ax3.set_xticklabels(classes, fontsize=9, color='white')
        ax3.set_ylim(0, 1.0)
        ax3.tick_params(colors='white')
        ax3.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars3, student_probs)):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', color='white', fontsize=9)
        
        # Plot 4: Loss curves
        ax4 = fig.add_subplot(gs[1, :])
        epochs = np.arange(int(progress * 100) + 1)
        
        # Simulated loss curves
        distill_loss = 2.0 * np.exp(-epochs / 30) + 0.1
        hard_loss = 1.5 * np.exp(-epochs / 25) + 0.05
        total_loss = 0.8 * distill_loss + 0.2 * hard_loss
        
        ax4.plot(epochs, distill_loss[:len(epochs)], label='Distillation Loss (KL div)', 
                color='#1e3a8a', linewidth=2)
        ax4.plot(epochs, hard_loss[:len(epochs)], label='Hard Label Loss (CE)',
                color='#b91c1c', linewidth=2)
        ax4.plot(epochs, total_loss[:len(epochs)], label='Total Loss (α=0.8)',
                color='#15803d', linewidth=3, linestyle='--')
        
        ax4.set_title('Training Loss Curves', fontsize=14, color='white', weight='bold')
        ax4.set_xlabel('Epoch', color='white', fontsize=12)
        ax4.set_ylabel('Loss', color='white', fontsize=12)
        ax4.set_xlim(0, 100)
        ax4.set_ylim(0, 2.5)
        ax4.tick_params(colors='white')
        ax4.legend(loc='upper right', fontsize=10)
        ax4.grid(alpha=0.3)
        
        # Plot 5: KL divergence (student vs teacher)
        ax5 = fig.add_subplot(gs[2, :2])
        
        kl_div = np.sum(teacher_probs * np.log((teacher_probs + 1e-10) / (student_probs + 1e-10)))
        kl_history = 1.5 * np.exp(-epochs / 35) + 0.05
        
        ax5.plot(epochs, kl_history[:len(epochs)], color='#b45309', linewidth=2)
        ax5.scatter([int(progress * 100)], [kl_div], color='#15803d', s=100, zorder=5)
        ax5.set_title('KL Divergence (Teacher || Student)', fontsize=14, color='white', weight='bold')
        ax5.set_xlabel('Epoch', color='white', fontsize=12)
        ax5.set_ylabel('KL Divergence', color='white', fontsize=12)
        ax5.set_xlim(0, 100)
        ax5.set_ylim(0, 2.0)
        ax5.tick_params(colors='white')
        ax5.grid(alpha=0.3)
        
        # Annotation
        ax5.text(int(progress * 100) + 5, kl_div + 0.1, f'KL = {kl_div:.3f}',
                fontsize=10, color='white', bbox=dict(boxstyle='round', facecolor='#15803d', alpha=0.8))
        
        # Plot 6: Model accuracy
        ax6 = fig.add_subplot(gs[2, 2])
        
        teacher_acc = 85.4
        baseline_acc = 78.1
        student_acc = baseline_acc + progress * (83.2 - baseline_acc)
        
        models = ['Teacher\n(ResNet-50)', 'Student\n(Baseline)', f'Student\n(Distilled)\nEpoch {int(progress*100)}']
        accs = [teacher_acc, baseline_acc, student_acc]
        colors_bar = ['#1e3a8a', '#b45309', '#15803d']
        
        bars6 = ax6.bar(range(3), accs, color=colors_bar, alpha=0.8, edgecolor='white')
        ax6.axhline(y=85, color='red', linestyle='--', linewidth=2, label='Target (85%)')
        ax6.set_title('Detection Accuracy (mAP@0.5)', fontsize=12, color='white', weight='bold')
        ax6.set_ylabel('mAP (%)', color='white')
        ax6.set_xticks(range(3))
        ax6.set_xticklabels(models, fontsize=9, color='white')
        ax6.set_ylim(70, 90)
        ax6.tick_params(colors='white')
        ax6.legend(fontsize=8)
        ax6.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars6, accs):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', color='white', fontsize=9)
        
        # Save frame
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, facecolor=DARK_BG, bbox_inches='tight')
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()
        plt.close(fig)
    
    return frames

if __name__ == '__main__':
    print("Generating distillation process animation...")
    frames = create_distillation_frames(num_frames=60)
    
    # Save as GIF
    frames[0].save(
        '../img/ch09-distillation-process.gif',
        save_all=True,
        append_images=frames[1:],
        duration=100,  # ms per frame
        loop=0
    )
    
    # Save key frames as PNG
    frames[0].save('../img/ch09-distillation-process-start.png')
    frames[30].save('../img/ch09-distillation-process-mid.png')
    frames[-1].save('../img/ch09-distillation-process-end.png')
    
    print("✅ Animation saved: ch09-distillation-process.gif")
    print("✅ Key frames saved: start, mid, end .png")
