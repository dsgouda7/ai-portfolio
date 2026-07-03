"""
Generate temperature scaling comparison animation for Ch.9 - Knowledge Distillation.

Visualizes how temperature parameter τ affects probability distribution softness.
Shows transition from hard (τ=1) to soft (τ=5) probabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# Dark theme
plt.style.use('dark_background')
DARK_BG = '#1a1a2e'

def softmax(logits, tau=1.0):
    """Compute temperature-scaled softmax."""
    scaled_logits = logits / tau
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))  # Numerical stability
    return exp_logits / exp_logits.sum()

def create_temperature_frames(num_frames=50):
    """Generate frames showing temperature effect on probability distribution."""
    frames = []
    
    # Example logits (soda can classification)
    logits = np.array([5.0, 1.0, 0.5, -1.0])
    classes = ['Soda\nCan', 'Water\nBottle', 'Juice\nBox', 'Cereal\nBox']
    
    # Temperature range: 1 (hard) → 5 (soft) → 1 (return)
    temperatures = np.concatenate([
        np.linspace(1, 5, num_frames // 2),
        np.linspace(5, 1, num_frames // 2)
    ])
    
    for frame_idx, tau in enumerate(temperatures):
        fig = plt.figure(figsize=(14, 8), facecolor=DARK_BG)
        fig.patch.set_facecolor(DARK_BG)
        
        # Compute probabilities at current temperature
        probs = softmax(logits, tau)
        
        # Create layout
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle(
            f'Temperature Scaling: τ = {tau:.2f}',
            fontsize=20, color='white', weight='bold', y=0.98
        )
        
        # Plot 1: Probability distribution
        ax1 = fig.add_subplot(gs[0, :])
        bars = ax1.bar(range(4), probs, color='#15803d', alpha=0.8, edgecolor='white', linewidth=2)
        ax1.set_title(f'Probability Distribution (τ={tau:.2f})', fontsize=16, color='white', weight='bold')
        ax1.set_ylabel('Probability', color='white', fontsize=14)
        ax1.set_xticks(range(4))
        ax1.set_xticklabels(classes, fontsize=12, color='white')
        ax1.set_ylim(0, 1.0)
        ax1.tick_params(colors='white', labelsize=12)
        ax1.grid(axis='y', alpha=0.3, linewidth=1.5)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, probs)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', color='white', 
                    fontsize=12, weight='bold')
        
        # Color bars based on probability (gradient effect)
        for i, bar in enumerate(bars):
            # Higher probability → brighter green
            alpha = 0.5 + 0.5 * (probs[i] / probs.max())
            bar.set_alpha(alpha)
        
        # Plot 2: Entropy (measure of distribution uniformity)
        ax2 = fig.add_subplot(gs[1, 0])
        
        # Entropy: H = -Σ p_i * log(p_i)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(4)  # Uniform distribution over 4 classes
        
        # Plot entropy history
        temp_history = temperatures[:frame_idx + 1]
        entropy_history = []
        for t in temp_history:
            p = softmax(logits, t)
            h = -np.sum(p * np.log(p + 1e-10))
            entropy_history.append(h)
        
        ax2.plot(temp_history, entropy_history, color='#b45309', linewidth=3)
        ax2.scatter([tau], [entropy], color='#15803d', s=200, zorder=5, edgecolor='white', linewidth=2)
        ax2.axhline(y=max_entropy, color='red', linestyle='--', linewidth=2, label='Max Entropy (Uniform)')
        ax2.set_title('Distribution Entropy', fontsize=14, color='white', weight='bold')
        ax2.set_xlabel('Temperature (τ)', color='white', fontsize=12)
        ax2.set_ylabel('Entropy', color='white', fontsize=12)
        ax2.set_xlim(1, 5)
        ax2.set_ylim(0, max_entropy + 0.2)
        ax2.tick_params(colors='white', labelsize=11)
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)
        
        # Annotation
        ax2.text(tau + 0.15, entropy + 0.05, f'H = {entropy:.3f}',
                fontsize=11, color='white', weight='bold',
                bbox=dict(boxstyle='round', facecolor='#15803d', alpha=0.8))
        
        # Plot 3: Comparison table
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis('off')
        
        # Reference temperatures
        ref_temps = [1.0, 3.0, 5.0, 10.0]
        ref_probs = [softmax(logits, t) for t in ref_temps]
        
        table_data = [['τ', 'Soda', 'Water', 'Juice', 'Cereal', 'Entropy']]
        for t, p in zip(ref_temps, ref_probs):
            h = -np.sum(p * np.log(p + 1e-10))
            row = [f'{t:.1f}'] + [f'{val:.3f}' for val in p] + [f'{h:.3f}']
            table_data.append(row)
        
        # Highlight current temperature row
        cell_colors = [['#2d2d44'] * 6 for _ in range(5)]
        cell_colors[0] = ['#1e3a8a'] * 6  # Header
        
        # Find closest reference temperature
        closest_idx = np.argmin(np.abs(np.array(ref_temps) - tau)) + 1
        if abs(ref_temps[closest_idx - 1] - tau) < 0.5:
            cell_colors[closest_idx] = ['#15803d'] * 6  # Highlight current
        
        table = ax3.table(cellText=table_data, cellLoc='center', loc='center',
                         cellColours=cell_colors, edges='closed')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # Style table
        for (i, j), cell in table.get_celld().items():
            cell.set_text_props(color='white', weight='bold' if i == 0 else 'normal')
            cell.set_linewidth(1.5)
            cell.set_edgecolor('white')
        
        ax3.set_title('Temperature Effect Reference', fontsize=14, color='white', 
                     weight='bold', pad=20)
        
        # Add explanation text
        explanation_y = 0.05
        if tau < 2.0:
            explanation = "Low τ → Hard distribution\\n(one class dominates)"
            color = '#b91c1c'
        elif tau < 4.0:
            explanation = "Medium τ → Moderate softness\\n(class similarities emerge)"
            color = '#b45309'
        else:
            explanation = "High τ → Soft distribution\\n(strong class similarity signal)"
            color = '#15803d'
        
        fig.text(0.5, explanation_y, explanation, ha='center', va='bottom',
                fontsize=13, color=color, weight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.2, edgecolor='white', linewidth=2))
        
        # Save frame
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, facecolor=DARK_BG, bbox_inches='tight')
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()
        plt.close(fig)
    
    return frames

if __name__ == '__main__':
    print("Generating temperature scaling animation...")
    frames = create_temperature_frames(num_frames=50)
    
    # Save as GIF
    frames[0].save(
        '../img/ch09-temperature-scaling.gif',
        save_all=True,
        append_images=frames[1:],
        duration=120,  # ms per frame
        loop=0
    )
    
    # Save key frames
    frames[0].save('../img/ch09-temperature-tau1.png')  # τ=1 (hard)
    frames[25].save('../img/ch09-temperature-tau5.png')  # τ=5 (soft)
    frames[-1].save('../img/ch09-temperature-tau1-return.png')  # Back to τ=1
    
    print("✅ Animation saved: ch09-temperature-scaling.gif")
    print("✅ Key frames saved: tau1, tau5, tau1-return .png")
