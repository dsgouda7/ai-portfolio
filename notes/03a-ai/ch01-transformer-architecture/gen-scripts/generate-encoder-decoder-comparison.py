"""
Generate encoder-decoder comparison visualization showing attention mask patterns.
Usage: python generate-encoder-decoder-comparison.py
Output: ../img/encoder-decoder-comparison.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def create_attention_mask_visualization():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Define 6x6 matrices for illustration
    n = 6
    
    # 1. Encoder (Bidirectional - full attention)
    encoder_mask = np.ones((n, n))
    
    # 2. Decoder (Causal - lower triangular)
    decoder_mask = np.tril(np.ones((n, n)))
    
    # 3. Encoder-Decoder Cross-Attention
    # Decoder can attend to all encoder outputs
    cross_mask = np.ones((n, n))
    
    masks = [encoder_mask, decoder_mask, cross_mask]
    titles = [
        'Encoder\n(Bidirectional Attention)',
        'Decoder\n(Causal Attention)',
        'Cross-Attention\n(Decoder → Encoder)'
    ]
    descriptions = [
        'Every token attends to\nall other tokens',
        'Each token attends only to\nitself and previous tokens',
        'Decoder tokens attend to\nall encoder outputs'
    ]
    
    for ax, mask, title, desc in zip(axes, masks, titles, descriptions):
        # Create heatmap
        im = ax.imshow(mask, cmap='RdYlGn', vmin=0, vmax=1, aspect='equal')
        
        # Add grid
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels([f't{i+1}' for i in range(n)])
        ax.set_yticklabels([f't{i+1}' for i in range(n)])
        
        # Add labels
        ax.set_xlabel('Key/Value Tokens', fontsize=11, fontweight='bold')
        ax.set_ylabel('Query Tokens', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        
        # Add description
        ax.text(0.5, -0.25, desc, transform=ax.transAxes,
                ha='center', fontsize=10, style='italic',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Add grid lines
        for i in range(n+1):
            ax.axhline(i-0.5, color='gray', linewidth=0.5)
            ax.axvline(i-0.5, color='gray', linewidth=0.5)
        
        # Annotate mask pattern
        for i in range(n):
            for j in range(n):
                if mask[i, j] == 1:
                    ax.text(j, i, '✓', ha='center', va='center',
                           color='darkgreen', fontsize=14, fontweight='bold')
                else:
                    ax.text(j, i, '✗', ha='center', va='center',
                           color='darkred', fontsize=14, fontweight='bold')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='#90ee90', label='Attention allowed (weight computed)'),
        mpatches.Patch(facecolor='#ffcccb', label='Attention masked (weight = 0)')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2,
              bbox_to_anchor=(0.5, 0.02), fontsize=11, frameon=True)
    
    # Add architecture type annotations
    arch_annotations = [
        ('BERT, RoBERTa, E5\n(Understanding)', 0.17, 0.92),
        ('GPT, Claude, LLaMA\n(Generation)', 0.5, 0.92),
        ('T5, BART\n(Seq2Seq)', 0.83, 0.92)
    ]
    
    for text, x, y in arch_annotations:
        fig.text(x, y, text, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.suptitle('Transformer Attention Patterns: Encoder vs Decoder vs Cross-Attention',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    
    # Save figure
    output_path = '../img/encoder-decoder-comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'✓ Generated: {output_path}')
    plt.close()

if __name__ == '__main__':
    create_attention_mask_visualization()
