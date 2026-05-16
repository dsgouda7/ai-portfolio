"""
Generate throughput comparison bar chart for Ch.6 Model Serving Frameworks.

Shows throughput (requests/second) comparison across:
- HuggingFace Transformers (baseline)
- ONNX Runtime (quantized)
- vLLM (continuous batching)
- TensorRT-LLM (max performance)
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from benchmarks (Llama-2-7B on A100 40GB)
frameworks = ['HuggingFace', 'ONNX Runtime', 'vLLM', 'TensorRT']
throughputs = [10, 24, 147, 189]  # requests per second
colors = ['#b91c1c', '#1d4ed8', '#15803d', '#b45309']

# Create figure
fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#16213e')

# Create bars
bars = ax.bar(frameworks, throughputs, color=colors, edgecolor='white', linewidth=2, alpha=0.9)

# Add value labels on bars
for bar, value in zip(bars, throughputs):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 5,
            f'{value} req/s',
            ha='center', va='bottom', fontweight='bold', fontsize=13, color='white')
    
    # Add speedup vs baseline
    if value != throughputs[0]:
        speedup = value / throughputs[0]
        ax.text(bar.get_x() + bar.get_width()/2, height/2,
                f'{speedup:.1f}×',
                ha='center', va='center', fontweight='bold', fontsize=16, 
                color='white', bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))

# Styling
ax.set_ylabel('Throughput (requests/second)', fontsize=14, fontweight='bold', color='white')
ax.set_title('Model Serving Framework Throughput Comparison\nLlama-2-7B on A100 40GB',
             fontsize=16, fontweight='bold', pad=20, color='white')
ax.set_ylim(0, max(throughputs) * 1.15)
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
ax.tick_params(colors='white', labelsize=12)

# Add annotation
ax.text(0.98, 0.98, 'Higher is better', 
        transform=ax.transAxes,
        ha='right', va='top',
        fontsize=11, style='italic', color='#888888',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#16213e', edgecolor='#888888', linewidth=1))

plt.tight_layout()
plt.savefig('../img/ch06-throughput-comparison.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
print("✅ Generated: ../img/ch06-throughput-comparison.png")
plt.close()
