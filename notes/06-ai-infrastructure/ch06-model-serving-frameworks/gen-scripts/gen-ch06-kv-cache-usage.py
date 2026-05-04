"""
Generate KV cache memory usage visualization for Ch.6 Model Serving Frameworks.

Shows how memory scales with batch size for:
- HuggingFace (static allocation, inefficient)
- vLLM without paging (better but still wasteful)
- vLLM with PagedAttention (optimal memory management)
"""

import matplotlib.pyplot as plt
import numpy as np

# Batch sizes tested
batch_sizes = np.array([1, 4, 8, 16, 32, 64])

# Model weights (constant)
model_weights = 14.2  # GB (Llama-2-7B in FP16)

# KV cache per request (200 tokens @ FP16)
kv_per_request = 0.11  # GB

# Memory usage profiles

# HuggingFace: Static allocation, allocates for max seq_len even if unused
# Overallocates by ~50%
hf_memory = model_weights + batch_sizes * kv_per_request * 1.5

# vLLM without paging: Better but still has fragmentation
# ~20% overhead from fragmentation
vllm_no_paging = model_weights + batch_sizes * kv_per_request * 1.2

# vLLM with PagedAttention: Optimal paging, minimal overhead
# ~5% overhead from page boundaries
vllm_paging = model_weights + batch_sizes * kv_per_request * 1.05

# A100 40GB limit
vram_limit = 40

# Create figure
fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#16213e')

# Plot lines
ax.plot(batch_sizes, hf_memory, 'o-', linewidth=2.5, markersize=10,
        color='#b91c1c', label='HuggingFace (static allocation)', alpha=0.8)

ax.plot(batch_sizes, vllm_no_paging, 's-', linewidth=2.5, markersize=10,
        color='#b45309', label='vLLM (no paging)', alpha=0.8)

ax.plot(batch_sizes, vllm_paging, '^-', linewidth=2.5, markersize=10,
        color='#15803d', label='vLLM (PagedAttention)', alpha=0.8)

# VRAM limit line
ax.axhline(y=vram_limit, color='#888888', linestyle='--', linewidth=2,
           label='A100 40GB VRAM limit', alpha=0.7)

# Mark OOM point for HuggingFace
oom_batch_size = batch_sizes[hf_memory > vram_limit][0] if any(hf_memory > vram_limit) else None
if oom_batch_size:
    oom_idx = np.where(batch_sizes == oom_batch_size)[0][0]
    ax.plot(oom_batch_size, hf_memory[oom_idx], 'x', markersize=20, 
            markeredgewidth=4, color='#b91c1c')
    ax.text(oom_batch_size, hf_memory[oom_idx] + 1.5, 'OOM!',
            ha='center', fontweight='bold', fontsize=12, color='#b91c1c')

# Styling
ax.set_xlabel('Batch Size (concurrent requests)', fontsize=14, fontweight='bold', color='white')
ax.set_ylabel('Peak Memory Usage (GB)', fontsize=14, fontweight='bold', color='white')
ax.set_title('KV Cache Memory Scaling with vLLM PagedAttention\nLlama-2-7B, 200 tokens per request',
             fontsize=16, fontweight='bold', pad=20, color='white')
ax.set_ylim(10, 50)
ax.set_xticks(batch_sizes)
ax.grid(alpha=0.3, linestyle='--', linewidth=0.8)
ax.legend(fontsize=12, loc='upper left', framealpha=0.9)
ax.tick_params(colors='white', labelsize=12)

# Add efficiency annotation
ax.text(0.98, 0.35, 
        'vLLM PagedAttention:\n• Zero fragmentation\n• Free pages on request completion\n• 2-4× more concurrent requests',
        transform=ax.transAxes,
        ha='right', va='top',
        fontsize=11, color='white',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#15803d', edgecolor='white', linewidth=2, alpha=0.8))

plt.tight_layout()
plt.savefig('../img/ch06-kv-cache-usage.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
print("✅ Generated: ../img/ch06-kv-cache-usage.png")
plt.close()
