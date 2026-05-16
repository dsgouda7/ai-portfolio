#!/usr/bin/env python3
"""
Generate latency heatmap showing GPU-to-GPU communication patterns.
Visualizes the impact of topology (NVLink mesh vs PCIe) on inter-GPU latency.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Configure matplotlib for dark backgrounds
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#1a1a2e'
plt.rcParams['axes.facecolor'] = '#1a1a2e'

def generate_latency_heatmap():
    """Generate latency heatmap for different GPU topologies."""
    
    # Create figure with 3 subplots (NVLink, PCIe, InfiniBand)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # === NVLink Topology (8 GPUs fully connected) ===
    # Latency in microseconds (μs)
    nvlink_latency = np.array([
        [0.0, 0.8, 0.8, 1.2, 0.8, 1.2, 1.2, 1.6],
        [0.8, 0.0, 1.2, 0.8, 1.2, 0.8, 1.6, 1.2],
        [0.8, 1.2, 0.0, 0.8, 1.2, 1.6, 0.8, 1.2],
        [1.2, 0.8, 0.8, 0.0, 1.6, 1.2, 1.2, 0.8],
        [0.8, 1.2, 1.2, 1.6, 0.0, 0.8, 0.8, 1.2],
        [1.2, 0.8, 1.6, 1.2, 0.8, 0.0, 1.2, 0.8],
        [1.2, 1.6, 0.8, 1.2, 0.8, 1.2, 0.0, 0.8],
        [1.6, 1.2, 1.2, 0.8, 1.2, 0.8, 0.8, 0.0]
    ])
    
    sns.heatmap(nvlink_latency, annot=True, fmt='.1f', cmap='Greens_r',
                cbar_kws={'label': 'Latency (μs)'}, ax=axes[0],
                vmin=0, vmax=15, linewidths=0.5, linecolor='white')
    axes[0].set_title('NVLink Topology (8× A100)\nDirect GPU-to-GPU', 
                      fontsize=13, fontweight='bold', color='#15803d', pad=15)
    axes[0].set_xlabel('Target GPU', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Source GPU', fontsize=11, fontweight='bold')
    axes[0].set_xticklabels(range(8), fontsize=10)
    axes[0].set_yticklabels(range(8), fontsize=10, rotation=0)
    
    # Add annotation
    axes[0].text(4, -1.2, '✅ Uniform low latency (<2μs)', ha='center', fontsize=11,
                 color='#15803d', fontweight='bold')
    
    # === PCIe Topology (8 GPUs via CPU) ===
    # Higher latency due to CPU relay, cross-socket penalty
    pcie_latency = np.array([
        [0.0, 5.5, 5.5, 5.5, 8.5, 8.5, 8.5, 8.5],  # GPU0 same socket 0-3, cross-socket 4-7
        [5.5, 0.0, 5.5, 5.5, 8.5, 8.5, 8.5, 8.5],
        [5.5, 5.5, 0.0, 5.5, 8.5, 8.5, 8.5, 8.5],
        [5.5, 5.5, 5.5, 0.0, 8.5, 8.5, 8.5, 8.5],
        [8.5, 8.5, 8.5, 8.5, 0.0, 5.5, 5.5, 5.5],  # GPU4 socket 1
        [8.5, 8.5, 8.5, 8.5, 5.5, 0.0, 5.5, 5.5],
        [8.5, 8.5, 8.5, 8.5, 5.5, 5.5, 0.0, 5.5],
        [8.5, 8.5, 8.5, 8.5, 5.5, 5.5, 5.5, 0.0]
    ])
    
    sns.heatmap(pcie_latency, annot=True, fmt='.1f', cmap='Reds',
                cbar_kws={'label': 'Latency (μs)'}, ax=axes[1],
                vmin=0, vmax=15, linewidths=0.5, linecolor='white')
    axes[1].set_title('PCIe Topology (8 GPUs)\nCPU-Relayed Communication', 
                      fontsize=13, fontweight='bold', color='#b91c1c', pad=15)
    axes[1].set_xlabel('Target GPU', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Source GPU', fontsize=11, fontweight='bold')
    axes[1].set_xticklabels(range(8), fontsize=10)
    axes[1].set_yticklabels(range(8), fontsize=10, rotation=0)
    
    # Add NUMA boundary annotation
    axes[1].axhline(y=4, color='yellow', linestyle='--', linewidth=2, alpha=0.7)
    axes[1].axvline(x=4, color='yellow', linestyle='--', linewidth=2, alpha=0.7)
    axes[1].text(2, 8.5, 'Socket 0', ha='center', fontsize=10, color='yellow', fontweight='bold')
    axes[1].text(6, 8.5, 'Socket 1', ha='center', fontsize=10, color='yellow', fontweight='bold')
    
    axes[1].text(4, -1.2, '⚠️  High latency, cross-socket penalty', ha='center', fontsize=11,
                 color='#b91c1c', fontweight='bold')
    
    # === InfiniBand Multi-Node (2 nodes × 4 GPUs) ===
    # Low latency within node (NVLink), higher cross-node (InfiniBand)
    infiniband_latency = np.array([
        [0.0, 0.8, 0.8, 1.2, 2.5, 3.0, 3.0, 3.5],  # GPU0 node 0, cross to node 1
        [0.8, 0.0, 1.2, 0.8, 3.0, 2.5, 3.5, 3.0],
        [0.8, 1.2, 0.0, 0.8, 3.0, 3.5, 2.5, 3.0],
        [1.2, 0.8, 0.8, 0.0, 3.5, 3.0, 3.0, 2.5],
        [2.5, 3.0, 3.0, 3.5, 0.0, 0.8, 0.8, 1.2],  # GPU4 node 1
        [3.0, 2.5, 3.5, 3.0, 0.8, 0.0, 1.2, 0.8],
        [3.0, 3.5, 2.5, 3.0, 0.8, 1.2, 0.0, 0.8],
        [3.5, 3.0, 3.0, 2.5, 1.2, 0.8, 0.8, 0.0]
    ])
    
    sns.heatmap(infiniband_latency, annot=True, fmt='.1f', cmap='Blues',
                cbar_kws={'label': 'Latency (μs)'}, ax=axes[2],
                vmin=0, vmax=15, linewidths=0.5, linecolor='white')
    axes[2].set_title('InfiniBand Multi-Node (2×4 GPUs)\nNVLink + InfiniBand Fabric', 
                      fontsize=13, fontweight='bold', color='#1d4ed8', pad=15)
    axes[2].set_xlabel('Target GPU', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('Source GPU', fontsize=11, fontweight='bold')
    axes[2].set_xticklabels(range(8), fontsize=10)
    axes[2].set_yticklabels(range(8), fontsize=10, rotation=0)
    
    # Add node boundary
    axes[2].axhline(y=4, color='cyan', linestyle='--', linewidth=2, alpha=0.7)
    axes[2].axvline(x=4, color='cyan', linestyle='--', linewidth=2, alpha=0.7)
    axes[2].text(2, 8.5, 'Node 0', ha='center', fontsize=10, color='cyan', fontweight='bold')
    axes[2].text(6, 8.5, 'Node 1', ha='center', fontsize=10, color='cyan', fontweight='bold')
    
    axes[2].text(4, -1.2, '✅ Low within-node, acceptable cross-node', ha='center', fontsize=11,
                 color='#1d4ed8', fontweight='bold')
    
    # Main title
    fig.suptitle('GPU-to-GPU Communication Latency Heatmaps\nImpact of Interconnect Topology on Multi-GPU Performance',
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Footer comparison
    fig.text(0.5, -0.05,
             '🔑 Key: Darker = Lower Latency (Better) | ' +
             'NVLink: <2μs | PCIe: 5-9μs | InfiniBand: 2-4μs cross-node',
             ha='center', fontsize=12, color='white', weight='bold',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#1a1a2e',
                      edgecolor='white', linewidth=2))
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    
    # Save figure
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'img')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'latency_heatmap.png')
    
    plt.savefig(output_path, dpi=150, facecolor='#1a1a2e', bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    
    plt.close()

if __name__ == '__main__':
    generate_latency_heatmap()
