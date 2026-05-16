#!/usr/bin/env python3
"""
Generate bandwidth comparison bar chart for different GPU interconnects.
Compares PCIe Gen3/Gen4, NVLink 3.0/4.0, and InfiniBand HDR/NDR.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Configure matplotlib for dark backgrounds
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#1a1a2e'
plt.rcParams['axes.facecolor'] = '#1a1a2e'

def generate_bandwidth_comparison():
    """Generate bandwidth comparison bar chart."""
    
    # Data: interconnect types and their bandwidth (GB/s)
    interconnects = [
        'PCIe\nGen3 ×16',
        'PCIe\nGen4 ×16',
        'PCIe\nGen5 ×16',
        'InfiniBand\nHDR',
        'InfiniBand\nNDR',
        'NVLink 2.0\n(V100)',
        'NVLink 3.0\n(A100)',
        'NVLink 4.0\n(H100)'
    ]
    
    # Bandwidth in GB/s (bidirectional)
    bandwidths = [
        16,     # PCIe Gen3
        32,     # PCIe Gen4
        64,     # PCIe Gen5
        50,     # InfiniBand HDR (200 Gb/s = 25 GB/s per direction, 50 GB/s bidir)
        100,    # InfiniBand NDR (400 Gb/s = 50 GB/s per direction, 100 GB/s bidir)
        300,    # NVLink 2.0 (V100)
        600,    # NVLink 3.0 (A100)
        900     # NVLink 4.0 (H100)
    ]
    
    # Colors: Red for PCIe, Blue for InfiniBand, Green for NVLink
    colors = [
        '#b91c1c', '#b91c1c', '#b91c1c',  # PCIe (red shades)
        '#1d4ed8', '#1d4ed8',              # InfiniBand (blue shades)
        '#15803d', '#15803d', '#15803d'   # NVLink (green shades)
    ]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create bars
    x_pos = np.arange(len(interconnects))
    bars = ax.bar(x_pos, bandwidths, color=colors, alpha=0.85, 
                   edgecolor='white', linewidth=2)
    
    # Add value labels on bars
    for bar, bw in zip(bars, bandwidths):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 20,
                f'{bw} GB/s',
                ha='center', va='bottom', fontsize=11, fontweight='bold', color='white')
    
    # Add speedup annotations relative to PCIe Gen4
    pcie_gen4_bw = 32
    for i, (bar, bw) in enumerate(zip(bars, bandwidths)):
        if bw > pcie_gen4_bw:
            speedup = bw / pcie_gen4_bw
            ax.text(bar.get_x() + bar.get_width()/2., bw + 50,
                    f'{speedup:.1f}×',
                    ha='center', va='bottom', fontsize=10, 
                    color='yellow', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))
    
    # Formatting
    ax.set_xlabel('Interconnect Type', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel('Bidirectional Bandwidth (GB/s)', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_title('GPU Interconnect Bandwidth Comparison\nHigher is Better for Multi-GPU AI Workloads',
                 fontsize=15, fontweight='bold', pad=20)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(interconnects, fontsize=11)
    ax.set_ylim(0, max(bandwidths) * 1.2)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add reference lines
    ax.axhline(y=pcie_gen4_bw, color='yellow', linestyle='--', linewidth=2, 
               alpha=0.5, label='PCIe Gen4 baseline (32 GB/s)')
    
    # Add legend for categories
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#b91c1c', edgecolor='white', label='PCIe (CPU-centric)'),
        Patch(facecolor='#1d4ed8', edgecolor='white', label='InfiniBand (Multi-node fabric)'),
        Patch(facecolor='#15803d', edgecolor='white', label='NVLink (GPU-to-GPU direct)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.9)
    
    # Add annotations for use cases
    ax.text(1, 700, 'Consumer GPUs\n(RTX 4090, 3090)', ha='center', fontsize=9, 
            color='#b91c1c', style='italic')
    ax.text(4, 700, 'Multi-node\nClusters', ha='center', fontsize=9,
            color='#1d4ed8', style='italic')
    ax.text(7, 700, 'Datacenter GPUs\n(A100, H100)', ha='center', fontsize=9,
            color='#15803d', style='italic')
    
    # Footer note
    fig.text(0.5, 0.02,
             '💡 Key Insight: NVLink is 18-28× faster than PCIe Gen4, enabling efficient tensor parallelism',
             ha='center', fontsize=12, color='white', weight='bold',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#1a1a2e',
                      edgecolor='yellow', linewidth=2))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    # Save figure
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'img')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'bandwidth_comparison.png')
    
    plt.savefig(output_path, dpi=150, facecolor='#1a1a2e', bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    
    plt.close()

if __name__ == '__main__':
    generate_bandwidth_comparison()
