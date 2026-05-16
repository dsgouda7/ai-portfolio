#!/usr/bin/env python3
"""
Generate network topology diagram comparing PCIe vs NVLink interconnects.
Shows GPU communication paths and bandwidth differences.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

# Configure matplotlib for dark backgrounds
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#1a1a2e'
plt.rcParams['axes.facecolor'] = '#1a1a2e'

def draw_nvlink_topology(ax):
    """Draw NVLink-based GPU topology (direct GPU-to-GPU)."""
    ax.set_title('NVLink Topology\nDirect GPU-to-GPU Communication', 
                 fontsize=14, fontweight='bold', pad=20, color='#15803d')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Draw 4 GPUs in a mesh
    gpu_positions = [(2, 7), (8, 7), (2, 3), (8, 3)]
    gpu_labels = ['GPU 0', 'GPU 1', 'GPU 2', 'GPU 3']
    
    for i, ((x, y), label) in enumerate(zip(gpu_positions, gpu_labels)):
        # GPU box
        gpu_box = FancyBboxPatch((x-0.8, y-0.6), 1.6, 1.2,
                                 boxstyle="round,pad=0.1",
                                 edgecolor='#15803d', facecolor='#15803d',
                                 linewidth=3, alpha=0.8)
        ax.add_patch(gpu_box)
        ax.text(x, y, label, ha='center', va='center',
                fontsize=12, fontweight='bold', color='white')
    
    # NVLink connections (green thick arrows)
    connections = [
        ((2, 7), (8, 7)),  # GPU0 - GPU1
        ((2, 3), (8, 3)),  # GPU2 - GPU3
        ((2, 7), (2, 3)),  # GPU0 - GPU2
        ((8, 7), (8, 3)),  # GPU1 - GPU3
        ((2, 7), (8, 3)),  # GPU0 - GPU3 (diagonal)
        ((8, 7), (2, 3)),  # GPU1 - GPU2 (diagonal)
    ]
    
    for (x1, y1), (x2, y2) in connections:
        # Adjust endpoints to be outside boxes
        dx = x2 - x1
        dy = y2 - y1
        length = (dx**2 + dy**2)**0.5
        dx_norm = dx / length * 0.9
        dy_norm = dy / length * 0.7
        
        ax.annotate('', xy=(x2 - dx_norm, y2 - dy_norm), 
                   xytext=(x1 + dx_norm, y1 + dy_norm),
                   arrowprops=dict(arrowstyle='<->', color='#15803d', lw=4, alpha=0.7))
    
    # Bandwidth annotation
    ax.text(5, 9, 'NVLink 3.0 (A100)', ha='center', fontsize=13,
            fontweight='bold', color='#15803d')
    ax.text(5, 8.3, '600 GB/s bidirectional', ha='center', fontsize=12,
            color='#15803d', style='italic')
    
    # Latency annotation
    ax.text(5, 0.8, 'Latency: <1 μs', ha='center', fontsize=11,
            fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#15803d', alpha=0.3))

def draw_pcie_topology(ax):
    """Draw PCIe-based GPU topology (via CPU relay)."""
    ax.set_title('PCIe Topology\nCPU-Relayed Communication', 
                 fontsize=14, fontweight='bold', pad=20, color='#b91c1c')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # CPU in center
    cpu_box = FancyBboxPatch((3.5, 4.5), 3, 1.5,
                            boxstyle="round,pad=0.1",
                            edgecolor='#b45309', facecolor='#b45309',
                            linewidth=3, alpha=0.8)
    ax.add_patch(cpu_box)
    ax.text(5, 5.25, 'CPU\nPCIe Root', ha='center', va='center',
            fontsize=12, fontweight='bold', color='white')
    
    # GPUs around CPU
    gpu_positions = [(2, 8.5), (5, 8.5), (8, 8.5), (2, 1.5), (5, 1.5), (8, 1.5)][:4]
    gpu_labels = ['GPU 0', 'GPU 1', 'GPU 2', 'GPU 3']
    
    for i, ((x, y), label) in enumerate(zip(gpu_positions, gpu_labels)):
        # GPU box
        gpu_box = FancyBboxPatch((x-0.7, y-0.5), 1.4, 1.0,
                                 boxstyle="round,pad=0.1",
                                 edgecolor='#b91c1c', facecolor='#b91c1c',
                                 linewidth=3, alpha=0.8)
        ax.add_patch(gpu_box)
        ax.text(x, y, label, ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')
        
        # PCIe connection to CPU
        target_y = 6 if y > 5 else 4.75
        ax.annotate('', xy=(x, y - 0.5 if y > 5 else y + 0.5), 
                   xytext=(5, target_y),
                   arrowprops=dict(arrowstyle='<->', color='#b91c1c', lw=3,
                                 linestyle='--', alpha=0.7))
    
    # Example GPU-to-GPU path (GPU0 → GPU3)
    # Draw highlighted path
    ax.annotate('', xy=(2, 8), xytext=(2, 6.5),
               arrowprops=dict(arrowstyle='->', color='yellow', lw=3, alpha=0.9))
    ax.annotate('', xy=(5, 6), xytext=(2.7, 5.8),
               arrowprops=dict(arrowstyle='->', color='yellow', lw=3, alpha=0.9))
    ax.annotate('', xy=(8, 4.75), xytext=(6.5, 5.2),
               arrowprops=dict(arrowstyle='->', color='yellow', lw=3, alpha=0.9))
    ax.annotate('', xy=(8, 2), xytext=(8, 4.25),
               arrowprops=dict(arrowstyle='->', color='yellow', lw=3, alpha=0.9))
    
    ax.text(1.5, 7.2, 'GPU0→CPU', fontsize=9, color='yellow', fontweight='bold')
    ax.text(6.5, 5.8, 'CPU routing', fontsize=9, color='yellow', fontweight='bold')
    ax.text(8.5, 3.5, 'CPU→GPU3', fontsize=9, color='yellow', fontweight='bold')
    
    # Bandwidth annotation
    ax.text(5, 0.3, 'PCIe Gen4: 16 GB/s per direction', ha='center', fontsize=11,
            fontweight='bold', color='#b91c1c',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#b91c1c', alpha=0.3))
    
    # Latency annotation
    ax.text(5, 9.5, 'Latency: 5-10 μs (CPU relay overhead)', ha='center', fontsize=10,
            fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#b91c1c', alpha=0.3))

def main():
    """Generate network topology comparison diagram."""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    
    # Draw topologies
    draw_nvlink_topology(ax1)
    draw_pcie_topology(ax2)
    
    # Main title
    fig.suptitle('GPU Interconnect Topologies: NVLink vs PCIe\n' + 
                 'Impact on Multi-GPU Communication Patterns',
                 fontsize=16, fontweight='bold', y=0.98, color='white')
    
    # Footer comparison
    fig.text(0.5, 0.02, 
             '🚀 NVLink: 600 GB/s, <1μs latency | ' +
             '🐌 PCIe: 16 GB/s, 5-10μs latency | ' +
             '⚡ Speedup: 37× bandwidth, 5-10× lower latency',
             ha='center', fontsize=12, color='white', weight='bold',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#1a1a2e', 
                      edgecolor='white', linewidth=2))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    
    # Save figure
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'img')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'network_topology_comparison.png')
    
    plt.savefig(output_path, dpi=150, facecolor='#1a1a2e', bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    
    plt.close()

if __name__ == '__main__':
    main()
