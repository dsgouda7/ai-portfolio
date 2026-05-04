#!/usr/bin/env python3
"""
Generate pattern comparison heatmap for Chapter 11 — Advanced Agentic Patterns

Creates a heatmap table comparing all patterns across multiple metrics:
- Cost (tokens), Latency (seconds), Error Reduction (multiplier), Use Case Fit

Output: ../img/ch11-pattern-comparison.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path


def create_pattern_comparison():
    """Generate heatmap comparing all agentic patterns"""
    
    # Pattern comparison data
    patterns = [
        "Single-pass",
        "Reflection",
        "Debate",
        "Hierarchical",
        "Tool Selection"
    ]
    
    metrics = [
        "Cost\n(tokens)",
        "Latency\n(seconds)",
        "Error\nReduction",
        "Use Case Fit"
    ]
    
    # Data matrix [pattern][metric]
    data = [
        ["150", "0.5s", "1× baseline", "Simple queries"],
        ["430", "1.5s", "6× better", "Ambiguous inputs"],
        ["900", "3.0s", "8× better", "High-stakes decisions"],
        ["750", "2.5s", "15× better", "Multi-step tasks"],
        ["165-375", "0.6-2.0s", "2× better", "Multiple tools"]
    ]
    
    # Numeric values for coloring (normalized 0-1)
    # Cost: 150-900 range
    cost_values = [150, 430, 900, 750, 270]  # avg for tool selection
    cost_normalized = [(v - 150) / (900 - 150) for v in cost_values]
    
    # Latency: 0.5-3.0 range
    latency_values = [0.5, 1.5, 3.0, 2.5, 1.3]  # avg for tool selection
    latency_normalized = [(v - 0.5) / (3.0 - 0.5) for v in latency_values]
    
    # Error reduction: 1-15 range (higher is better, so invert)
    error_values = [1, 6, 8, 15, 2]
    error_normalized = [(v - 1) / (15 - 1) for v in error_values]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    # Cell dimensions
    cell_width = 3.0
    cell_height = 1.0
    
    # Title
    ax.text(6.0, 6.5, "Pattern Comparison Matrix",
            fontsize=16, fontweight='bold', color='white',
            ha='center', va='center')
    
    # Draw column headers
    for j, metric in enumerate(metrics):
        x = 1.5 + j * cell_width
        y = 5.5
        
        # Header background
        rect = patches.Rectangle((x - cell_width/2, y - cell_height/2),
                                cell_width, cell_height,
                                linewidth=1.5, edgecolor='white',
                                facecolor='#2d2d44')
        ax.add_patch(rect)
        
        # Header text
        ax.text(x, y, metric, fontsize=12, fontweight='bold',
                color='white', ha='center', va='center')
    
    # Draw data cells with heatmap coloring
    for i, pattern in enumerate(patterns):
        y = 4.5 - i * cell_height
        
        # Row label
        ax.text(0.2, y, pattern, fontsize=11, fontweight='bold',
                color='white', ha='left', va='center')
        
        for j, value in enumerate(data[i]):
            x = 1.5 + j * cell_width
            
            # Determine cell color based on metric
            if j == 0:  # Cost (low is good)
                color_intensity = cost_normalized[i]
                cell_color = plt.cm.RdYlGn_r(color_intensity)
            elif j == 1:  # Latency (low is good)
                color_intensity = latency_normalized[i]
                cell_color = plt.cm.RdYlGn_r(color_intensity)
            elif j == 2:  # Error reduction (high is good)
                color_intensity = error_normalized[i]
                cell_color = plt.cm.RdYlGn(color_intensity)
            else:  # Use case (no color)
                cell_color = '#2d2d44'
            
            # Cell background
            rect = patches.Rectangle((x - cell_width/2, y - cell_height/2),
                                    cell_width, cell_height,
                                    linewidth=1, edgecolor='#555',
                                    facecolor=cell_color)
            ax.add_patch(rect)
            
            # Cell text
            text_color = 'white' if j < 3 else '#e0e0e0'
            ax.text(x, y, value, fontsize=11, color=text_color,
                    ha='center', va='center', fontweight='normal')
    
    # Add color legend
    legend_y = 0.3
    legend_x_start = 1.5
    
    ax.text(6.0, legend_y + 0.5, "Heatmap Legend",
            fontsize=11, fontweight='bold', color='white',
            ha='center', va='center')
    
    # Cost/Latency legend (low is good)
    for i in range(10):
        x = legend_x_start + i * 0.5
        color = plt.cm.RdYlGn_r(i / 9)
        rect = patches.Rectangle((x - 0.25, legend_y - 0.15),
                                0.5, 0.3, facecolor=color, edgecolor='none')
        ax.add_patch(rect)
    
    ax.text(legend_x_start - 0.5, legend_y, "High",
            fontsize=9, color='white', ha='right', va='center')
    ax.text(legend_x_start + 4.5 + 0.5, legend_y, "Low",
            fontsize=9, color='white', ha='left', va='center')
    ax.text(legend_x_start + 2.25, legend_y - 0.6, "Cost / Latency (lower is better)",
            fontsize=9, color='#aaa', ha='center', va='top')
    
    # Error reduction legend (high is good)
    legend_x_start2 = 7.5
    for i in range(10):
        x = legend_x_start2 + i * 0.5
        color = plt.cm.RdYlGn(i / 9)
        rect = patches.Rectangle((x - 0.25, legend_y - 0.15),
                                0.5, 0.3, facecolor=color, edgecolor='none')
        ax.add_patch(rect)
    
    ax.text(legend_x_start2 - 0.5, legend_y, "Low",
            fontsize=9, color='white', ha='right', va='center')
    ax.text(legend_x_start2 + 4.5 + 0.5, legend_y, "High",
            fontsize=9, color='white', ha='left', va='center')
    ax.text(legend_x_start2 + 2.25, legend_y - 0.6, "Error Reduction (higher is better)",
            fontsize=9, color='#aaa', ha='center', va='top')
    
    # Add insight annotation
    insight_box = patches.FancyBboxPatch((0.5, -1.2), 11, 0.7,
                                        boxstyle="round,pad=0.1",
                                        linewidth=2, edgecolor='#4a9eff',
                                        facecolor='#1e3a5f', alpha=0.8)
    ax.add_patch(insight_box)
    
    ax.text(6.0, -0.85, "💡 Key Insight: Trade tokens for reliability",
            fontsize=11, fontweight='bold', color='#4a9eff',
            ha='center', va='center')
    ax.text(6.0, -1.05, "3-6× cost increase → 6-15× error reduction",
            fontsize=10, color='white', ha='center', va='center')
    
    # Set axis limits and remove axes
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-1.8, 7.0)
    ax.axis('off')
    
    # Save
    output_path = Path(__file__).parent.parent / "img" / "ch11-pattern-comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='#1a1a2e', edgecolor='none')
    print(f"Generated {output_path}")


if __name__ == "__main__":
    create_pattern_comparison()
