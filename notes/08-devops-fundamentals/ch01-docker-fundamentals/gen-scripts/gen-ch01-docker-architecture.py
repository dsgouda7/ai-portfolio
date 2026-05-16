#!/usr/bin/env python3
"""
Generate Docker architecture diagram showing image layers.

Demonstrates how Docker images are built from layered filesystems,
with each Dockerfile instruction creating a cached layer.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Dark theme styling (consistent with other ML notebooks)
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(12, 8), facecolor='#1a1a2e')
ax.set_facecolor('#1a1a2e')

# Layer data (bottom to top)
layers = [
    {
        'label': 'FROM python:3.11-slim',
        'size': '130 MB',
        'color': '#1e3a8a',  # Blue
        'description': 'Base OS + Python runtime'
    },
    {
        'label': 'WORKDIR /app',
        'size': '0 B',
        'color': '#15803d',  # Green
        'description': 'Set working directory'
    },
    {
        'label': 'COPY requirements.txt .',
        'size': '32 B',
        'color': '#15803d',
        'description': 'Dependency list'
    },
    {
        'label': 'RUN pip install -r requirements.txt',
        'size': '12.3 MB',
        'color': '#b45309',  # Orange
        'description': 'Flask + Redis packages'
    },
    {
        'label': 'COPY app.py .',
        'size': '789 B',
        'color': '#15803d',
        'description': 'Application code'
    },
    {
        'label': 'CMD ["python", "app.py"]',
        'size': '0 B',
        'color': '#1d4ed8',  # Light blue
        'description': 'Startup command'
    }
]

# Draw layers as stacked boxes
y_offset = 1.0
layer_height = 0.8
layer_width = 8.0

for i, layer in enumerate(layers):
    # Main layer box
    box = FancyBboxPatch(
        (1.0, y_offset),
        layer_width,
        layer_height,
        boxstyle="round,pad=0.05",
        edgecolor='white',
        facecolor=layer['color'],
        linewidth=2,
        alpha=0.8
    )
    ax.add_patch(box)
    
    # Layer instruction (left side)
    ax.text(
        1.3, y_offset + layer_height / 2,
        f"{layer['label']}",
        fontsize=10,
        fontweight='bold',
        color='white',
        va='center',
        ha='left',
        family='monospace'
    )
    
    # Layer size (right side)
    ax.text(
        8.5, y_offset + layer_height / 2,
        layer['size'],
        fontsize=9,
        color='white',
        va='center',
        ha='right',
        family='monospace'
    )
    
    # Description (outside box, right side)
    ax.text(
        9.2, y_offset + layer_height / 2,
        layer['description'],
        fontsize=8,
        color='#cccccc',
        va='center',
        ha='left',
        style='italic'
    )
    
    # Layer number badge
    circle = plt.Circle(
        (0.6, y_offset + layer_height / 2),
        0.25,
        color='white',
        zorder=10
    )
    ax.add_patch(circle)
    ax.text(
        0.6, y_offset + layer_height / 2,
        str(i + 1),
        fontsize=10,
        fontweight='bold',
        color='#1a1a2e',
        va='center',
        ha='center',
        zorder=11
    )
    
    y_offset += layer_height + 0.3

# Add cache indicators
cache_x = 9.8
cache_indicators = [
    (2.1, '✓ Cached', '#15803d'),  # Layer 2 (WORKDIR)
    (3.2, '✓ Cached', '#15803d'),  # Layer 3 (COPY requirements)
    (4.3, '✓ Cached', '#15803d'),  # Layer 4 (pip install)
    (5.4, '✗ Changed', '#b91c1c'),  # Layer 5 (COPY app.py)
]

for y_pos, label, color in cache_indicators:
    ax.text(
        cache_x, y_pos,
        label,
        fontsize=8,
        fontweight='bold',
        color=color,
        bbox=dict(
            boxstyle='round,pad=0.3',
            facecolor='#2a2a3e',
            edgecolor=color,
            linewidth=1.5
        )
    )

# Add title and annotations
ax.text(
    6.0, 7.8,
    'Docker Image Layers',
    fontsize=18,
    fontweight='bold',
    color='white',
    ha='center'
)

ax.text(
    6.0, 7.3,
    'Each Dockerfile instruction creates a cached layer',
    fontsize=11,
    color='#cccccc',
    ha='center',
    style='italic'
)

# Add total size annotation
total_size = 130 + 12.3 + 0.001  # MB
ax.text(
    6.0, 0.3,
    f'Total Image Size: {total_size:.1f} MB',
    fontsize=12,
    fontweight='bold',
    color='white',
    ha='center',
    bbox=dict(
        boxstyle='round,pad=0.5',
        facecolor='#2a2a3e',
        edgecolor='white',
        linewidth=2
    )
)

# Add cache optimization insight box
insight_text = (
    "💡 Layer Caching Optimization:\n"
    "Copy requirements.txt BEFORE app.py\n"
    "→ Changing app.py doesn't invalidate pip install cache\n"
    "→ Rebuild time: 2s instead of 30s"
)
ax.text(
    11.5, 4.5,
    insight_text,
    fontsize=9,
    color='white',
    va='center',
    ha='left',
    bbox=dict(
        boxstyle='round,pad=0.5',
        facecolor='#1d4ed8',
        edgecolor='white',
        linewidth=2,
        alpha=0.9
    )
)

# Add "Read-only" and "Writable" labels
ax.text(
    0.3, 6.5,
    'READ\nONLY',
    fontsize=9,
    fontweight='bold',
    color='#cccccc',
    rotation=90,
    va='center',
    ha='center'
)

# Arrow showing layer build order
arrow = FancyArrowPatch(
    (0.3, 1.5),
    (0.3, 6.0),
    arrowstyle='->',
    color='#cccccc',
    linewidth=2,
    mutation_scale=20
)
ax.add_patch(arrow)

ax.text(
    0.1, 3.8,
    'Build\nOrder',
    fontsize=8,
    color='#cccccc',
    rotation=90,
    va='center',
    ha='center'
)

# Set axis limits and remove axes
ax.set_xlim(-0.2, 13.5)
ax.set_ylim(0, 8.5)
ax.axis('off')

# Save figure
plt.tight_layout()
output_path = '../img/ch01-docker-architecture.png'
plt.savefig(output_path, dpi=150, facecolor='#1a1a2e', bbox_inches='tight')
print(f"✅ Saved: {output_path}")

plt.close()
