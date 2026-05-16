#!/usr/bin/env python3
"""
Generate Docker volume mounts diagram showing:
Host filesystem vs container filesystem, with volume persistence.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# Dark theme styling
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(14, 10), facecolor='#1a1a2e')
ax.set_facecolor('#1a1a2e')

# Draw host filesystem section (left)
host_x = 1.5
host_y = 5.0
host_width = 4.5
host_height = 4.0

host_box = FancyBboxPatch(
    (host_x, host_y),
    host_width,
    host_height,
    boxstyle="round,pad=0.15",
    edgecolor='#1e3a8a',
    facecolor='#1a1a2e',
    linewidth=3,
    alpha=0.9
)
ax.add_patch(host_box)

ax.text(
    host_x + host_width / 2, host_y + host_height + 0.3,
    'Host Machine',
    fontsize=14,
    fontweight='bold',
    color='white',
    ha='center'
)

# Host filesystem tree
host_files = [
    ('📁 /home/user/project/', 8.5, '#1e3a8a'),
    ('  📄 app.py', 8.1, '#15803d'),
    ('  📄 Dockerfile', 7.7, '#15803d'),
    ('  📁 data/', 7.3, '#b45309'),
    ('    📄 cache.db (500 MB)', 6.9, '#b91c1c'),
]

for label, y_pos, color in host_files:
    ax.text(
        host_x + 0.3, y_pos,
        label,
        fontsize=9,
        color=color,
        family='monospace',
        va='center'
    )

# Named volume indicator
volume_box = FancyBboxPatch(
    (host_x + 0.5, 5.5),
    3.5,
    1.0,
    boxstyle="round,pad=0.1",
    edgecolor='#f59e0b',
    facecolor='#2a2a3e',
    linewidth=2,
    linestyle='--',
    alpha=0.8
)
ax.add_patch(volume_box)

ax.text(
    host_x + 2.25, 5.8,
    'Named Volume: redis-data',
    fontsize=8,
    fontweight='bold',
    color='#f59e0b',
    ha='center'
)

ax.text(
    host_x + 2.25, 5.5,
    'Managed by Docker',
    fontsize=7,
    color='#cccccc',
    ha='center',
    style='italic'
)

# Draw container filesystem section (right)
container_x = 8.5
container_y = 5.0
container_width = 4.5
container_height = 4.0

container_box = FancyBboxPatch(
    (container_x, container_y),
    container_width,
    container_height,
    boxstyle="round,pad=0.15",
    edgecolor='#15803d',
    facecolor='#1a1a2e',
    linewidth=3,
    alpha=0.9
)
ax.add_patch(container_box)

ax.text(
    container_x + container_width / 2, container_y + container_height + 0.3,
    'Container',
    fontsize=14,
    fontweight='bold',
    color='white',
    ha='center'
)

# Container filesystem tree
container_files = [
    ('📁 /app/', 8.5, '#15803d'),
    ('  📄 app.py', 8.1, '#15803d'),
    ('  📄 __pycache__/', 7.7, '#7c3aed'),
    ('📁 /data/ (mounted)', 7.0, '#f59e0b'),
    ('  📄 cache.db (persists)', 6.6, '#f59e0b'),
]

for label, y_pos, color in container_files:
    ax.text(
        container_x + 0.3, y_pos,
        label,
        fontsize=9,
        color=color,
        family='monospace',
        va='center'
    )

# Mount arrows showing volume binding
# Arrow 1: Host data → Container /data
mount_arrow_1 = FancyArrowPatch(
    (host_x + host_width - 0.5, 6.9),
    (container_x + 0.5, 6.8),
    arrowstyle='<->,head_width=0.3,head_length=0.3',
    color='#f59e0b',
    linewidth=3,
    mutation_scale=20,
    linestyle='--'
)
ax.add_patch(mount_arrow_1)

ax.text(
    6.5, 7.2,
    'Volume Mount\n-v redis-data:/data',
    fontsize=8,
    fontweight='bold',
    color='#f59e0b',
    ha='center',
    family='monospace',
    bbox=dict(
        boxstyle='round,pad=0.3',
        facecolor='#2a2a3e',
        edgecolor='#f59e0b',
        linewidth=2
    )
)

# Arrow 2: Host project → Container /app (bind mount, dev only)
bind_arrow = FancyArrowPatch(
    (host_x + host_width - 0.5, 8.1),
    (container_x + 0.5, 8.1),
    arrowstyle='<->,head_width=0.3,head_length=0.3',
    color='#14b8a6',
    linewidth=2.5,
    mutation_scale=18,
    linestyle=':'
)
ax.add_patch(bind_arrow)

ax.text(
    6.5, 8.6,
    'Bind Mount (dev only)\n-v $(pwd):/app',
    fontsize=8,
    fontweight='bold',
    color='#14b8a6',
    ha='center',
    family='monospace',
    bbox=dict(
        boxstyle='round,pad=0.3',
        facecolor='#2a2a3e',
        edgecolor='#14b8a6',
        linewidth=2
    )
)

# Comparison table
table_y = 3.5
table_data = [
    ('Type', 'Managed by', 'Persists?', 'Use case'),
    ('Named Volume', 'Docker', '✅ Yes', 'Production data'),
    ('Bind Mount', 'Host OS', '⚠️ Tied to host', 'Development only'),
    ('tmpfs', 'Memory', '❌ No', 'Temp secrets'),
]

# Table header
for i, col in enumerate(table_data[0]):
    ax.text(
        2.0 + i * 2.8, table_y + 0.5,
        col,
        fontsize=9,
        fontweight='bold',
        color='white',
        ha='left'
    )

# Table rows
colors = ['#1e3a8a', '#14b8a6', '#7c3aed']
for row_idx, row in enumerate(table_data[1:]):
    y_pos = table_y - (row_idx + 1) * 0.4
    
    # Row background
    row_box = Rectangle(
        (1.5, y_pos - 0.15),
        10.5,
        0.35,
        facecolor='#2a2a3e',
        edgecolor=colors[row_idx],
        linewidth=1.5,
        alpha=0.5
    )
    ax.add_patch(row_box)
    
    for col_idx, cell in enumerate(row):
        ax.text(
            2.0 + col_idx * 2.8, y_pos,
            cell,
            fontsize=8,
            color='white',
            ha='left',
            family='monospace' if col_idx == 0 else None
        )

# Add title
ax.text(
    7.0, 10.2,
    'Docker Volume Mounts',
    fontsize=18,
    fontweight='bold',
    color='white',
    ha='center'
)

ax.text(
    7.0, 9.8,
    'Host Filesystem vs Container Filesystem',
    fontsize=11,
    color='#cccccc',
    ha='center',
    style='italic'
)

# Add key insights
insights = [
    "💡 Volume Persistence:",
    "• Named volumes survive docker rm",
    "• Container filesystem is ephemeral (lost on remove)",
    "• Never use bind mounts in production (breaks portability)",
    "• Database data MUST go in named volumes"
]

insights_text = '\n'.join(insights)
ax.text(
    7.0, 0.8,
    insights_text,
    fontsize=9,
    color='white',
    ha='center',
    va='bottom',
    bbox=dict(
        boxstyle='round,pad=0.5',
        facecolor='#1d4ed8',
        edgecolor='white',
        linewidth=2,
        alpha=0.9
    )
)

# Add example commands
commands = [
    "# Create named volume",
    "docker volume create redis-data",
    "",
    "# Mount volume (production)",
    "docker run -v redis-data:/data redis:7",
    "",
    "# Bind mount (development)",
    "docker run -v $(pwd):/app flask-app:v1"
]

commands_text = '\n'.join(commands)
ax.text(
    11.5, 3.5,
    commands_text,
    fontsize=8,
    color='#15803d',
    ha='left',
    va='top',
    family='monospace',
    bbox=dict(
        boxstyle='round,pad=0.5',
        facecolor='#1a1a2e',
        edgecolor='#15803d',
        linewidth=2,
        alpha=0.9
    )
)

# Add ephemeral warning
ax.text(
    10.5, 5.3,
    '⚠️ Files here are\nlost on docker rm',
    fontsize=7,
    fontweight='bold',
    color='#b91c1c',
    ha='center',
    bbox=dict(
        boxstyle='round,pad=0.2',
        facecolor='#2a2a3e',
        edgecolor='#b91c1c',
        linewidth=1.5
    )
)

# Set axis limits and remove axes
ax.set_xlim(0, 14)
ax.set_ylim(0, 11)
ax.axis('off')

# Save figure
plt.tight_layout()
output_path = '../img/ch01-volume-mounts.png'
plt.savefig(output_path, dpi=150, facecolor='#1a1a2e', bbox_inches='tight')
print(f"✅ Saved: {output_path}")

plt.close()
