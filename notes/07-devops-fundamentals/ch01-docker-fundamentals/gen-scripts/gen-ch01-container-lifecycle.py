#!/usr/bin/env python3
"""
Generate Docker container lifecycle animation showing:
Build → Run → Stop → Remove flow with state transitions.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

# Dark theme styling
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(14, 8), facecolor='#1a1a2e')
ax.set_facecolor('#1a1a2e')

# Lifecycle states (left to right)
states = [
    {
        'label': 'Dockerfile',
        'x': 1.5,
        'y': 4.0,
        'color': '#1e3a8a',
        'icon': '📄',
        'description': 'Build instructions'
    },
    {
        'label': 'Image',
        'x': 4.5,
        'y': 4.0,
        'color': '#15803d',
        'icon': '📦',
        'description': 'Read-only template'
    },
    {
        'label': 'Container\n(Running)',
        'x': 7.5,
        'y': 4.0,
        'color': '#b45309',
        'icon': '▶️',
        'description': 'Active instance'
    },
    {
        'label': 'Container\n(Stopped)',
        'x': 10.5,
        'y': 4.0,
        'color': '#7c3aed',
        'icon': '⏸️',
        'description': 'Paused, data retained'
    },
    {
        'label': 'Removed',
        'x': 13.5,
        'y': 4.0,
        'color': '#b91c1c',
        'icon': '🗑️',
        'description': 'Deleted, data lost'
    }
]

# Draw state boxes
for state in states:
    # Main state box
    box = FancyBboxPatch(
        (state['x'] - 0.8, state['y'] - 0.6),
        1.6,
        1.2,
        boxstyle="round,pad=0.1",
        edgecolor='white',
        facecolor=state['color'],
        linewidth=2.5,
        alpha=0.9
    )
    ax.add_patch(box)
    
    # Icon
    ax.text(
        state['x'], state['y'] + 0.3,
        state['icon'],
        fontsize=24,
        ha='center',
        va='center'
    )
    
    # Label
    ax.text(
        state['x'], state['y'] - 0.1,
        state['label'],
        fontsize=10,
        fontweight='bold',
        color='white',
        ha='center',
        va='center'
    )
    
    # Description below box
    ax.text(
        state['x'], state['y'] - 1.0,
        state['description'],
        fontsize=8,
        color='#cccccc',
        ha='center',
        va='top',
        style='italic'
    )

# Draw arrows with command labels
transitions = [
    {
        'from': states[0],
        'to': states[1],
        'command': 'docker build',
        'color': '#15803d',
        'y_offset': 0.1
    },
    {
        'from': states[1],
        'to': states[2],
        'command': 'docker run',
        'color': '#b45309',
        'y_offset': 0.1
    },
    {
        'from': states[2],
        'to': states[3],
        'command': 'docker stop',
        'color': '#7c3aed',
        'y_offset': 0.1
    },
    {
        'from': states[3],
        'to': states[4],
        'command': 'docker rm',
        'color': '#b91c1c',
        'y_offset': 0.1
    }
]

for trans in transitions:
    # Arrow
    arrow = FancyArrowPatch(
        (trans['from']['x'] + 0.8, trans['from']['y'] + trans['y_offset']),
        (trans['to']['x'] - 0.8, trans['to']['y'] + trans['y_offset']),
        arrowstyle='->,head_width=0.4,head_length=0.4',
        color=trans['color'],
        linewidth=3,
        mutation_scale=20
    )
    ax.add_patch(arrow)
    
    # Command label above arrow
    mid_x = (trans['from']['x'] + trans['to']['x']) / 2
    ax.text(
        mid_x, trans['from']['y'] + 0.75,
        trans['command'],
        fontsize=9,
        fontweight='bold',
        color='white',
        ha='center',
        va='bottom',
        family='monospace',
        bbox=dict(
            boxstyle='round,pad=0.3',
            facecolor='#2a2a3e',
            edgecolor=trans['color'],
            linewidth=2
        )
    )

# Add restart loop (stopped → running)
restart_arrow = FancyArrowPatch(
    (states[3]['x'] - 0.3, states[3]['y'] - 0.7),
    (states[2]['x'] + 0.3, states[2]['y'] - 0.7),
    arrowstyle='->,head_width=0.3,head_length=0.3',
    color='#14b8a6',  # Teal
    linewidth=2.5,
    linestyle='--',
    mutation_scale=15,
    connectionstyle="arc3,rad=-.3"
)
ax.add_patch(restart_arrow)

ax.text(
    9.0, 2.3,
    'docker start',
    fontsize=8,
    fontweight='bold',
    color='#14b8a6',
    ha='center',
    family='monospace',
    bbox=dict(
        boxstyle='round,pad=0.2',
        facecolor='#2a2a3e',
        edgecolor='#14b8a6',
        linewidth=1.5
    )
)

# Add multiple container instances from single image
instance_arrow = FancyArrowPatch(
    (states[1]['x'], states[1]['y'] - 0.7),
    (states[1]['x'], states[1]['y'] - 1.8),
    arrowstyle='->,head_width=0.3,head_length=0.3',
    color='#f59e0b',  # Amber
    linewidth=2,
    mutation_scale=15
)
ax.add_patch(instance_arrow)

# Mini containers below image
mini_containers_y = 1.5
for i, offset_x in enumerate([-0.6, 0, 0.6]):
    mini_box = FancyBboxPatch(
        (states[1]['x'] + offset_x - 0.2, mini_containers_y - 0.15),
        0.4,
        0.3,
        boxstyle="round,pad=0.02",
        edgecolor='#f59e0b',
        facecolor='#2a2a3e',
        linewidth=1.5,
        alpha=0.8
    )
    ax.add_patch(mini_box)
    ax.text(
        states[1]['x'] + offset_x, mini_containers_y,
        f'C{i+1}',
        fontsize=7,
        color='white',
        ha='center',
        va='center'
    )

ax.text(
    states[1]['x'], 0.9,
    'One image → Many containers',
    fontsize=8,
    color='#f59e0b',
    ha='center',
    style='italic'
)

# Add title
ax.text(
    7.5, 6.5,
    'Docker Container Lifecycle',
    fontsize=18,
    fontweight='bold',
    color='white',
    ha='center'
)

ax.text(
    7.5, 6.0,
    'Image vs Container: Blueprint vs Running Instance',
    fontsize=11,
    color='#cccccc',
    ha='center',
    style='italic'
)

# Add key insights box
insights = [
    "💡 Key Insights:",
    "• Images are immutable — never modify a running container",
    "• Stopped containers retain data (until removed)",
    "• docker rm -f = stop + remove in one command",
    "• Use --rm flag for auto-cleanup: docker run --rm"
]

insights_text = '\n'.join(insights)
ax.text(
    7.5, 0.3,
    insights_text,
    fontsize=8,
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

# Add persistence warning
ax.text(
    13.5, 2.5,
    '⚠️ Data lost\nforever',
    fontsize=8,
    fontweight='bold',
    color='#b91c1c',
    ha='center',
    bbox=dict(
        boxstyle='round,pad=0.3',
        facecolor='#2a2a3e',
        edgecolor='#b91c1c',
        linewidth=2
    )
)

# Set axis limits and remove axes
ax.set_xlim(0, 15)
ax.set_ylim(0, 7)
ax.axis('off')

# Save figure
plt.tight_layout()
output_path = '../img/ch01-container-lifecycle.png'
plt.savefig(output_path, dpi=150, facecolor='#1a1a2e', bbox_inches='tight')
print(f"✅ Saved: {output_path}")

plt.close()
