#!/usr/bin/env python3
"""
Generate Terraform Resource Graph Diagram

Shows dependency graph visualization for Terraform resources
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib
matplotlib.use('Agg')  # Headless backend

# Set dark background style
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(14, 10), facecolor='#1a1a2e')
ax.set_facecolor('#1a1a2e')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9.5, 'Terraform Resource Dependency Graph', 
        fontsize=24, fontweight='bold', ha='center', va='top',
        color='#ffffff')
ax.text(5, 9.1, 'Example: Web Application Stack', 
        fontsize=14, ha='center', va='top',
        color='#a0a0a0', style='italic')

# Color palette
color_network = '#1d4ed8'      # Blue for network
color_storage = '#b45309'      # Orange for storage
color_compute = '#15803d'      # Green for compute
color_database = '#9333ea'     # Purple for database
color_lb = '#dc2626'           # Red for load balancer
color_text = '#ffffff'
color_box_bg = '#2d3748'

# Level 1: Network (foundation)
network_x, network_y = 5, 8
network_box = FancyBboxPatch((network_x - 1, network_y - 0.4), 2, 0.8,
                            boxstyle="round,pad=0.1", 
                            edgecolor=color_network, facecolor=color_box_bg,
                            linewidth=3)
ax.add_patch(network_box)
ax.text(network_x, network_y + 0.15, 'docker_network', ha='center', va='center',
        fontsize=12, fontweight='bold', color=color_text)
ax.text(network_x, network_y - 0.15, '"app_network"', ha='center', va='center',
        fontsize=10, color=color_network, family='monospace')

# Level 2: Storage (depends on network)
storage_x, storage_y = 2.5, 6
storage_box = FancyBboxPatch((storage_x - 0.9, storage_y - 0.4), 1.8, 0.8,
                            boxstyle="round,pad=0.1", 
                            edgecolor=color_storage, facecolor=color_box_bg,
                            linewidth=2)
ax.add_patch(storage_box)
ax.text(storage_x, storage_y + 0.15, 'docker_volume', ha='center', va='center',
        fontsize=12, fontweight='bold', color=color_text)
ax.text(storage_x, storage_y - 0.15, '"db_data"', ha='center', va='center',
        fontsize=10, color=color_storage, family='monospace')

# Level 2: Database (depends on network and storage)
db_x, db_y = 5, 6
db_box = FancyBboxPatch((db_x - 0.9, db_y - 0.4), 1.8, 0.8,
                       boxstyle="round,pad=0.1", 
                       edgecolor=color_database, facecolor=color_box_bg,
                       linewidth=2)
ax.add_patch(db_box)
ax.text(db_x, db_y + 0.15, 'docker_container', ha='center', va='center',
        fontsize=12, fontweight='bold', color=color_text)
ax.text(db_x, db_y - 0.15, '"postgres"', ha='center', va='center',
        fontsize=10, color=color_database, family='monospace')

# Level 3: Web containers (depend on database and network)
web1_x, web1_y = 2, 4
web1_box = FancyBboxPatch((web1_x - 0.8, web1_y - 0.4), 1.6, 0.8,
                         boxstyle="round,pad=0.1", 
                         edgecolor=color_compute, facecolor=color_box_bg,
                         linewidth=2)
ax.add_patch(web1_box)
ax.text(web1_x, web1_y + 0.15, 'docker_container', ha='center', va='center',
        fontsize=12, fontweight='bold', color=color_text)
ax.text(web1_x, web1_y - 0.15, '"web1"', ha='center', va='center',
        fontsize=10, color=color_compute, family='monospace')

web2_x, web2_y = 4.5, 4
web2_box = FancyBboxPatch((web2_x - 0.8, web2_y - 0.4), 1.6, 0.8,
                         boxstyle="round,pad=0.1", 
                         edgecolor=color_compute, facecolor=color_box_bg,
                         linewidth=2)
ax.add_patch(web2_box)
ax.text(web2_x, web2_y + 0.15, 'docker_container', ha='center', va='center',
        fontsize=12, fontweight='bold', color=color_text)
ax.text(web2_x, web2_y - 0.15, '"web2"', ha='center', va='center',
        fontsize=10, color=color_compute, family='monospace')

web3_x, web3_y = 7, 4
web3_box = FancyBboxPatch((web3_x - 0.8, web3_y - 0.4), 1.6, 0.8,
                         boxstyle="round,pad=0.1", 
                         edgecolor=color_compute, facecolor=color_box_bg,
                         linewidth=2)
ax.add_patch(web3_box)
ax.text(web3_x, web3_y + 0.15, 'docker_container', ha='center', va='center',
        fontsize=12, fontweight='bold', color=color_text)
ax.text(web3_x, web3_y - 0.15, '"web3"', ha='center', va='center',
        fontsize=10, color=color_compute, family='monospace')

# Level 4: Load balancer (depends on web containers)
lb_x, lb_y = 4.5, 2
lb_box = FancyBboxPatch((lb_x - 0.9, lb_y - 0.4), 1.8, 0.8,
                       boxstyle="round,pad=0.1", 
                       edgecolor=color_lb, facecolor=color_box_bg,
                       linewidth=2)
ax.add_patch(lb_box)
ax.text(lb_x, lb_y + 0.15, 'docker_container', ha='center', va='center',
        fontsize=12, fontweight='bold', color=color_text)
ax.text(lb_x, lb_y - 0.15, '"nginx_lb"', ha='center', va='center',
        fontsize=10, color=color_lb, family='monospace')

# Dependencies: Network → All others
arrow_net_storage = FancyArrowPatch((network_x - 0.7, network_y - 0.5), 
                                    (storage_x + 0.4, storage_y + 0.5),
                                    arrowstyle='->', mutation_scale=15,
                                    color=color_network, linewidth=2, alpha=0.7)
ax.add_patch(arrow_net_storage)

arrow_net_db = FancyArrowPatch((network_x, network_y - 0.5), 
                              (db_x, db_y + 0.5),
                              arrowstyle='->', mutation_scale=15,
                              color=color_network, linewidth=2, alpha=0.7)
ax.add_patch(arrow_net_db)

# Dependencies: Storage → Database
arrow_storage_db = FancyArrowPatch((storage_x + 0.9, storage_y), 
                                  (db_x - 0.9, db_y),
                                  arrowstyle='->', mutation_scale=15,
                                  color=color_storage, linewidth=2, alpha=0.7)
ax.add_patch(arrow_storage_db)

# Dependencies: Database → Web containers
arrow_db_web1 = FancyArrowPatch((db_x - 0.7, db_y - 0.5), 
                               (web1_x + 0.4, web1_y + 0.5),
                               arrowstyle='->', mutation_scale=15,
                               color=color_database, linewidth=2, alpha=0.7)
ax.add_patch(arrow_db_web1)

arrow_db_web2 = FancyArrowPatch((db_x, db_y - 0.5), 
                               (web2_x, web2_y + 0.5),
                               arrowstyle='->', mutation_scale=15,
                               color=color_database, linewidth=2, alpha=0.7)
ax.add_patch(arrow_db_web2)

arrow_db_web3 = FancyArrowPatch((db_x + 0.7, db_y - 0.5), 
                               (web3_x - 0.4, web3_y + 0.5),
                               arrowstyle='->', mutation_scale=15,
                               color=color_database, linewidth=2, alpha=0.7)
ax.add_patch(arrow_db_web3)

# Dependencies: Network → Web containers (direct)
arrow_net_web1 = FancyArrowPatch((network_x - 0.9, network_y - 0.5), 
                                (web1_x + 0.1, web1_y + 0.5),
                                arrowstyle='->', mutation_scale=15,
                                color=color_network, linewidth=1.5, alpha=0.4, linestyle='--')
ax.add_patch(arrow_net_web1)

arrow_net_web2 = FancyArrowPatch((network_x - 0.2, network_y - 0.5), 
                                (web2_x, web2_y + 0.5),
                                arrowstyle='->', mutation_scale=15,
                                color=color_network, linewidth=1.5, alpha=0.4, linestyle='--')
ax.add_patch(arrow_net_web2)

arrow_net_web3 = FancyArrowPatch((network_x + 0.5, network_y - 0.5), 
                                (web3_x - 0.1, web3_y + 0.5),
                                arrowstyle='->', mutation_scale=15,
                                color=color_network, linewidth=1.5, alpha=0.4, linestyle='--')
ax.add_patch(arrow_net_web3)

# Dependencies: Web containers → Load balancer
arrow_web1_lb = FancyArrowPatch((web1_x + 0.5, web1_y - 0.5), 
                               (lb_x - 0.4, lb_y + 0.5),
                               arrowstyle='->', mutation_scale=15,
                               color=color_compute, linewidth=2, alpha=0.7)
ax.add_patch(arrow_web1_lb)

arrow_web2_lb = FancyArrowPatch((web2_x, web2_y - 0.5), 
                               (lb_x, lb_y + 0.5),
                               arrowstyle='->', mutation_scale=15,
                               color=color_compute, linewidth=2, alpha=0.7)
ax.add_patch(arrow_web2_lb)

arrow_web3_lb = FancyArrowPatch((web3_x - 0.5, web3_y - 0.5), 
                               (lb_x + 0.4, lb_y + 0.5),
                               arrowstyle='->', mutation_scale=15,
                               color=color_compute, linewidth=2, alpha=0.7)
ax.add_patch(arrow_web3_lb)

# Execution order annotations
ax.text(9, 8, 'Execution Order', ha='center', va='top',
        fontsize=12, fontweight='bold', color='#ffffff')

# Level badges
level_colors = ['#1d4ed8', '#b45309', '#15803d', '#dc2626']
level_labels = ['Level 1', 'Level 2', 'Level 3', 'Level 4']
level_y_start = 7.5
for i, (color, label) in enumerate(zip(level_colors, level_labels)):
    circle = Circle((8.5, level_y_start - i * 0.5), 0.15, 
                   color=color, ec='white', linewidth=1.5)
    ax.add_patch(circle)
    ax.text(9, level_y_start - i * 0.5, label, ha='left', va='center',
           fontsize=10, color='#a0a0a0')

# Parallelism note
parallel_box = FancyBboxPatch((0.3, 0.3), 3.8, 1.8,
                             boxstyle="round,pad=0.1", 
                             edgecolor='#6366f1', facecolor='#1e1e3f',
                             linewidth=1.5)
ax.add_patch(parallel_box)
ax.text(2.2, 1.95, '⚡ Parallel Execution', ha='center', va='top',
        fontsize=11, fontweight='bold', color='#6366f1')
ax.text(0.5, 1.7, 'Resources at the same level', ha='left', va='top',
        fontsize=9, color='#a0aec0')
ax.text(0.5, 1.5, 'with no dependencies can be', ha='left', va='top',
        fontsize=9, color='#a0aec0')
ax.text(0.5, 1.3, 'created in parallel.', ha='left', va='top',
        fontsize=9, color='#a0aec0')
ax.text(0.5, 1, 'Example: web1, web2, web3', ha='left', va='top',
        fontsize=9, fontweight='bold', color=color_compute, family='monospace')
ax.text(0.5, 0.8, 'are created simultaneously', ha='left', va='top',
        fontsize=9, color='#a0aec0')
ax.text(0.5, 0.5, '(3x faster than sequential)', ha='left', va='top',
        fontsize=9, color='#6366f1', style='italic')

# Legend
legend_box = FancyBboxPatch((5, 0.3), 4.7, 1.3,
                           boxstyle="round,pad=0.1", 
                           edgecolor='#718096', facecolor='#1e1e3f',
                           linewidth=1.5)
ax.add_patch(legend_box)
ax.text(7.35, 1.45, 'Resource Types', ha='center', va='top',
        fontsize=11, fontweight='bold', color='#ffffff')

legend_items = [
    (color_network, 'Network'),
    (color_storage, 'Storage'),
    (color_database, 'Database'),
    (color_compute, 'Compute'),
    (color_lb, 'Load Balancer')
]

legend_x_start = 5.3
legend_y = 1.1
for i, (color, label) in enumerate(legend_items):
    if i < 3:
        x = legend_x_start + (i * 1.5)
        y = legend_y
    else:
        x = legend_x_start + ((i - 3) * 1.5) + 0.75
        y = legend_y - 0.35
    
    circle = Circle((x, y), 0.1, color=color, ec='white', linewidth=1)
    ax.add_patch(circle)
    ax.text(x + 0.15, y, label, ha='left', va='center',
           fontsize=8, color='#a0aec0')

# Arrow legend
ax.text(5.3, 0.5, '━━▶', ha='left', va='center',
        fontsize=12, color='#ffffff')
ax.text(5.6, 0.5, 'Explicit dependency', ha='left', va='center',
        fontsize=8, color='#a0aec0')

ax.text(7.5, 0.5, '╍╍▶', ha='left', va='center',
        fontsize=12, color='#ffffff', alpha=0.6)
ax.text(7.8, 0.5, 'Implicit dependency', ha='left', va='center',
        fontsize=8, color='#a0aec0')

# Save
output_path = '../img/ch06-resource-graph.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
print(f"✅ Diagram saved: {output_path}")
plt.close()
