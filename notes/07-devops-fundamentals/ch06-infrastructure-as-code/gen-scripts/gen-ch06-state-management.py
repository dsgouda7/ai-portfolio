#!/usr/bin/env python3
"""
Generate Terraform State Management Diagram

Shows local vs remote state backends and their tradeoffs
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
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
ax.text(5, 9.5, 'Terraform State Management', 
        fontsize=24, fontweight='bold', ha='center', va='top',
        color='#ffffff')
ax.text(5, 9.1, 'Local vs Remote State Backends', 
        fontsize=14, ha='center', va='top',
        color='#a0a0a0', style='italic')

# Color palette
color_local = '#b45309'        # Orange for local
color_remote = '#15803d'       # Green for remote
color_danger = '#dc2626'       # Red for warnings
color_text = '#ffffff'
color_box_bg = '#2d3748'

# ============================================================================
# LOCAL STATE (Left side)
# ============================================================================

local_title_y = 8.5
ax.text(2.5, local_title_y, '📁 Local State', ha='center', va='center',
        fontsize=16, fontweight='bold', color=color_local)

# Local state file box
local_box = FancyBboxPatch((1, 7.2), 3, 0.8,
                          boxstyle="round,pad=0.1", 
                          edgecolor=color_local, facecolor=color_box_bg,
                          linewidth=2)
ax.add_patch(local_box)
ax.text(2.5, 7.7, 'terraform.tfstate', ha='center', va='center',
        fontsize=12, fontweight='bold', color=color_text, family='monospace')
ax.text(2.5, 7.4, 'Stored on local disk', ha='center', va='center',
        fontsize=9, color='#a0a0a0')

# Developer laptop
laptop_box = FancyBboxPatch((1, 5.8), 3, 1,
                           boxstyle="round,pad=0.1", 
                           edgecolor='#6366f1', facecolor=color_box_bg,
                           linewidth=2)
ax.add_patch(laptop_box)
ax.text(2.5, 6.5, '💻 Developer Laptop', ha='center', va='center',
        fontsize=11, fontweight='bold', color=color_text)
ax.text(2.5, 6.2, '/my-project/', ha='center', va='center',
        fontsize=9, color='#6366f1', family='monospace')
ax.text(2.5, 5.95, '├── main.tf\n├── terraform.tfstate\n└── .terraform/', 
        ha='center', va='center',
        fontsize=8, color='#a0aec0', family='monospace')

# Arrow: State → Laptop
arrow_local = FancyArrowPatch((2.5, 7.1), (2.5, 6.9),
                             arrowstyle='->', mutation_scale=15,
                             color=color_local, linewidth=2)
ax.add_patch(arrow_local)

# Local pros/cons
pros_box = FancyBboxPatch((1, 4.5), 3, 1.1,
                         boxstyle="round,pad=0.05", 
                         edgecolor='#15803d', facecolor='#1a2d1a',
                         linewidth=1.5)
ax.add_patch(pros_box)
ax.text(1.2, 5.45, '✅ Pros:', ha='left', va='center',
        fontsize=10, fontweight='bold', color='#15803d')
ax.text(1.3, 5.2, '• Simple setup', ha='left', va='center',
        fontsize=9, color='#a0aec0')
ax.text(1.3, 4.95, '• No network dependency', ha='left', va='center',
        fontsize=9, color='#a0aec0')
ax.text(1.3, 4.7, '• Fast operations', ha='left', va='center',
        fontsize=9, color='#a0aec0')

cons_box = FancyBboxPatch((1, 3), 3, 1.3,
                         boxstyle="round,pad=0.05", 
                         edgecolor=color_danger, facecolor='#3f1515',
                         linewidth=1.5)
ax.add_patch(cons_box)
ax.text(1.2, 4.15, '❌ Cons:', ha='left', va='center',
        fontsize=10, fontweight='bold', color=color_danger)
ax.text(1.3, 3.9, '• No team collaboration', ha='left', va='center',
        fontsize=9, color='#a0aec0')
ax.text(1.3, 3.65, '• No state locking', ha='left', va='center',
        fontsize=9, color='#a0aec0')
ax.text(1.3, 3.4, '• Risk of loss/corruption', ha='left', va='center',
        fontsize=9, color='#a0aec0')
ax.text(1.3, 3.15, '• No version history', ha='left', va='center',
        fontsize=9, color='#a0aec0')

# Warning badge
warning_badge = FancyBboxPatch((1, 2.4), 3, 0.5,
                              boxstyle="round,pad=0.05", 
                              edgecolor=color_danger, facecolor=color_danger,
                              linewidth=2, alpha=0.2)
ax.add_patch(warning_badge)
ax.text(2.5, 2.65, '⚠️ NOT for teams!', ha='center', va='center',
        fontsize=11, fontweight='bold', color=color_danger)

# ============================================================================
# REMOTE STATE (Right side)
# ============================================================================

remote_title_y = 8.5
ax.text(7.5, remote_title_y, '☁️ Remote State', ha='center', va='center',
        fontsize=16, fontweight='bold', color=color_remote)

# Remote backend box (cloud)
remote_box = FancyBboxPatch((6, 7.2), 3, 0.8,
                           boxstyle="round,pad=0.1", 
                           edgecolor=color_remote, facecolor=color_box_bg,
                           linewidth=2)
ax.add_patch(remote_box)
ax.text(7.5, 7.7, 'Remote Backend', ha='center', va='center',
        fontsize=12, fontweight='bold', color=color_text)
ax.text(7.5, 7.4, 'S3 / Azure Blob / GCS', ha='center', va='center',
        fontsize=9, color='#a0aec0')

# Multiple developers
dev1_box = FancyBboxPatch((5.5, 5.8), 1.2, 0.7,
                         boxstyle="round,pad=0.05", 
                         edgecolor='#6366f1', facecolor=color_box_bg,
                         linewidth=1.5)
ax.add_patch(dev1_box)
ax.text(6.1, 6.3, '💻 Dev 1', ha='center', va='center',
        fontsize=9, fontweight='bold', color=color_text)
ax.text(6.1, 6.05, 'Alice', ha='center', va='center',
        fontsize=8, color='#a0aec0')

dev2_box = FancyBboxPatch((7, 5.8), 1.2, 0.7,
                         boxstyle="round,pad=0.05", 
                         edgecolor='#6366f1', facecolor=color_box_bg,
                         linewidth=1.5)
ax.add_patch(dev2_box)
ax.text(7.6, 6.3, '💻 Dev 2', ha='center', va='center',
        fontsize=9, fontweight='bold', color=color_text)
ax.text(7.6, 6.05, 'Bob', ha='center', va='center',
        fontsize=8, color='#a0aec0')

dev3_box = FancyBboxPatch((8.5, 5.8), 1.2, 0.7,
                         boxstyle="round,pad=0.05", 
                         edgecolor='#6366f1', facecolor=color_box_bg,
                         linewidth=1.5)
ax.add_patch(dev3_box)
ax.text(9.1, 6.3, '🤖 CI/CD', ha='center', va='center',
        fontsize=9, fontweight='bold', color=color_text)
ax.text(9.1, 6.05, 'Pipeline', ha='center', va='center',
        fontsize=8, color='#a0aec0')

# Arrows: Devs → Remote backend
arrow_dev1 = FancyArrowPatch((6.1, 6.5), (7.2, 7.1),
                            arrowstyle='<->', mutation_scale=15,
                            color=color_remote, linewidth=2)
ax.add_patch(arrow_dev1)

arrow_dev2 = FancyArrowPatch((7.6, 6.5), (7.5, 7.1),
                            arrowstyle='<->', mutation_scale=15,
                            color=color_remote, linewidth=2)
ax.add_patch(arrow_dev2)

arrow_dev3 = FancyArrowPatch((9.1, 6.5), (7.8, 7.1),
                            arrowstyle='<->', mutation_scale=15,
                            color=color_remote, linewidth=2)
ax.add_patch(arrow_dev3)

# State locking indicator
lock_box = FancyBboxPatch((6.5, 6.8), 2, 0.3,
                         boxstyle="round,pad=0.05", 
                         edgecolor='#eab308', facecolor='#3f3710',
                         linewidth=1.5)
ax.add_patch(lock_box)
ax.text(7.5, 6.95, '🔒 State Locking', ha='center', va='center',
        fontsize=9, fontweight='bold', color='#eab308')

# Remote pros
pros_box2 = FancyBboxPatch((6, 4.1), 3, 1.5,
                          boxstyle="round,pad=0.05", 
                          edgecolor=color_remote, facecolor='#1a2d1a',
                          linewidth=1.5)
ax.add_patch(pros_box2)
ax.text(6.2, 5.45, '✅ Pros:', ha='left', va='center',
        fontsize=10, fontweight='bold', color=color_remote)
ax.text(6.3, 5.2, '• Team collaboration', ha='left', va='center',
        fontsize=9, color='#a0aec0')
ax.text(6.3, 4.95, '• State locking (no conflicts)', ha='left', va='center',
        fontsize=9, color='#a0aec0')
ax.text(6.3, 4.7, '• Encrypted at rest', ha='left', va='center',
        fontsize=9, color='#a0aec0')
ax.text(6.3, 4.45, '• Version history', ha='left', va='center',
        fontsize=9, color='#a0aec0')
ax.text(6.3, 4.2, '• Automated backups', ha='left', va='center',
        fontsize=9, color='#a0aec0')

# Remote cons
cons_box2 = FancyBboxPatch((6, 3), 3, 0.9,
                          boxstyle="round,pad=0.05", 
                          edgecolor='#b45309', facecolor='#2d2416',
                          linewidth=1.5)
ax.add_patch(cons_box2)
ax.text(6.2, 3.75, '⚠️ Cons:', ha='left', va='center',
        fontsize=10, fontweight='bold', color='#b45309')
ax.text(6.3, 3.5, '• Setup complexity', ha='left', va='center',
        fontsize=9, color='#a0aec0')
ax.text(6.3, 3.25, '• Network dependency', ha='left', va='center',
        fontsize=9, color='#a0aec0')

# Recommended badge
rec_badge = FancyBboxPatch((6, 2.4), 3, 0.5,
                          boxstyle="round,pad=0.05", 
                          edgecolor=color_remote, facecolor=color_remote,
                          linewidth=2, alpha=0.2)
ax.add_patch(rec_badge)
ax.text(7.5, 2.65, '✅ Recommended for teams!', ha='center', va='center',
        fontsize=11, fontweight='bold', color=color_remote)

# ============================================================================
# Bottom: State Operations
# ============================================================================

operations_box = FancyBboxPatch((0.5, 0.3), 9, 1.7,
                               boxstyle="round,pad=0.1", 
                               edgecolor='#6366f1', facecolor='#1e1e3f',
                               linewidth=2)
ax.add_patch(operations_box)
ax.text(5, 1.85, 'State Operations', ha='center', va='center',
        fontsize=12, fontweight='bold', color='#6366f1')

# Commands
commands = [
    ('terraform state list', 'List all resources in state'),
    ('terraform state show <resource>', 'Show details of a resource'),
    ('terraform state mv <src> <dst>', 'Move/rename a resource'),
    ('terraform state rm <resource>', 'Remove resource from state'),
    ('terraform state pull', 'Download remote state'),
    ('terraform state push', 'Upload state to remote backend'),
    ('terraform refresh', 'Sync state with reality'),
    ('terraform import <resource> <id>', 'Import existing resource'),
]

cmd_y_start = 1.6
cmd_spacing = 0.18
for i, (cmd, desc) in enumerate(commands):
    y = cmd_y_start - (i * cmd_spacing)
    ax.text(0.7, y, cmd, ha='left', va='center',
           fontsize=8, fontweight='bold', color='#a0aec0', family='monospace')
    ax.text(3.5, y, desc, ha='left', va='center',
           fontsize=7, color='#718096')

# Warning
warning_text = "⚠️ Never edit terraform.tfstate manually! Use 'terraform state' commands only."
ax.text(5, 0.5, warning_text, ha='center', va='center',
        fontsize=9, fontweight='bold', color=color_danger,
        bbox=dict(boxstyle='round,pad=0.3', edgecolor=color_danger, facecolor='#3f1515'))

# Divider line
ax.plot([5, 5], [2.2, 8.8], color='#4a5568', linewidth=2, linestyle='--', alpha=0.5)

# Save
output_path = '../img/ch06-state-management.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
print(f"✅ Diagram saved: {output_path}")
plt.close()
