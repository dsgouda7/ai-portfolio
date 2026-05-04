#!/usr/bin/env python3
"""
Generate Terraform Workflow Diagram

Shows the flow: Write → Plan → Apply → State
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
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
ax.text(5, 9.5, 'Terraform Workflow', 
        fontsize=24, fontweight='bold', ha='center', va='top',
        color='#ffffff')

# Color palette
color_primary = '#1e3a8a'      # Primary blue
color_success = '#15803d'      # Success green
color_caution = '#b45309'      # Caution orange
color_info = '#1d4ed8'         # Info blue
color_text = '#ffffff'         # White text
color_box_bg = '#2d3748'       # Box background

# Step 1: Write .tf files (top)
write_x, write_y = 2, 8
write_box = FancyBboxPatch((write_x - 0.8, write_y - 0.5), 1.6, 1,
                           boxstyle="round,pad=0.1", 
                           edgecolor=color_info, facecolor=color_box_bg,
                           linewidth=2)
ax.add_patch(write_box)
ax.text(write_x, write_y + 0.2, '1. Write', ha='center', va='center',
        fontsize=14, fontweight='bold', color=color_text)
ax.text(write_x, write_y - 0.1, '.tf files', ha='center', va='center',
        fontsize=10, color=color_text)

# File icons
ax.text(write_x, write_y - 0.7, 'main.tf\nvariables.tf\noutputs.tf', 
        ha='center', va='top', fontsize=8, color='#a0a0a0')

# Step 2: terraform init (middle left)
init_x, init_y = 2, 6
init_box = FancyBboxPatch((init_x - 0.8, init_y - 0.5), 1.6, 1,
                          boxstyle="round,pad=0.1", 
                          edgecolor=color_primary, facecolor=color_box_bg,
                          linewidth=2)
ax.add_patch(init_box)
ax.text(init_x, init_y + 0.2, '2. Init', ha='center', va='center',
        fontsize=14, fontweight='bold', color=color_text)
ax.text(init_x, init_y - 0.1, 'terraform init', ha='center', va='center',
        fontsize=10, color=color_text)

# Download providers note
ax.text(init_x, init_y - 0.7, 'Download providers\nInitialize backend', 
        ha='center', va='top', fontsize=8, color='#a0a0a0')

# Arrow: Write → Init
arrow1 = FancyArrowPatch((write_x, write_y - 0.6), (init_x, init_y + 0.6),
                         arrowstyle='->', mutation_scale=20,
                         color=color_info, linewidth=2)
ax.add_patch(arrow1)

# Step 3: terraform plan (middle center)
plan_x, plan_y = 5, 6
plan_box = FancyBboxPatch((plan_x - 0.8, plan_y - 0.5), 1.6, 1,
                          boxstyle="round,pad=0.1", 
                          edgecolor=color_caution, facecolor=color_box_bg,
                          linewidth=2)
ax.add_patch(plan_box)
ax.text(plan_x, plan_y + 0.2, '3. Plan', ha='center', va='center',
        fontsize=14, fontweight='bold', color=color_text)
ax.text(plan_x, plan_y - 0.1, 'terraform plan', ha='center', va='center',
        fontsize=10, color=color_text)

# Plan details
ax.text(plan_x, plan_y - 0.7, 'Preview changes\n+ create | ~ update | - destroy', 
        ha='center', va='top', fontsize=8, color='#a0a0a0')

# Arrow: Init → Plan
arrow2 = FancyArrowPatch((init_x + 0.9, init_y), (plan_x - 0.9, plan_y),
                         arrowstyle='->', mutation_scale=20,
                         color=color_primary, linewidth=2)
ax.add_patch(arrow2)

# Step 4: terraform apply (middle right)
apply_x, apply_y = 8, 6
apply_box = FancyBboxPatch((apply_x - 0.8, apply_y - 0.5), 1.6, 1,
                           boxstyle="round,pad=0.1", 
                           edgecolor=color_success, facecolor=color_box_bg,
                           linewidth=2)
ax.add_patch(apply_box)
ax.text(apply_x, apply_y + 0.2, '4. Apply', ha='center', va='center',
        fontsize=14, fontweight='bold', color=color_text)
ax.text(apply_x, apply_y - 0.1, 'terraform apply', ha='center', va='center',
        fontsize=10, color=color_text)

# Apply details
ax.text(apply_x, apply_y - 0.7, 'Provision resources\nUpdate state', 
        ha='center', va='top', fontsize=8, color='#a0a0a0')

# Arrow: Plan → Apply
arrow3 = FancyArrowPatch((plan_x + 0.9, plan_y), (apply_x - 0.9, apply_y),
                         arrowstyle='->', mutation_scale=20,
                         color=color_caution, linewidth=2)
ax.add_patch(arrow3)

# State file (bottom)
state_x, state_y = 8, 4
state_box = FancyBboxPatch((state_x - 1, state_y - 0.6), 2, 1.2,
                           boxstyle="round,pad=0.1", 
                           edgecolor='#9333ea', facecolor='#2d1b4e',
                           linewidth=2)
ax.add_patch(state_box)
ax.text(state_x, state_y + 0.3, 'State File', ha='center', va='center',
        fontsize=14, fontweight='bold', color=color_text)
ax.text(state_x, state_y, 'terraform.tfstate', ha='center', va='center',
        fontsize=10, color='#a0a0a0')
ax.text(state_x, state_y - 0.3, 'Records managed resources', ha='center', va='center',
        fontsize=9, color='#a0a0a0')

# Arrow: Apply → State
arrow4 = FancyArrowPatch((apply_x, apply_y - 0.6), (state_x, state_y + 0.7),
                         arrowstyle='->', mutation_scale=20,
                         color=color_success, linewidth=2)
ax.add_patch(arrow4)

# Arrow: State → Plan (feedback loop)
arrow5 = FancyArrowPatch((state_x - 1.1, state_y + 0.5), (plan_x + 0.5, plan_y - 0.6),
                         arrowstyle='->', mutation_scale=15,
                         color='#9333ea', linewidth=2, linestyle='--')
ax.add_patch(arrow5)
ax.text(6.5, 4.8, 'Read current state', ha='center', va='center',
        fontsize=8, color='#9333ea', style='italic')

# Infrastructure (right side)
infra_x, infra_y = 8, 1.5
infra_box = FancyBboxPatch((infra_x - 1, infra_y - 0.6), 2, 1.2,
                          boxstyle="round,pad=0.1", 
                          edgecolor=color_success, facecolor=color_box_bg,
                          linewidth=2)
ax.add_patch(infra_box)
ax.text(infra_x, infra_y + 0.3, 'Infrastructure', ha='center', va='center',
        fontsize=14, fontweight='bold', color=color_text)
ax.text(infra_x, infra_y - 0.1, '☁️ 🐳 🗄️', ha='center', va='center',
        fontsize=16)
ax.text(infra_x, infra_y - 0.4, 'Provisioned resources', ha='center', va='center',
        fontsize=9, color='#a0a0a0')

# Arrow: Apply → Infrastructure
arrow6 = FancyArrowPatch((apply_x, apply_y - 0.6), (infra_x, infra_y + 0.7),
                         arrowstyle='->', mutation_scale=20,
                         color=color_success, linewidth=3)
ax.add_patch(arrow6)

# Destroy path (optional)
destroy_x, destroy_y = 5, 1.5
destroy_box = FancyBboxPatch((destroy_x - 0.8, destroy_y - 0.4), 1.6, 0.8,
                            boxstyle="round,pad=0.1", 
                            edgecolor='#dc2626', facecolor='#3f1515',
                            linewidth=2, linestyle='--')
ax.add_patch(destroy_box)
ax.text(destroy_x, destroy_y + 0.1, 'terraform destroy', ha='center', va='center',
        fontsize=10, color='#dc2626', fontweight='bold')
ax.text(destroy_x, destroy_y - 0.2, 'Remove all resources', ha='center', va='center',
        fontsize=8, color='#dc2626')

# Arrow: Destroy → Infrastructure (removal)
arrow7 = FancyArrowPatch((destroy_x + 0.9, destroy_y), (infra_x - 1.1, infra_y),
                         arrowstyle='->', mutation_scale=20,
                         color='#dc2626', linewidth=2, linestyle='--')
ax.add_patch(arrow7)

# Annotations
# Declarative vs Imperative
ax.text(0.5, 8.5, '💡 Declarative', ha='left', va='top',
        fontsize=10, fontweight='bold', color=color_info)
ax.text(0.5, 8.2, 'Describe desired state\nTerraform calculates diff', 
        ha='left', va='top', fontsize=8, color='#a0a0a0')

# Idempotent
ax.text(0.5, 7, '🔄 Idempotent', ha='left', va='top',
        fontsize=10, fontweight='bold', color=color_success)
ax.text(0.5, 6.7, 'Run apply multiple times\nSame result every time', 
        ha='left', va='top', fontsize=8, color='#a0a0a0')

# Version controlled
ax.text(0.5, 5.5, '📋 Version Controlled', ha='left', va='top',
        fontsize=10, fontweight='bold', color=color_caution)
ax.text(0.5, 5.2, 'Commit .tf files to Git\nPeer review infrastructure', 
        ha='left', va='top', fontsize=8, color='#a0a0a0')

# Key commands box
commands_box = FancyBboxPatch((0.3, 0.3), 3.5, 2.3,
                              boxstyle="round,pad=0.1", 
                              edgecolor='#6366f1', facecolor='#1e1e3f',
                              linewidth=1.5)
ax.add_patch(commands_box)
ax.text(2.05, 2.4, 'Key Commands', ha='center', va='top',
        fontsize=11, fontweight='bold', color='#6366f1')
ax.text(0.5, 2.1, 'terraform init', ha='left', va='top',
        fontsize=9, fontweight='bold', color='#a0aec0', family='monospace')
ax.text(0.5, 1.9, '  Initialize project, download providers', ha='left', va='top',
        fontsize=8, color='#718096')
ax.text(0.5, 1.6, 'terraform plan', ha='left', va='top',
        fontsize=9, fontweight='bold', color='#a0aec0', family='monospace')
ax.text(0.5, 1.4, '  Preview changes (dry run)', ha='left', va='top',
        fontsize=8, color='#718096')
ax.text(0.5, 1.1, 'terraform apply', ha='left', va='top',
        fontsize=9, fontweight='bold', color='#a0aec0', family='monospace')
ax.text(0.5, 0.9, '  Provision infrastructure', ha='left', va='top',
        fontsize=8, color='#718096')
ax.text(0.5, 0.6, 'terraform destroy', ha='left', va='top',
        fontsize=9, fontweight='bold', color='#a0aec0', family='monospace')
ax.text(0.5, 0.4, '  Remove all managed resources', ha='left', va='top',
        fontsize=8, color='#718096')

# Save
output_path = '../img/ch06-terraform-workflow.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
print(f"✅ Diagram saved: {output_path}")
plt.close()
