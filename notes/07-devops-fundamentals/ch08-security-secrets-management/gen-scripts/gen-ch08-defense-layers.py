#!/usr/bin/env python3
"""
Generate defense layers diagram for Ch.8 Security & Secrets Management.

Diagram: Shows defense-in-depth strategy with multiple layers:
Pre-commit hooks → Image scanning → Runtime policies → Audit logs
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np
import os

# Output configuration
OUTPUT_DIR = "../img"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Defense layers (innermost to outermost)
DEFENSE_LAYERS = [
    {
        "name": "Application Core",
        "description": "Secrets never in source code",
        "tools": ["Environment variables", "Secret SDK"],
        "color": "#4CAF50",
        "size": 1.0
    },
    {
        "name": "Runtime Policies",
        "description": "Secrets injected at container startup",
        "tools": ["Docker Secrets", "K8s Secrets", "Managed Identity"],
        "color": "#2196F3",
        "size": 1.3
    },
    {
        "name": "Image Scanning",
        "description": "Block images with vulnerabilities or secrets",
        "tools": ["Trivy", "Snyk", "Azure Defender"],
        "color": "#FF9800",
        "size": 1.6
    },
    {
        "name": "Pre-Commit Hooks",
        "description": "Prevent secrets from reaching git",
        "tools": ["gitleaks", "pre-commit", "GitHub Secret Scanning"],
        "color": "#9C27B0",
        "size": 1.9
    },
    {
        "name": "Audit & Monitoring",
        "description": "Track all secret access events",
        "tools": ["Key Vault logs", "CloudTrail", "Azure Monitor"],
        "color": "#F44336",
        "size": 2.2
    }
]

def create_diagram():
    """Create defense layers diagram showing depth-in-defense strategy."""
    fig, ax = plt.subplots(figsize=(16, 12), facecolor='white')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Title
    ax.text(50, 96, "Defense-in-Depth Strategy", 
            ha='center', va='top', fontsize=26, fontweight='bold')
    ax.text(50, 92, "Multiple Layers of Secret Protection", 
            ha='center', va='top', fontsize=14, color='#666666')
    
    # Draw concentric rectangles (layers)
    center_x, center_y = 50, 48
    base_width = 12
    base_height = 10
    
    for i, layer in enumerate(DEFENSE_LAYERS):
        width = base_width * layer["size"]
        height = base_height * layer["size"]
        
        # Draw layer box
        box_x = center_x - width/2
        box_y = center_y - height/2
        
        box = FancyBboxPatch(
            (box_x, box_y), width, height,
            boxstyle="round,pad=1",
            facecolor='none',
            edgecolor=layer["color"],
            linewidth=3
        )
        ax.add_patch(box)
        
        # Layer label (positioned outside the box)
        if i % 2 == 0:  # Alternate left and right
            label_x = box_x - 2
            label_ha = 'right'
        else:
            label_x = box_x + width + 2
            label_ha = 'left'
        
        label_y = box_y + height/2
        
        # Layer number and name
        ax.text(label_x, label_y + 2, f"Layer {len(DEFENSE_LAYERS) - i}",
                ha=label_ha, va='center', fontsize=11, fontweight='bold',
                color=layer["color"])
        
        ax.text(label_x, label_y, layer["name"],
                ha=label_ha, va='center', fontsize=13, fontweight='bold',
                color='black')
        
        ax.text(label_x, label_y - 2, layer["description"],
                ha=label_ha, va='center', fontsize=9, color='#666666',
                style='italic')
    
    # Tools/technologies details (bottom section)
    tools_y_start = 18
    ax.text(50, tools_y_start + 2, "Defense Mechanisms by Layer",
            ha='center', va='top', fontsize=14, fontweight='bold')
    
    for i, layer in enumerate(DEFENSE_LAYERS):
        y = tools_y_start - (i + 1) * 3
        
        # Layer name
        ax.text(12, y, f"{len(DEFENSE_LAYERS) - i}. {layer['name']}:",
                ha='left', va='center', fontsize=10, fontweight='bold',
                color=layer["color"])
        
        # Tools
        tools_text = ", ".join(layer["tools"])
        ax.text(38, y, tools_text,
                ha='left', va='center', fontsize=9, color='#333333')
    
    # Attack flow visualization (right side)
    attack_x = 75
    attack_y_start = 75
    
    ax.text(attack_x, attack_y_start + 3, "Attack Prevention Flow",
            ha='center', va='top', fontsize=12, fontweight='bold',
            color='#D32F2F')
    
    attack_stages = [
        "Developer writes code",
        "↓ Pre-commit hook blocks secrets",
        "Code committed to git",
        "↓ Image scanning detects CVEs",
        "Container deployed",
        "↓ Runtime policy enforces mounts",
        "App accesses secrets",
        "↓ Audit logs track access"
    ]
    
    for i, stage in enumerate(attack_stages):
        y = attack_y_start - i * 3
        color = '#D32F2F' if '↓' in stage else '#333333'
        fontweight = 'bold' if '↓' in stage else 'normal'
        
        ax.text(attack_x, y, stage,
                ha='center', va='center', fontsize=8,
                color=color, fontweight=fontweight)
    
    # Threat actor (bottom left)
    threat_box = FancyBboxPatch(
        (5, 2), 18, 8,
        boxstyle="round,pad=0.5",
        facecolor='#FFEBEE',
        edgecolor='#D32F2F',
        linewidth=2
    )
    ax.add_patch(threat_box)
    
    ax.text(14, 8, "⚠ Threat Actor",
            ha='center', va='top', fontsize=11, fontweight='bold',
            color='#D32F2F')
    
    threats = [
        "• Malicious insider",
        "• External attacker",
        "• Compromised developer"
    ]
    
    for i, threat in enumerate(threats):
        ax.text(14, 6 - i * 1.5, threat,
                ha='center', va='center', fontsize=8, color='#C62828')
    
    # Blocked indicator
    blocked_box = FancyBboxPatch(
        (26, 2), 18, 8,
        boxstyle="round,pad=0.5",
        facecolor='#E8F5E9',
        edgecolor='#4CAF50',
        linewidth=2
    )
    ax.add_patch(blocked_box)
    
    ax.text(35, 8, "✓ Blocked by Layers",
            ha='center', va='top', fontsize=11, fontweight='bold',
            color='#2E7D32')
    
    protections = [
        "• Secrets never reach git",
        "• Images scanned before deploy",
        "• Runtime access controlled"
    ]
    
    for i, protection in enumerate(protections):
        ax.text(35, 6 - i * 1.5, protection,
                ha='center', va='center', fontsize=8, color='#1B5E20')
    
    # Arrow from threat to layers (showing attack path)
    attack_arrow = FancyArrowPatch(
        (23, 6), (30, 48),
        arrowstyle='->', mutation_scale=30,
        color='#D32F2F', linewidth=2, linestyle='--'
    )
    ax.add_patch(attack_arrow)
    
    ax.text(26, 27, "Attack\nAttempt",
            ha='center', va='center', fontsize=9,
            color='#D32F2F', fontweight='bold')
    
    plt.tight_layout()
    return fig

def generate_diagram():
    """Generate PNG for the defense layers diagram."""
    print("Generating defense layers diagram...")
    
    fig = create_diagram()
    
    # Save PNG
    output_png = os.path.join(OUTPUT_DIR, "ch08-defense-layers.png")
    fig.savefig(output_png, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_png}")
    
    plt.close(fig)

if __name__ == "__main__":
    generate_diagram()
    print("\n✓ Defense layers diagram complete!")
    print("  File: ch08-defense-layers.png")
