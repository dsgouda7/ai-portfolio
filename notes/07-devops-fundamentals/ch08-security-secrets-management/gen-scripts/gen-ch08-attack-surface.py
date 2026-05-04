#!/usr/bin/env python3
"""
Generate attack surface diagram for Ch.8 Security & Secrets Management.

Diagram: Shows where secrets can leak (Git, Docker images, logs, environment variables)
and highlights vulnerable areas in red.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Wedge
import numpy as np
import os

# Output configuration
OUTPUT_DIR = "../img"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Attack vectors with risk levels
ATTACK_VECTORS = [
    {
        "name": "Git History",
        "risk": "CRITICAL",
        "description": "Hardcoded secrets in commits",
        "mitigation": "Pre-commit hooks, .gitignore",
        "color": "#D32F2F"
    },
    {
        "name": "Docker Images",
        "risk": "CRITICAL",
        "description": "ENV secrets baked into layers",
        "mitigation": "Never use ENV for secrets",
        "color": "#D32F2F"
    },
    {
        "name": "Application Logs",
        "risk": "HIGH",
        "description": "Secrets logged during errors",
        "mitigation": "Sanitize logs, mask secrets",
        "color": "#F57C00"
    },
    {
        "name": "Environment Variables",
        "risk": "MEDIUM",
        "description": "Visible in docker inspect",
        "mitigation": "Use mounted files instead",
        "color": "#FBC02D"
    },
    {
        "name": "Process Lists",
        "risk": "MEDIUM",
        "description": "Command-line args visible",
        "mitigation": "Read from files, not args",
        "color": "#FBC02D"
    },
    {
        "name": "Shell History",
        "risk": "MEDIUM",
        "description": "docker run -e PASSWORD=...",
        "mitigation": "Use --env-file or mounts",
        "color": "#FBC02D"
    }
]

def create_diagram():
    """Create attack surface diagram showing all vulnerability points."""
    fig, ax = plt.subplots(figsize=(16, 10), facecolor='white')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Title
    ax.text(50, 96, "Secrets Attack Surface", 
            ha='center', va='top', fontsize=26, fontweight='bold')
    ax.text(50, 92, "Where Secrets Can Leak in Your Infrastructure", 
            ha='center', va='top', fontsize=14, color='#666666')
    
    # Central application (the target)
    center_x, center_y = 50, 50
    app_circle = Circle((center_x, center_y), 8, 
                         facecolor='#4CAF50', edgecolor='black', linewidth=2)
    ax.add_patch(app_circle)
    ax.text(center_x, center_y, "Your\nApplication", 
            ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # Draw attack vectors in a circle around the app
    num_vectors = len(ATTACK_VECTORS)
    radius = 28
    
    for i, vector in enumerate(ATTACK_VECTORS):
        # Calculate position
        angle = 2 * np.pi * i / num_vectors - np.pi/2  # Start from top
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        
        # Draw attack vector box
        box_width = 18
        box_height = 12
        box_x = x - box_width/2
        box_y = y - box_height/2
        
        box = FancyBboxPatch(
            (box_x, box_y), box_width, box_height,
            boxstyle="round,pad=0.5",
            facecolor=vector["color"],
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(box)
        
        # Risk level badge
        risk_colors = {
            "CRITICAL": "#FFFFFF",
            "HIGH": "#FFFFFF",
            "MEDIUM": "#333333"
        }
        ax.text(x, y + box_height/2 - 2, vector["risk"],
                ha='center', va='center', fontsize=9, fontweight='bold',
                color=risk_colors[vector["risk"]])
        
        # Vector name
        ax.text(x, y + 2, vector["name"],
                ha='center', va='center', fontsize=11, fontweight='bold',
                color='white' if vector["risk"] != "MEDIUM" else 'black')
        
        # Description
        ax.text(x, y - 2, vector["description"],
                ha='center', va='center', fontsize=8,
                color='white' if vector["risk"] != "MEDIUM" else 'black')
        
        # Draw arrow from app to vector (showing attack path)
        arrow_start_x = center_x + 8 * np.cos(angle)
        arrow_start_y = center_y + 8 * np.sin(angle)
        arrow_end_x = x - (box_width/2) * np.cos(angle)
        arrow_end_y = y - (box_height/2) * np.sin(angle)
        
        arrow = FancyArrowPatch(
            (arrow_start_x, arrow_start_y),
            (arrow_end_x, arrow_end_y),
            arrowstyle='<-', mutation_scale=20,
            color=vector["color"], linewidth=2, linestyle='--'
        )
        ax.add_patch(arrow)
    
    # Mitigation strategies box (bottom)
    mitigation_box = FancyBboxPatch(
        (5, 2), 90, 12,
        boxstyle="round,pad=0.8",
        facecolor='#E8F5E9',
        edgecolor='#4CAF50',
        linewidth=2
    )
    ax.add_patch(mitigation_box)
    
    ax.text(50, 12, "Mitigation Strategies", 
            ha='center', va='top', fontsize=14, fontweight='bold',
            color='#2E7D32')
    
    mitigations = [
        "✓ Pre-commit hooks (detect secrets before push)",
        "✓ Docker Secrets / K8s Secrets (mounted files)",
        "✓ Key Vault / Secrets Manager (centralized storage)",
        "✓ Trivy / gitleaks (automated scanning)",
        "✓ Log sanitization (mask passwords in logs)",
        "✓ RBAC (least privilege access)"
    ]
    
    cols = 3
    rows = 2
    for i, mitigation in enumerate(mitigations):
        col = i % cols
        row = i // cols
        x = 10 + col * 30
        y = 9 - row * 3
        ax.text(x, y, mitigation, ha='left', va='center',
                fontsize=9, color='#1B5E20')
    
    # Legend (risk levels)
    legend_x = 82
    legend_y = 78
    
    ax.text(legend_x, legend_y + 2, "Risk Levels:", 
            ha='left', va='top', fontsize=11, fontweight='bold')
    
    risk_levels = [
        ("CRITICAL", "#D32F2F", "Immediate data breach"),
        ("HIGH", "#F57C00", "High exposure risk"),
        ("MEDIUM", "#FBC02D", "Moderate risk")
    ]
    
    for i, (level, color, desc) in enumerate(risk_levels):
        y = legend_y - i * 4
        # Color box
        box = FancyBboxPatch(
            (legend_x, y - 1), 3, 2,
            facecolor=color, edgecolor='black', linewidth=1
        )
        ax.add_patch(box)
        # Text
        ax.text(legend_x + 4, y, f"{level}: {desc}",
                ha='left', va='center', fontsize=8)
    
    plt.tight_layout()
    return fig

def generate_diagram():
    """Generate PNG for the attack surface diagram."""
    print("Generating attack surface diagram...")
    
    fig = create_diagram()
    
    # Save PNG
    output_png = os.path.join(OUTPUT_DIR, "ch08-attack-surface.png")
    fig.savefig(output_png, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_png}")
    
    plt.close(fig)

if __name__ == "__main__":
    generate_diagram()
    print("\n✓ Attack surface diagram complete!")
    print("  File: ch08-attack-surface.png")
