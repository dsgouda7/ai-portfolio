"""
Generate serving architecture diagram for Ch.6 Model Serving Frameworks.

Shows production deployment architecture:
- Load balancer distributing requests
- Multiple vLLM replicas (GPU instances)
- Monitoring with Prometheus
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#16213e')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Colors
color_lb = '#1d4ed8'      # Blue - Load balancer
color_vllm = '#15803d'    # Green - vLLM instances
color_prom = '#b45309'    # Orange - Prometheus
color_text = 'white'

# Title
ax.text(5, 9.5, 'Production vLLM Serving Architecture', 
        ha='center', fontsize=20, fontweight='bold', color=color_text)

# Load Balancer (top)
lb_box = FancyBboxPatch((3.5, 7.5), 3, 1, 
                        boxstyle="round,pad=0.1", 
                        edgecolor='white', facecolor=color_lb, linewidth=2)
ax.add_patch(lb_box)
ax.text(5, 8, 'Load Balancer (NGINX)', ha='center', va='center',
        fontsize=13, fontweight='bold', color='white')

# Load balancer details
ax.text(5, 7.2, '• Health checks every 10s\n• Round-robin routing\n• 450 req/s total capacity',
        ha='center', va='top', fontsize=10, color='white',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#0f172a', edgecolor='white', linewidth=1))

# vLLM Replicas (middle row)
replica_positions = [1.5, 5, 8.5]
for i, x_pos in enumerate(replica_positions):
    # GPU instance box
    replica_box = FancyBboxPatch((x_pos - 0.8, 4), 1.6, 2,
                                 boxstyle="round,pad=0.1",
                                 edgecolor='white', facecolor=color_vllm, linewidth=2)
    ax.add_patch(replica_box)
    
    # Labels
    ax.text(x_pos, 5.5, f'vLLM #{i+1}', ha='center', va='center',
            fontsize=12, fontweight='bold', color='white')
    ax.text(x_pos, 5, 'A100 GPU', ha='center', va='center',
            fontsize=10, color='white')
    ax.text(x_pos, 4.5, '150 req/s', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')
    
    # Arrow from LB to replica
    arrow = FancyArrowPatch((5, 7.5), (x_pos, 6),
                           arrowstyle='->', mutation_scale=25, linewidth=2,
                           color='white', alpha=0.7)
    ax.add_patch(arrow)

# Monitoring (bottom)
prom_box = FancyBboxPatch((3.5, 1.5), 3, 1.2,
                          boxstyle="round,pad=0.1",
                          edgecolor='white', facecolor=color_prom, linewidth=2)
ax.add_patch(prom_box)
ax.text(5, 2.1, 'Prometheus Monitoring', ha='center', va='center',
        fontsize=13, fontweight='bold', color='white')
ax.text(5, 1.7, 'Scrapes /metrics every 15s', ha='center', va='center',
        fontsize=9, color='white')

# Arrows from replicas to monitoring
for x_pos in replica_positions:
    arrow = FancyArrowPatch((x_pos, 4), (5, 2.7),
                           arrowstyle='->', mutation_scale=20, linewidth=1.5,
                           color='white', linestyle='--', alpha=0.5)
    ax.add_patch(arrow)

# Metrics panel
metrics_text = """Monitored Metrics:
• Request rate (req/s)
• P50/P95/P99 latency
• Error rate (4xx, 5xx)
• GPU utilization
• KV cache usage"""

ax.text(5, 0.5, metrics_text, ha='center', va='center',
        fontsize=9, color='white',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#0f172a', 
                 edgecolor=color_prom, linewidth=2))

# Key benefits callout
benefits_text = """✅ High availability (3 replicas)
✅ Automatic failover (health checks)
✅ Linear scaling (add more replicas)
✅ Real-time monitoring"""

ax.text(0.3, 5, benefits_text, ha='left', va='center',
        fontsize=10, color='white',
        bbox=dict(boxstyle='round,pad=0.7', facecolor='#15803d', 
                 edgecolor='white', linewidth=2, alpha=0.8))

plt.tight_layout()
plt.savefig('../img/ch06-serving-architecture.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
print("✅ Generated: ../img/ch06-serving-architecture.png")
plt.close()
