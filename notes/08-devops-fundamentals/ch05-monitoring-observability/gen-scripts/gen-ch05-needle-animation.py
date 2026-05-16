#!/usr/bin/env python3
"""
Generate Chapter Animation — Monitoring & Observability Needle

Shows metric needle movement: error rate rising, alert firing, then dropping
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Wedge, FancyBboxPatch
import matplotlib
matplotlib.use('Agg')
from PIL import Image
import io

def generate_frame(error_rate, alert_fired=False):
    """Generate a single frame of the metric needle animation"""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.axis('off')
    
    # Title
    ax.text(0, 1.15, 'Error Rate Monitor', ha='center', va='top',
            fontsize=18, fontweight='bold', color='#ffffff')
    
    # Draw gauge background (semicircle)
    theta = np.linspace(0, np.pi, 100)
    radius = 1.0
    
    # Gauge zones
    # Green zone: 0-3%
    wedge_green = Wedge((0, 0), radius, 0, 60, width=0.1, 
                       facecolor='#15803d', edgecolor='#15803d', alpha=0.3)
    ax.add_patch(wedge_green)
    
    # Yellow zone: 3-5%
    wedge_yellow = Wedge((0, 0), radius, 60, 90, width=0.1,
                        facecolor='#b45309', edgecolor='#b45309', alpha=0.3)
    ax.add_patch(wedge_yellow)
    
    # Red zone: 5-10%
    wedge_red = Wedge((0, 0), radius, 90, 180, width=0.1,
                     facecolor='#b91c1c', edgecolor='#b91c1c', alpha=0.3)
    ax.add_patch(wedge_red)
    
    # Gauge outline
    arc_outer = mpatches.Arc((0, 0), 2*radius, 2*radius, angle=0, theta1=0, theta2=180,
                            color='#ffffff', linewidth=3)
    ax.add_patch(arc_outer)
    
    # Tick marks and labels
    tick_angles = [0, 60, 90, 120, 180]  # 0%, 3%, 5%, 7%, 10%
    tick_labels = ['0%', '3%', '5%', '7%', '10%']
    
    for angle, label in zip(tick_angles, tick_labels):
        rad = np.radians(angle)
        # Tick mark
        x_start = (radius - 0.15) * np.cos(rad)
        y_start = (radius - 0.15) * np.sin(rad)
        x_end = (radius - 0.05) * np.cos(rad)
        y_end = (radius - 0.05) * np.sin(rad)
        ax.plot([x_start, x_end], [y_start, y_end], color='#ffffff', linewidth=2)
        
        # Label
        x_label = (radius + 0.15) * np.cos(rad)
        y_label = (radius + 0.15) * np.sin(rad)
        ax.text(x_label, y_label, label, ha='center', va='center',
                fontsize=10, color='#ffffff', fontweight='bold')
    
    # Needle
    # Map error_rate (0-10%) to angle (0-180 degrees)
    needle_angle = (error_rate / 10.0) * 180
    needle_rad = np.radians(needle_angle)
    
    needle_length = radius - 0.15
    needle_x = needle_length * np.cos(needle_rad)
    needle_y = needle_length * np.sin(needle_rad)
    
    # Needle color based on zone
    if error_rate < 3:
        needle_color = '#15803d'
    elif error_rate < 5:
        needle_color = '#b45309'
    else:
        needle_color = '#b91c1c'
    
    # Draw needle
    ax.plot([0, needle_x], [0, needle_y], color=needle_color, linewidth=4)
    
    # Needle center dot
    circle = plt.Circle((0, 0), 0.05, color=needle_color, zorder=10)
    ax.add_patch(circle)
    
    # Current value display
    value_box = FancyBboxPatch((-0.3, -0.15), 0.6, 0.12,
                              boxstyle="round,pad=0.05",
                              edgecolor=needle_color, facecolor='#2d3748',
                              linewidth=2)
    ax.add_patch(value_box)
    ax.text(0, -0.09, f'{error_rate:.1f}%', ha='center', va='center',
            fontsize=16, fontweight='bold', color=needle_color)
    
    # Alert status
    if alert_fired:
        alert_text = '🚨 ALERT FIRED'
        alert_color = '#b91c1c'
        alert_box = FancyBboxPatch((-0.5, 0.7), 1.0, 0.15,
                                  boxstyle="round,pad=0.05",
                                  edgecolor=alert_color, facecolor='#2d1a1a',
                                  linewidth=3)
        ax.add_patch(alert_box)
        ax.text(0, 0.775, alert_text, ha='center', va='center',
                fontsize=14, fontweight='bold', color=alert_color)
    
    # Threshold line (5%)
    threshold_angle = (5.0 / 10.0) * 180
    threshold_rad = np.radians(threshold_angle)
    threshold_x = radius * np.cos(threshold_rad)
    threshold_y = radius * np.sin(threshold_rad)
    ax.plot([0, threshold_x], [0, threshold_y], color='#ef4444',
            linewidth=2, linestyle='--', alpha=0.7, label='Alert threshold')
    
    plt.tight_layout()
    
    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#1a1a2e', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    
    return img

# Generate animation frames
print("Generating animation frames...")

frames = []

# Phase 1: Normal operation (0-2%)
for i in range(20):
    error_rate = 1.0 + 0.5 * np.sin(i * 0.3)
    frames.append(generate_frame(error_rate, alert_fired=False))

# Phase 2: Error rate rising (2-6%)
for i in range(30):
    error_rate = 2.0 + (i / 30) * 4.5
    alert_fired = error_rate >= 5.0
    frames.append(generate_frame(error_rate, alert_fired=alert_fired))

# Phase 3: High error rate sustained (6-7%)
for i in range(15):
    error_rate = 6.5 + 0.3 * np.sin(i * 0.5)
    frames.append(generate_frame(error_rate, alert_fired=True))

# Phase 4: Error rate dropping (7-1%)
for i in range(30):
    error_rate = 7.0 - (i / 30) * 6.0
    alert_fired = error_rate >= 5.0
    frames.append(generate_frame(error_rate, alert_fired=alert_fired))

# Phase 5: Back to normal (1-2%)
for i in range(20):
    error_rate = 1.5 + 0.3 * np.sin(i * 0.3)
    frames.append(generate_frame(error_rate, alert_fired=False))

# Save as GIF
print("Saving animation...")
frames[0].save(
    '../img/ch05-monitoring-observability-needle.gif',
    save_all=True,
    append_images=frames[1:],
    duration=100,  # 100ms per frame
    loop=0
)

# Also save first frame as static PNG
frames[0].save('../img/ch05-monitoring-observability-needle.png')

print("✅ Generated: ../img/ch05-monitoring-observability-needle.gif")
print("✅ Generated: ../img/ch05-monitoring-observability-needle.png")
print(f"   Total frames: {len(frames)}")
print(f"   Duration: {len(frames) * 0.1:.1f} seconds")
