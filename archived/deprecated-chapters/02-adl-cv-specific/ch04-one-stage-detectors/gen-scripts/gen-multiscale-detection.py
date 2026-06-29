"""
Generate multi-scale detection visualization (Feature Pyramid Network).

Shows how one-stage detectors detect objects at multiple scales using
different feature map resolutions (P3, P4, P5).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Dark theme
plt.style.use('dark_background')

fig = plt.figure(figsize=(14, 10), facecolor='#1a1a2e')
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

# Main axes
ax_main = fig.add_subplot(gs[:, 0])
ax_p3 = fig.add_subplot(gs[0, 1:])
ax_p4 = fig.add_subplot(gs[1, 1:])
ax_p5 = fig.add_subplot(gs[2, 1:])

for ax in [ax_main, ax_p3, ax_p4, ax_p5]:
    ax.set_facecolor('#1a1a2e')
    ax.set_aspect('equal')
    ax.axis('off')

# ─────────────────────────────────────────────────────────────
# Main: Input image with objects at different scales
# ─────────────────────────────────────────────────────────────
img_size = 640
ax_main.set_xlim(0, img_size)
ax_main.set_ylim(0, img_size)
ax_main.invert_yaxis()

# Background
bg = patches.Rectangle((0, 0), img_size, img_size,
                       facecolor='#1a1a2e', edgecolor='#444', linewidth=2)
ax_main.add_patch(bg)

# Objects at different scales
objects = [
    # Small objects (50×50)
    {'pos': (100, 100), 'size': (50, 50), 'color': '#15803d', 'label': 'Small\n(50×50)', 'scale': 'P3'},
    {'pos': (180, 120), 'size': (50, 50), 'color': '#15803d', 'label': '', 'scale': 'P3'},
    
    # Medium objects (120×120)
    {'pos': (300, 250), 'size': (120, 120), 'color': '#b45309', 'label': 'Medium\n(120×120)', 'scale': 'P4'},
    {'pos': (450, 240), 'size': (120, 120), 'color': '#b45309', 'label': '', 'scale': 'P4'},
    
    # Large objects (200×200)
    {'pos': (100, 400), 'size': (200, 200), 'color': '#b91c1c', 'label': 'Large\n(200×200)', 'scale': 'P5'},
]

for obj in objects:
    x, y = obj['pos']
    w, h = obj['size']
    rect = patches.Rectangle((x, y), w, h,
                            facecolor=obj['color'], edgecolor='white',
                            linewidth=2, alpha=0.6)
    ax_main.add_patch(rect)
    
    if obj['label']:
        ax_main.text(x + w/2, y + h/2, obj['label'],
                    ha='center', va='center', fontsize=10,
                    color='white', weight='bold')

ax_main.set_title('Input Image (640×640)\nMultiple Object Scales', 
                 fontsize=13, weight='bold', color='white')

# ─────────────────────────────────────────────────────────────
# P3: 80×80 feature map (detects small objects)
# ─────────────────────────────────────────────────────────────
p3_size = 80
ax_p3.set_xlim(0, p3_size)
ax_p3.set_ylim(0, p3_size)
ax_p3.invert_yaxis()

# Draw grid
for i in range(0, p3_size + 1, 10):
    ax_p3.plot([i, i], [0, p3_size], color='#444', alpha=0.3, linewidth=0.5)
    ax_p3.plot([0, p3_size], [i, i], color='#444', alpha=0.3, linewidth=0.5)

# Highlight small object regions
for obj in objects:
    if obj['scale'] == 'P3':
        # Map to P3 coordinates (8× downsampling)
        x = obj['pos'][0] / 8
        y = obj['pos'][1] / 8
        w = obj['size'][0] / 8
        h = obj['size'][1] / 8
        
        highlight = patches.Rectangle((x, y), w, h,
                                     facecolor='#15803d', edgecolor='lime',
                                     linewidth=2, alpha=0.5)
        ax_p3.add_patch(highlight)
        
        # Detection marker
        cx, cy = x + w/2, y + h/2
        ax_p3.plot(cx, cy, 'o', color='lime', markersize=8)

ax_p3.set_title('P3 Feature Map (80×80)\nReceptive field: 64×64 px | Detects: Small objects',
               fontsize=11, weight='bold', color='#15803d')

# Info box
info = "Stride: 8\nDownsample: 8×\nAnchors: 3 (scales: 32, 40, 50)"
ax_p3.text(2, p3_size - 2, info, ha='left', va='top',
          fontsize=9, color='white',
          bbox=dict(boxstyle='round', facecolor='black', alpha=0.8,
                   edgecolor='#15803d', linewidth=2))

# ─────────────────────────────────────────────────────────────
# P4: 40×40 feature map (detects medium objects)
# ─────────────────────────────────────────────────────────────
p4_size = 40
ax_p4.set_xlim(0, p4_size)
ax_p4.set_ylim(0, p4_size)
ax_p4.invert_yaxis()

# Draw grid
for i in range(0, p4_size + 1, 5):
    ax_p4.plot([i, i], [0, p4_size], color='#444', alpha=0.3, linewidth=0.5)
    ax_p4.plot([0, p4_size], [i, i], color='#444', alpha=0.3, linewidth=0.5)

# Highlight medium object regions
for obj in objects:
    if obj['scale'] == 'P4':
        # Map to P4 coordinates (16× downsampling)
        x = obj['pos'][0] / 16
        y = obj['pos'][1] / 16
        w = obj['size'][0] / 16
        h = obj['size'][1] / 16
        
        highlight = patches.Rectangle((x, y), w, h,
                                     facecolor='#b45309', edgecolor='orange',
                                     linewidth=2, alpha=0.5)
        ax_p4.add_patch(highlight)
        
        # Detection marker
        cx, cy = x + w/2, y + h/2
        ax_p4.plot(cx, cy, 's', color='orange', markersize=8)

ax_p4.set_title('P4 Feature Map (40×40)\nReceptive field: 128×128 px | Detects: Medium objects',
               fontsize=11, weight='bold', color='#b45309')

# Info box
info = "Stride: 16\nDownsample: 16×\nAnchors: 3 (scales: 64, 80, 100)"
ax_p4.text(1, p4_size - 1, info, ha='left', va='top',
          fontsize=9, color='white',
          bbox=dict(boxstyle='round', facecolor='black', alpha=0.8,
                   edgecolor='#b45309', linewidth=2))

# ─────────────────────────────────────────────────────────────
# P5: 20×20 feature map (detects large objects)
# ─────────────────────────────────────────────────────────────
p5_size = 20
ax_p5.set_xlim(0, p5_size)
ax_p5.set_ylim(0, p5_size)
ax_p5.invert_yaxis()

# Draw grid
for i in range(0, p5_size + 1, 2):
    ax_p5.plot([i, i], [0, p5_size], color='#444', alpha=0.3, linewidth=0.5)
    ax_p5.plot([0, p5_size], [i, i], color='#444', alpha=0.3, linewidth=0.5)

# Highlight large object region
for obj in objects:
    if obj['scale'] == 'P5':
        # Map to P5 coordinates (32× downsampling)
        x = obj['pos'][0] / 32
        y = obj['pos'][1] / 32
        w = obj['size'][0] / 32
        h = obj['size'][1] / 32
        
        highlight = patches.Rectangle((x, y), w, h,
                                     facecolor='#b91c1c', edgecolor='red',
                                     linewidth=2, alpha=0.5)
        ax_p5.add_patch(highlight)
        
        # Detection marker
        cx, cy = x + w/2, y + h/2
        ax_p5.plot(cx, cy, '^', color='red', markersize=10)

ax_p5.set_title('P5 Feature Map (20×20)\nReceptive field: 256×256 px | Detects: Large objects',
               fontsize=11, weight='bold', color='#b91c1c')

# Info box
info = "Stride: 32\nDownsample: 32×\nAnchors: 3 (scales: 128, 160, 200)"
ax_p5.text(0.5, p5_size - 0.5, info, ha='left', va='top',
          fontsize=9, color='white',
          bbox=dict(boxstyle='round', facecolor='black', alpha=0.8,
                   edgecolor='#b91c1c', linewidth=2))

# Main title
fig.suptitle('YOLOv5 Multi-Scale Detection (Feature Pyramid Network)', 
            fontsize=16, weight='bold', color='white', y=0.98)

# Summary text
summary = (
    "Total predictions: 80×80×3 + 40×40×3 + 20×20×3 = 25,200 boxes\n"
    "After filtering: ~500 boxes (confidence > 0.25)\n"
    "After NMS: ~15 final detections\n"
    "✓ All scales covered: Small (P3), Medium (P4), Large (P5)"
)
fig.text(0.5, 0.02, summary, ha='center', va='bottom',
        fontsize=10, color='white',
        bbox=dict(boxstyle='round', facecolor='black', alpha=0.9,
                 edgecolor='#00d9ff', linewidth=2))

plt.savefig('../img/ch04-multiscale-detection.png', dpi=150, facecolor='#1a1a2e')
print("✓ Saved: ch04-multiscale-detection.png")

plt.close()
