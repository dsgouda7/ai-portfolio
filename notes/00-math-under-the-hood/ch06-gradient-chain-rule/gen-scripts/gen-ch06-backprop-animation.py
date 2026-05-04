"""Generate backpropagation animation: gradient flowing backward through a 3-layer network.

Network: x → h₁ (ReLU) → h₂ (ReLU) → ŷ (linear), loss = (y - ŷ)²

Animation shows:
  1. Forward pass (activations light up left → right, values displayed)
  2. Loss computation
  3. Backward pass (gradients light up right → left, ∂L/∂• displayed)
  4. Weight update arrows

30 frames total (10 forward, 2 loss, 16 backward, 2 update).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from pathlib import Path

DARK   = "#2C3E50"
BLUE   = "#2E86C1"
ORANGE = "#E67E22"
RED    = "#E74C3C"
GREEN  = "#27AE60"
GREY   = "#BDC3C7"
LIGHT  = "#ECF0F1"

# Simple 3-layer network (we'll use concrete numbers for visual clarity)
# x = 2.0, W1 = 0.5, b1 = 0.3 → z1 = 1.3 → h1 = ReLU(1.3) = 1.3
# W2 = 0.8, b2 = -0.2 → z2 = 0.84 → h2 = ReLU(0.84) = 0.84
# W3 = 1.2, b3 = 0.1 → z3 = 1.108 → ŷ = 1.108
# y_true = 2.0 → loss = (2.0 - 1.108)² = 0.795

x_val = 2.0
y_true = 2.0

W1, b1 = 0.5, 0.3
z1 = W1 * x_val + b1  # 1.3
h1 = max(0, z1)        # 1.3 (ReLU)

W2, b2 = 0.8, -0.2
z2 = W2 * h1 + b2      # 0.84
h2 = max(0, z2)        # 0.84

W3, b3 = 1.2, 0.1
z3 = W3 * h2 + b3      # 1.108
y_hat = z3             # 1.108

loss_val = 0.5 * (y_true - y_hat) ** 2  # 0.398

# Backward pass (gradients)
dL_dyhat = -(y_true - y_hat)           # -0.892
dL_dz3 = dL_dyhat                       # -0.892
dL_dW3 = dL_dz3 * h2                    # -0.749
dL_db3 = dL_dz3                         # -0.892
dL_dh2 = dL_dz3 * W3                    # -1.070

dL_dz2 = dL_dh2 * (1 if z2 > 0 else 0) # -1.070 (ReLU deriv)
dL_dW2 = dL_dz2 * h1                    # -1.391
dL_db2 = dL_dz2                         # -1.070
dL_dh1 = dL_dz2 * W2                    # -0.856

dL_dz1 = dL_dh1 * (1 if z1 > 0 else 0) # -0.856
dL_dW1 = dL_dz1 * x_val                 # -1.712
dL_db1 = dL_dz1                         # -0.856

# Network layout (node positions)
layer_x = [1, 3, 5, 7]  # x, h1, h2, ŷ
node_y = 3

fig, ax = plt.subplots(figsize=(12, 5), facecolor="white")
ax.set_xlim(0, 8)
ax.set_ylim(0, 6)
ax.axis("off")
ax.set_title("Backpropagation: Gradient Flows Backward", 
             fontsize=14, fontweight="bold", color=DARK, y=0.98)

# Static: network structure (edges as lines, nodes as circles)
# Edges (will be colored during animation)
edges = [
    # (from_x, to_x, weight_label, gradient_label)
    (layer_x[0], layer_x[1], f"W₁={W1}", f"∂L/∂W₁={dL_dW1:.2f}"),
    (layer_x[1], layer_x[2], f"W₂={W2}", f"∂L/∂W₂={dL_dW2:.2f}"),
    (layer_x[2], layer_x[3], f"W₃={W3}", f"∂L/∂W₃={dL_dW3:.2f}"),
]

edge_lines = []
edge_labels = []
for i, (x1, x2, w_lab, g_lab) in enumerate(edges):
    line, = ax.plot([x1, x2], [node_y, node_y], color=GREY, lw=3, alpha=0.3, zorder=1)
    edge_lines.append(line)
    
    # Weight label (top)
    label_w = ax.text((x1 + x2) / 2, node_y + 0.4, w_lab, ha="center", va="bottom",
                      fontsize=9, color=DARK, alpha=0.5)
    # Gradient label (bottom, initially hidden)
    label_g = ax.text((x1 + x2) / 2, node_y - 0.4, g_lab, ha="center", va="top",
                      fontsize=9, color=RED, alpha=0, fontweight="bold")
    edge_labels.append((label_w, label_g))

# Nodes
node_circles = []
node_labels = []
node_vals = [
    (layer_x[0], "x", f"{x_val:.1f}"),
    (layer_x[1], "h₁", f"{h1:.2f}"),
    (layer_x[2], "h₂", f"{h2:.2f}"),
    (layer_x[3], "ŷ", f"{y_hat:.2f}"),
]

for x_pos, name, val in node_vals:
    circ = plt.Circle((x_pos, node_y), 0.35, color=LIGHT, ec=DARK, lw=2, zorder=3)
    ax.add_patch(circ)
    node_circles.append(circ)
    
    # Name label (inside circle)
    label_n = ax.text(x_pos, node_y, name, ha="center", va="center",
                      fontsize=11, fontweight="bold", color=DARK)
    # Value label (below circle, initially hidden)
    label_v = ax.text(x_pos, node_y - 0.80, val, ha="center", va="top",
                      fontsize=9, color=BLUE, alpha=0)
    node_labels.append((label_n, label_v))

# Loss display (top-right)
loss_text = ax.text(7.5, 5.2, "", ha="right", va="top", fontsize=12,
                    color=DARK, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.4", fc=LIGHT, ec=DARK, lw=1.5))

# Phase indicator (top-left)
phase_text = ax.text(0.5, 5.2, "", ha="left", va="top", fontsize=11,
                     color=DARK, family="monospace",
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=DARK, lw=1))

# Gradient text (right of ŷ, initially hidden)
grad_text = ax.text(layer_x[3] + 0.8, node_y, "", ha="left", va="center",
                    fontsize=10, color=RED, alpha=0, fontweight="bold")

# Animation frames:
# 0-3:   forward x → h1
# 4-7:   forward h1 → h2
# 8-11:  forward h2 → ŷ
# 12-13: compute loss
# 14-16: backward ŷ → h2
# 17-19: backward h2 → h1
# 20-22: backward h1 → x
# 23-25: update weights (show gradient labels)
# 26-29: hold final state

n_frames = 30

def animate(frame):
    # Reset all to default
    for line in edge_lines:
        line.set_alpha(0.3)
        line.set_color(GREY)
        line.set_lw(3)
    
    for circ in node_circles:
        circ.set_facecolor(LIGHT)
    
    for label_n, label_v in node_labels:
        label_v.set_alpha(0)
    
    for label_w, label_g in edge_labels:
        label_g.set_alpha(0)
    
    grad_text.set_alpha(0)
    loss_text.set_text("")
    
    # --- Forward pass frames ---
    if frame < 4:  # x → h1
        phase_text.set_text("FORWARD: x → h₁")
        edge_lines[0].set_alpha(1.0)
        edge_lines[0].set_color(GREEN)
        node_circles[0].set_facecolor(GREEN)
        node_circles[1].set_facecolor(GREEN)
        node_labels[0][1].set_alpha(1)
        node_labels[1][1].set_alpha(1)
    
    elif frame < 8:  # h1 → h2
        phase_text.set_text("FORWARD: h₁ → h₂")
        edge_lines[0].set_alpha(0.5)
        edge_lines[1].set_alpha(1.0)
        edge_lines[1].set_color(GREEN)
        node_circles[1].set_facecolor(GREEN)
        node_circles[2].set_facecolor(GREEN)
        node_labels[1][1].set_alpha(1)
        node_labels[2][1].set_alpha(1)
    
    elif frame < 12:  # h2 → ŷ
        phase_text.set_text("FORWARD: h₂ → ŷ")
        edge_lines[1].set_alpha(0.5)
        edge_lines[2].set_alpha(1.0)
        edge_lines[2].set_color(GREEN)
        node_circles[2].set_facecolor(GREEN)
        node_circles[3].set_facecolor(GREEN)
        node_labels[2][1].set_alpha(1)
        node_labels[3][1].set_alpha(1)
    
    elif frame < 14:  # compute loss
        phase_text.set_text("COMPUTE LOSS")
        edge_lines[2].set_alpha(0.5)
        node_circles[3].set_facecolor(ORANGE)
        node_labels[3][1].set_alpha(1)
        loss_text.set_text(f"L = ½(y - ŷ)²\n  = {loss_val:.3f}")
    
    # --- Backward pass frames ---
    elif frame < 17:  # ŷ → h2
        phase_text.set_text("BACKWARD: ∂L/∂h₂")
        edge_lines[2].set_alpha(1.0)
        edge_lines[2].set_color(RED)
        edge_lines[2].set_lw(4)
        node_circles[3].set_facecolor(RED)
        node_circles[2].set_facecolor(ORANGE)
        grad_text.set_text(f"∂L/∂ŷ={dL_dyhat:.2f}")
        grad_text.set_alpha(1)
        edge_labels[2][1].set_alpha(1)
    
    elif frame < 20:  # h2 → h1
        phase_text.set_text("BACKWARD: ∂L/∂h₁")
        edge_lines[2].set_alpha(0.5)
        edge_lines[2].set_color(RED)
        edge_lines[1].set_alpha(1.0)
        edge_lines[1].set_color(RED)
        edge_lines[1].set_lw(4)
        node_circles[2].set_facecolor(RED)
        node_circles[1].set_facecolor(ORANGE)
        edge_labels[2][1].set_alpha(0.6)
        edge_labels[1][1].set_alpha(1)
    
    elif frame < 23:  # h1 → x
        phase_text.set_text("BACKWARD: ∂L/∂W₁")
        edge_lines[1].set_alpha(0.5)
        edge_lines[1].set_color(RED)
        edge_lines[0].set_alpha(1.0)
        edge_lines[0].set_color(RED)
        edge_lines[0].set_lw(4)
        node_circles[1].set_facecolor(RED)
        node_circles[0].set_facecolor(ORANGE)
        edge_labels[1][1].set_alpha(0.6)
        edge_labels[0][1].set_alpha(1)
    
    else:  # hold final state (all gradients visible)
        phase_text.set_text("GRADIENTS COMPUTED → UPDATE")
        for line in edge_lines:
            line.set_alpha(0.8)
            line.set_color(RED)
        for label_w, label_g in edge_labels:
            label_g.set_alpha(1)
        for circ in node_circles:
            circ.set_facecolor("#FFE5E5")
    
    return edge_lines + [c for c in node_circles] + [phase_text, loss_text, grad_text]

anim = animation.FuncAnimation(fig, animate, frames=n_frames,
                              interval=500, blit=False, repeat=True)

out = Path(__file__).with_name("ch06-backprop-animation.gif")
print(f"Rendering {n_frames} frames...")
anim.save(out, writer="pillow", fps=2, dpi=120)
print(f"wrote {out}")
plt.close()
