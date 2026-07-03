"""
Generate gradient flow comparison: Plain CNN vs ResNet.

Output: ch01-gradient-flow-comparison.png

Shows how gradients vanish in plain CNNs but are preserved in ResNets.
"""

import matplotlib.pyplot as plt
import numpy as np

# Dark theme
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='#1a1a2e')
ax1.set_facecolor('#1a1a2e')
ax2.set_facecolor('#1a1a2e')

# Simulate gradient magnitudes across 40 layers
layers = np.arange(1, 41)

# Plain CNN: exponential decay (vanishing gradient)
plain_gradients = np.exp(-0.1 * layers)  # Decay factor 0.9 per layer
plain_gradients = plain_gradients / plain_gradients[-1]  # Normalize to last layer = 1

# ResNet: preserved gradients (slight decay but much better)
resnet_gradients = 0.95 ** layers  # Much slower decay
resnet_gradients = resnet_gradients / resnet_gradients[-1]  # Normalize to last layer = 1

# Add noise for realism
np.random.seed(42)
plain_gradients += np.random.normal(0, 0.01, len(plain_gradients))
resnet_gradients += np.random.normal(0, 0.05, len(resnet_gradients))
plain_gradients = np.maximum(plain_gradients, 0)  # Clip negatives
resnet_gradients = np.maximum(resnet_gradients, 0)

# Plot 1: Plain CNN (vanishing)
ax1.plot(layers, plain_gradients, color='#ef4444', linewidth=3, marker='o', markersize=5, label='Gradient Magnitude')
ax1.fill_between(layers, 0, plain_gradients, color='#ef4444', alpha=0.2)
ax1.axhline(y=1.0, color='white', linestyle='--', linewidth=1.5, alpha=0.5, label='Target (1.0)')
ax1.axhline(y=0.1, color='#f59e0b', linestyle='--', linewidth=1.5, alpha=0.5, label='Critical Threshold (0.1)')
ax1.set_xlabel('Layer (1 = first, 40 = last)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Gradient Magnitude (normalized)', fontsize=12, fontweight='bold')
ax1.set_title('Plain 40-Layer CNN\n❌ Vanishing Gradient Problem', fontsize=14, fontweight='bold', color='#ef4444')
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(alpha=0.3)
ax1.set_ylim(-0.1, 1.5)

# Annotate first layer
first_grad_plain = plain_gradients[0]
ax1.annotate(f'Layer 1: {first_grad_plain:.3f}\n(98% vanished!)', 
             xy=(1, first_grad_plain), xytext=(8, 0.4),
             fontsize=10, color='#ef4444', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#ef4444', lw=2),
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e', edgecolor='#ef4444', linewidth=2))

# Plot 2: ResNet (preserved)
ax2.plot(layers, resnet_gradients, color='#10b981', linewidth=3, marker='o', markersize=5, label='Gradient Magnitude')
ax2.fill_between(layers, 0, resnet_gradients, color='#10b981', alpha=0.2)
ax2.axhline(y=1.0, color='white', linestyle='--', linewidth=1.5, alpha=0.5, label='Target (1.0)')
ax2.axhline(y=0.1, color='#f59e0b', linestyle='--', linewidth=1.5, alpha=0.5, label='Critical Threshold (0.1)')
ax2.set_xlabel('Layer (1 = first, 40 = last)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Gradient Magnitude (normalized)', fontsize=12, fontweight='bold')
ax2.set_title('ResNet with Skip Connections\n✅ Gradient Preserved', fontsize=14, fontweight='bold', color='#10b981')
ax2.legend(fontsize=10, loc='upper right')
ax2.grid(alpha=0.3)
ax2.set_ylim(-0.1, 1.5)

# Annotate first layer
first_grad_resnet = resnet_gradients[0]
ax2.annotate(f'Layer 1: {first_grad_resnet:.3f}\n(only 20% loss!)', 
             xy=(1, first_grad_resnet), xytext=(8, 1.2),
             fontsize=10, color='#10b981', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#10b981', lw=2),
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e', edgecolor='#10b981', linewidth=2))

plt.tight_layout()
plt.savefig('../img/ch01-gradient-flow-comparison.png', dpi=150, facecolor='#1a1a2e', bbox_inches='tight')
print('✅ Generated: ../img/ch01-gradient-flow-comparison.png')
plt.close()
