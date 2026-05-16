"""Generate encoder-decoder-comparison.png — attention pattern visualization.

Shows:
  1. Encoder (bidirectional attention) — full attention matrix
  2. Decoder (causal attention) — lower-triangular mask
  3. Encoder-Decoder (cross-attention) — decoder queries encoder outputs
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np

plt.xkcd(scale=0.25, length=80, randomness=0.8)
plt.rcParams["path.effects"] = []

fig = plt.figure(figsize=(20, 7), facecolor="white")
fig.suptitle("Encoder vs Decoder vs Encoder-Decoder — Attention Patterns",
             fontsize=18, fontweight="bold", y=0.96)

# Color palette
BLUE = "#2E86C1"
GREEN = "#27AE60"
RED = "#E74C3C"
ORANGE = "#E67E22"
DARK = "#2C3E50"
GREY = "#95A5A6"
LIGHTGREY = "#ECF0F1"
YELLOW = "#F39C12"

# Create three subplots
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

def draw_attention_matrix(ax, n=6, mask_type="none", title="", color_allowed=GREEN, color_masked=RED):
    """Draw an attention matrix with optional masking"""
    ax.set_xlim(-0.5, n+0.5)
    ax.set_ylim(-0.5, n+0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()

    # Draw grid
    for i in range(n):
        for j in range(n):
            if mask_type == "none":  # Bidirectional (encoder)
                color = color_allowed
                alpha = 0.6
            elif mask_type == "causal":  # Decoder
                if j <= i:  # Lower triangle
                    color = color_allowed
                    alpha = 0.6
                else:  # Upper triangle (masked)
                    color = color_masked
                    alpha = 0.3

            rect = Rectangle((j-0.45, i-0.45), 0.9, 0.9,
                           facecolor=color, edgecolor="white", lw=2, alpha=alpha)
            ax.add_patch(rect)

    # Labels
    tokens = [f"T{i+1}" for i in range(n)]
    ax.set_xticks(range(n))
    ax.set_xticklabels(tokens, fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels(tokens, fontsize=10)
    ax.set_xlabel("Attending TO (Keys)", fontsize=11, fontweight="bold", color=DARK)
    ax.set_ylabel("Attending FROM (Queries)", fontsize=11, fontweight="bold", color=DARK)
    ax.set_title(title, fontsize=13, fontweight="bold", color=DARK, pad=15)

    # Add grid lines
    for i in range(n+1):
        ax.axhline(i-0.5, color='white', lw=2, zorder=10)
        ax.axvline(i-0.5, color='white', lw=2, zorder=10)

# ============= PANEL 1: ENCODER (BIDIRECTIONAL) =============
draw_attention_matrix(ax1, n=6, mask_type="none",
                     title="Encoder: Bidirectional Attention",
                     color_allowed=BLUE)

ax1.text(3, -2.5,
         "BERT, RoBERTa, E5\n"
         "• Every token sees every token\n"
         "• Full $(n \\times n)$ matrix\n"
         "• Best for understanding\n"
         "• Cannot generate",
         ha="center", fontsize=10, color=DARK,
         bbox=dict(boxstyle="round,pad=0.6", facecolor="#E8F6F3",
                  edgecolor=BLUE, lw=2))

# ============= PANEL 2: DECODER (CAUSAL) =============
draw_attention_matrix(ax2, n=6, mask_type="causal",
                     title="Decoder: Causal Attention (Masked)",
                     color_allowed=GREEN, color_masked=RED)

ax2.text(3, -2.5,
         "GPT, Claude, LLaMA\n"
         "• Token $i$ sees only $\\leq i$\n"
         "• Lower-triangular mask\n"
         "• Enables autoregressive gen\n"
         "• Weaker embeddings",
         ha="center", fontsize=10, color=DARK,
         bbox=dict(boxstyle="round,pad=0.6", facecolor="#FEF5E7",
                  edgecolor=ORANGE, lw=2))

# ============= PANEL 3: ENCODER-DECODER =============
ax3.set_xlim(0, 12)
ax3.set_ylim(0, 10)
ax3.axis("off")
ax3.set_title("Encoder-Decoder: Cross-Attention", fontsize=13, fontweight="bold", color=DARK)

# Encoder stack (left)
encoder_x = 1.5
encoder_y = 5
ax3.add_patch(FancyBboxPatch((encoder_x-0.8, encoder_y-1), 2, 2.5,
                            boxstyle="round,pad=0.1",
                            facecolor=BLUE, edgecolor="white", lw=3, alpha=0.3))
ax3.text(encoder_x+0.2, encoder_y+0.5, "ENCODER\n(Bidirectional)",
        ha="center", fontsize=11, fontweight="bold", color=BLUE)

# Encoder attention matrix (small)
for i in range(3):
    for j in range(3):
        rect = Rectangle((encoder_x-0.6 + j*0.5, encoder_y-0.8 + i*0.5), 0.45, 0.45,
                       facecolor=BLUE, edgecolor="white", lw=1.5, alpha=0.6)
        ax3.add_patch(rect)

# Input text
ax3.text(encoder_x+0.2, encoder_y-2, '"Translate:\nThe bank"',
        ha="center", fontsize=9, color=DARK, style="italic")

# Decoder stack (right)
decoder_x = 7.5
decoder_y = 5
ax3.add_patch(FancyBboxPatch((decoder_x-0.8, decoder_y-1), 2, 2.5,
                            boxstyle="round,pad=0.1",
                            facecolor=ORANGE, edgecolor="white", lw=3, alpha=0.3))
ax3.text(decoder_x+0.2, decoder_y+0.5, "DECODER\n(Causal)",
        ha="center", fontsize=11, fontweight="bold", color=ORANGE)

# Decoder attention matrix (causal, small)
for i in range(3):
    for j in range(3):
        if j <= i:
            color = GREEN
            alpha = 0.6
        else:
            color = RED
            alpha = 0.3
        rect = Rectangle((decoder_x-0.6 + j*0.5, decoder_y-0.8 + i*0.5), 0.45, 0.45,
                       facecolor=color, edgecolor="white", lw=1.5, alpha=alpha)
        ax3.add_patch(rect)

# Output text
ax3.text(decoder_x+0.2, decoder_y-2, '"La rive\ndu..."',
        ha="center", fontsize=9, color=DARK, style="italic")

# Cross-attention arrow
cross_y = 6
arrow = FancyArrowPatch((encoder_x+1.5, cross_y), (decoder_x-1, cross_y),
                       arrowstyle="<->", color=PURPLE, lw=4,
                       connectionstyle="arc3,rad=0.3", zorder=5,
                       mutation_scale=30)
ax3.add_patch(arrow)
ax3.text(4.5, cross_y+1, "CROSS-ATTENTION", ha="center", fontsize=11,
        fontweight="bold", color=PURPLE,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=YELLOW,
                 edgecolor=PURPLE, lw=2, alpha=0.9))

ax3.text(4.5, cross_y-0.5,
        "Decoder Q × Encoder K,V\n"
        "\"Which input parts matter\nfor this output token?\"",
        ha="center", fontsize=9, color=DARK, style="italic")

# Bottom explanation
ax3.text(6, 1.5,
         "T5, BART, Whisper\n"
         "Encoder: bidirectional (understands input)\n"
         "Decoder: causal (generates output)\n"
         "Cross-attn: decoder queries encoder\n"
         "Best for: translation, summarization, speech-to-text",
         ha="center", fontsize=10, color=DARK,
         bbox=dict(boxstyle="round,pad=0.6", facecolor=LIGHTGREY,
                  edgecolor=DARK, lw=2))

# Add legend
PURPLE = "#8E44AD"

plt.tight_layout()
output_path = Path(__file__).parent.parent / "img" / "encoder-decoder-comparison.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150, facecolor="white", bbox_inches="tight")
print(f"✓ Generated: {output_path}")
plt.close()
