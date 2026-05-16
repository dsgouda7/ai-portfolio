"""
Generate chain of verification animation: Generate → Verify → Iterate flow
Shows claim generation and systematic verification with marked verified/unverified claims.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import imageio
import os

# Output paths
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'img')
os.makedirs(OUT_DIR, exist_ok=True)
OUTPUT_GIF = os.path.join(OUT_DIR, 'ch11_chain_of_verification.gif')

# Color scheme
COLOR_GENERATE = '#3498DB'   # Blue
COLOR_VERIFY = '#F39C12'     # Orange
COLOR_CORRECT = '#27AE60'    # Green
COLOR_INCORRECT = '#E74C3C'  # Red
COLOR_PENDING = '#95A5A6'    # Gray
COLOR_TEXT = '#2C3E50'       # Dark blue-gray
COLOR_BG = '#ECF0F1'         # Light gray

def draw_stage_box(ax, x, y, w, h, label, color, active=False):
    """Draw a stage indicator box."""
    alpha = 1.0 if active else 0.3
    linewidth = 3 if active else 1.5
    
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08",
        facecolor=color,
        edgecolor=COLOR_TEXT,
        linewidth=linewidth,
        alpha=alpha
    )
    ax.add_patch(box)
    
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            fontsize=11, fontweight='bold' if active else 'normal',
            color='white')

def draw_claim_card(ax, x, y, w, h, claim_num, claim_text, status, score=None):
    """Draw a claim card with verification status.
    
    status: 'pending', 'verifying', 'correct', 'incorrect'
    """
    colors = {
        'pending': COLOR_PENDING,
        'verifying': COLOR_VERIFY,
        'correct': COLOR_CORRECT,
        'incorrect': COLOR_INCORRECT
    }
    color = colors.get(status, COLOR_PENDING)
    
    linewidth = 3 if status == 'verifying' else 1.5
    
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05",
        facecolor='white',
        edgecolor=color,
        linewidth=linewidth
    )
    ax.add_patch(box)
    
    # Claim number badge
    badge = Circle((x + 0.25, y + h - 0.25), 0.15,
                  facecolor=color, edgecolor=COLOR_TEXT, linewidth=1.5)
    ax.add_patch(badge)
    ax.text(x + 0.25, y + h - 0.25, str(claim_num),
            ha='center', va='center', fontsize=9,
            fontweight='bold', color='white')
    
    # Claim text
    ax.text(x + 0.5, y + h - 0.25, claim_text,
            ha='left', va='center', fontsize=8, color=COLOR_TEXT)
    
    # Status indicator
    status_symbols = {
        'pending': '○',
        'verifying': '🔍',
        'correct': '✓',
        'incorrect': '✗'
    }
    symbol = status_symbols.get(status, '○')
    ax.text(x + w - 0.2, y + h/2, symbol,
            ha='center', va='center', fontsize=14,
            fontweight='bold', color=color)
    
    # Score if available
    if score is not None:
        ax.text(x + w/2, y + 0.15, f"conf: {score:.0%}",
                ha='center', va='center', fontsize=7,
                color=COLOR_TEXT, style='italic')

def draw_arrow_flow(ax, x1, y1, x2, y2, label='', color=COLOR_TEXT):
    """Draw a flow arrow between stages."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->,head_width=0.4,head_length=0.6',
        color=color,
        linewidth=2.5,
        connectionstyle="arc3,rad=0.2"
    )
    ax.add_patch(arrow)
    
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.2, label, ha='center',
                fontsize=8, color=color, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def create_frame(stage, iteration=1):
    """Create a single frame of the chain of verification animation."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor(COLOR_BG)
    fig.patch.set_facecolor(COLOR_BG)
    
    # Title
    ax.text(7, 9.5, f"Chain of Verification - Iteration {iteration}", ha='center',
            fontsize=18, fontweight='bold', color=COLOR_TEXT)
    
    # Question
    question_box = FancyBboxPatch(
        (1.5, 8.3), 11, 0.8,
        boxstyle="round,pad=0.05",
        facecolor='white',
        edgecolor=COLOR_TEXT,
        linewidth=1.5
    )
    ax.add_patch(question_box)
    ax.text(7, 8.7, '❓ Question: "What are the nutritional benefits of our gluten-free pizza?"',
            ha='center', va='center', fontsize=10, color=COLOR_TEXT)
    
    # Stage indicators
    stage_y = 7.5
    draw_stage_box(ax, 1, stage_y, 2.5, 0.6, "1. Generate", COLOR_GENERATE, active=(stage == 1))
    draw_stage_box(ax, 4, stage_y, 2.5, 0.6, "2. Verify", COLOR_VERIFY, active=(stage in [2, 3, 4]))
    draw_stage_box(ax, 7, stage_y, 2.5, 0.6, "3. Correct", COLOR_CORRECT, active=(stage == 5))
    draw_stage_box(ax, 10, stage_y, 2.5, 0.6, "4. Output", COLOR_CORRECT, active=(stage == 6))
    
    # Flow arrows
    if stage >= 2:
        draw_arrow_flow(ax, 3.5, stage_y + 0.3, 4, stage_y + 0.3)
    if stage >= 5:
        draw_arrow_flow(ax, 6.5, stage_y + 0.3, 7, stage_y + 0.3)
    if stage >= 6:
        draw_arrow_flow(ax, 9.5, stage_y + 0.3, 10, stage_y + 0.3)
    
    # Claims area
    claims_y_start = 6
    claim_height = 0.8
    claim_spacing = 0.9
    
    claim_texts = [
        "High in protein (15g per slice)",
        "Rich in fiber (8g per serving)",
        "Contains essential vitamins B12"
    ]
    
    # Stage 0: Initial
    if stage == 0:
        ax.text(7, 4, "Preparing to generate claims...", ha='center',
                fontsize=12, color=COLOR_TEXT, style='italic')
    
    # Stage 1: Generate claims
    elif stage == 1:
        ax.text(7, 6.5, "Generated Claims:", ha='center',
                fontsize=11, fontweight='bold', color=COLOR_TEXT)
        
        for i, text in enumerate(claim_texts):
            y = claims_y_start - i * claim_spacing
            draw_claim_card(ax, 2, y - claim_height, 10, claim_height,
                          i + 1, text, 'pending')
        
        ax.text(7, 2.5, "Claims generated - ready for verification", ha='center',
                fontsize=10, color=COLOR_GENERATE, style='italic')
    
    # Stage 2: Verify claim 1
    elif stage == 2:
        ax.text(7, 6.5, "Verification in Progress:", ha='center',
                fontsize=11, fontweight='bold', color=COLOR_TEXT)
        
        # Claim 1 being verified
        draw_claim_card(ax, 2, claims_y_start - claim_height, 10, claim_height,
                       1, claim_texts[0], 'verifying')
        
        # Others pending
        for i in range(1, 3):
            y = claims_y_start - i * claim_spacing
            draw_claim_card(ax, 2, y - claim_height, 10, claim_height,
                          i + 1, claim_texts[i], 'pending')
        
        ax.text(7, 2.5, "🔍 Checking: Gluten-free flour protein content...", ha='center',
                fontsize=10, color=COLOR_VERIFY, style='italic')
    
    # Stage 3: Claim 1 verified (incorrect), verify claim 2
    elif stage == 3:
        ax.text(7, 6.5, "Verification in Progress:", ha='center',
                fontsize=11, fontweight='bold', color=COLOR_TEXT)
        
        # Claim 1 incorrect
        draw_claim_card(ax, 2, claims_y_start - claim_height, 10, claim_height,
                       1, claim_texts[0], 'incorrect', score=0.35)
        
        # Claim 2 being verified
        draw_claim_card(ax, 2, claims_y_start - 1 * claim_spacing - claim_height, 10, claim_height,
                       2, claim_texts[1], 'verifying')
        
        # Claim 3 pending
        draw_claim_card(ax, 2, claims_y_start - 2 * claim_spacing - claim_height, 10, claim_height,
                       3, claim_texts[2], 'pending')
        
        ax.text(7, 2.5, "✗ Claim 1 failed (actual: 8g protein) • 🔍 Verifying claim 2...", ha='center',
                fontsize=10, color=COLOR_TEXT, style='italic')
    
    # Stage 4: Claims 1-2 done, verify claim 3
    elif stage == 4:
        ax.text(7, 6.5, "Verification in Progress:", ha='center',
                fontsize=11, fontweight='bold', color=COLOR_TEXT)
        
        # Claim 1 incorrect
        draw_claim_card(ax, 2, claims_y_start - claim_height, 10, claim_height,
                       1, claim_texts[0], 'incorrect', score=0.35)
        
        # Claim 2 correct
        draw_claim_card(ax, 2, claims_y_start - 1 * claim_spacing - claim_height, 10, claim_height,
                       2, claim_texts[1], 'correct', score=0.92)
        
        # Claim 3 being verified
        draw_claim_card(ax, 2, claims_y_start - 2 * claim_spacing - claim_height, 10, claim_height,
                       3, claim_texts[2], 'verifying')
        
        ax.text(7, 2.5, "✓ Claim 2 verified • 🔍 Checking claim 3...", ha='center',
                fontsize=10, color=COLOR_TEXT, style='italic')
    
    # Stage 5: All verified, correction in progress
    elif stage == 5:
        ax.text(7, 6.5, "Corrections Applied:", ha='center',
                fontsize=11, fontweight='bold', color=COLOR_TEXT)
        
        # Claim 1 corrected
        corrected_text = "Moderate protein (8g per slice)"
        draw_claim_card(ax, 2, claims_y_start - claim_height, 10, claim_height,
                       1, corrected_text, 'correct', score=0.88)
        
        # Claim 2 correct (unchanged)
        draw_claim_card(ax, 2, claims_y_start - 1 * claim_spacing - claim_height, 10, claim_height,
                       2, claim_texts[1], 'correct', score=0.92)
        
        # Claim 3 correct
        draw_claim_card(ax, 2, claims_y_start - 2 * claim_spacing - claim_height, 10, claim_height,
                       3, claim_texts[2], 'correct', score=0.85)
        
        ax.text(7, 2.5, "✓ Claim 1 corrected with verified data", ha='center',
                fontsize=10, color=COLOR_CORRECT, fontweight='bold')
    
    # Stage 6: Final output
    elif stage == 6:
        ax.text(7, 6.5, "✓ Verified Output:", ha='center',
                fontsize=11, fontweight='bold', color=COLOR_SUCCESS)
        
        # All claims correct
        corrected_text = "Moderate protein (8g per slice)"
        draw_claim_card(ax, 2, claims_y_start - claim_height, 10, claim_height,
                       1, corrected_text, 'correct', score=0.88)
        draw_claim_card(ax, 2, claims_y_start - 1 * claim_spacing - claim_height, 10, claim_height,
                       2, claim_texts[1], 'correct', score=0.92)
        draw_claim_card(ax, 2, claims_y_start - 2 * claim_spacing - claim_height, 10, claim_height,
                       3, claim_texts[2], 'correct', score=0.85)
        
        # Summary box
        summary_box = FancyBboxPatch(
            (3, 1.8), 8, 0.7,
            boxstyle="round,pad=0.08",
            facecolor=COLOR_CORRECT,
            edgecolor=COLOR_TEXT,
            linewidth=2.5
        )
        ax.add_patch(summary_box)
        ax.text(7, 2.15, "All claims verified • 1 correction applied • 3/3 accurate", ha='center',
                fontsize=10, fontweight='bold', color='white')
    
    # Cost indicator
    cost_multiplier = stage + 1 if stage < 6 else 4
    cost_text = f"Token Cost: ~{cost_multiplier}× base (generate + {stage - 1 if stage > 1 else 0} verifications)"
    ax.text(7, 0.8, cost_text, ha='center',
            fontsize=9, color=COLOR_TEXT,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
    
    # Key
    if stage >= 1:
        key_y = 0.3
        ax.text(2.5, key_y, "○", color=COLOR_PENDING, fontsize=14, ha='center')
        ax.text(3.2, key_y, "Pending", fontsize=8, va='center', color=COLOR_TEXT)
        ax.text(5, key_y, "🔍", fontsize=11, ha='center')
        ax.text(5.9, key_y, "Verifying", fontsize=8, va='center', color=COLOR_TEXT)
        ax.text(7.5, key_y, "✓", color=COLOR_CORRECT, fontsize=14, ha='center', fontweight='bold')
        ax.text(8.2, key_y, "Verified", fontsize=8, va='center', color=COLOR_TEXT)
        ax.text(9.8, key_y, "✗", color=COLOR_INCORRECT, fontsize=14, ha='center', fontweight='bold')
        ax.text(10.6, key_y, "Failed", fontsize=8, va='center', color=COLOR_TEXT)
    
    return fig

# Generate frames
frames = []
frame_durations = []

stages_with_timing = [
    (0, 0.8),  # Initial
    (1, 1.3),  # Generate claims
    (2, 1.2),  # Verify claim 1
    (3, 1.2),  # Claim 1 failed, verify claim 2
    (4, 1.2),  # Verify claim 3
    (5, 1.5),  # Corrections
    (6, 2.0),  # Final output
]

for stage, duration in stages_with_timing:
    fig = create_frame(stage, iteration=1)
    fname = os.path.join(OUT_DIR, f'cov_frame_{len(frames):03d}.png')
    fig.tight_layout()
    fig.savefig(fname, dpi=100, bbox_inches='tight', facecolor=COLOR_BG)
    plt.close(fig)
    frames.append(imageio.v2.imread(fname))
    frame_durations.append(duration)

# Create GIF
imageio.mimsave(OUTPUT_GIF, frames, duration=frame_durations, loop=0)
print(f'✓ Saved {OUTPUT_GIF}')

# Clean up temp frames
for i in range(len(frames)):
    fname = os.path.join(OUT_DIR, f'cov_frame_{i:03d}.png')
    if os.path.exists(fname):
        os.remove(fname)

print(f'Generated {len(frames)} frames')
print(f'Total animation duration: {sum(frame_durations):.1f}s')
