"""
gen_ch01.py — Deterministic image generator for Ch.1 Pandas & EDA

Produces two assets in ../img/:
  1. ch01-pandas-eda-needle.gif  — Animated MAE needle (174k → ~140k, 10 frames)
  2. ch01-pandas-eda-progress-check.png — Constraint dashboard (5 constraints, current status)

Run from the chapter directory:
    cd notes/01-ml/00_data_fundamentals/ch01_pandas_eda
    python gen_scripts/gen_ch01.py

Dependencies: matplotlib, numpy, PIL/Pillow
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Arc
from PIL import Image
import io

# ── Seed everything ──────────────────────────────────────────────────────────
np.random.seed(42)

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHAPTER_DIR = os.path.dirname(SCRIPT_DIR)
IMG_DIR = os.path.join(CHAPTER_DIR, 'img')
os.makedirs(IMG_DIR, exist_ok=True)

# ── Colour palette (canonical dark theme) ────────────────────────────────────
BG       = '#1a1a2e'
PRIMARY  = '#1e3a8a'
SUCCESS  = '#15803d'
CAUTION  = '#b45309'
DANGER   = '#b91c1c'
INFO     = '#1d4ed8'
MUTED    = '#475569'
TEXT     = '#e2e8f0'
GOLD     = '#f59e0b'


# ─────────────────────────────────────────────────────────────────────────────
# 1. NEEDLE ANIMATION  (ch01-pandas-eda-needle.gif)
# ─────────────────────────────────────────────────────────────────────────────

def draw_needle_frame(ax, mae_value: float, frame_label: str, max_mae: float = 200.0,
                      target_mae: float = 95.0) -> None:
    """Draw a single speedometer-style MAE needle frame onto ax."""
    ax.set_facecolor(BG)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.3, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')

    # Arc background (180° sweep, left=danger, right=safe)
    # Danger zone: 150–200k (angle 0° to 60°)
    # Caution zone: 95–150k (angle 60° to 120°)
    # Safe zone: 0–95k (angle 120° to 180°)
    for start_deg, end_deg, color, label in [
        (0,   60,  DANGER,  ''),
        (60,  120, CAUTION, ''),
        (120, 180, SUCCESS, ''),
    ]:
        arc = Arc((0, 0), 2.0, 2.0, angle=0,
                  theta1=start_deg, theta2=end_deg,
                  color=color, lw=18, zorder=2)
        ax.add_patch(arc)

    # Thin inner arc (track)
    inner = Arc((0, 0), 1.6, 1.6, angle=0, theta1=0, theta2=180,
                color=MUTED, lw=2, alpha=0.4, zorder=3)
    ax.add_patch(inner)

    # Tick marks at 50k intervals
    for mae_tick in [0, 50, 95, 100, 150, 200]:
        angle_deg = 180 - (mae_tick / max_mae) * 180
        angle_rad = np.deg2rad(angle_deg)
        r_outer, r_inner = 1.05, 0.90
        x1, y1 = r_outer * np.cos(angle_rad), r_outer * np.sin(angle_rad)
        x2, y2 = r_inner * np.cos(angle_rad), r_inner * np.sin(angle_rad)
        color = SUCCESS if mae_tick <= target_mae else (CAUTION if mae_tick <= 150 else DANGER)
        ax.plot([x1, x2], [y1, y2], color=color, lw=2, zorder=4)
        # Tick label
        r_lbl = 1.18
        ax.text(r_lbl * np.cos(angle_rad), r_lbl * np.sin(angle_rad),
                f'{mae_tick}k', color=TEXT, fontsize=6, ha='center', va='center',
                fontweight='bold', zorder=5)

    # Target line at 95k
    target_angle = np.deg2rad(180 - (target_mae / max_mae) * 180)
    ax.plot([0.0, 1.05 * np.cos(target_angle)],
            [0.0, 1.05 * np.sin(target_angle)],
            color=SUCCESS, lw=1.5, ls='--', alpha=0.8, zorder=4)
    ax.text(1.25 * np.cos(target_angle), 1.25 * np.sin(target_angle),
            'Target\n95k', color=SUCCESS, fontsize=5.5, ha='center', va='center', zorder=5)

    # Needle
    needle_angle_deg = 180 - (mae_value / max_mae) * 180
    needle_angle_rad = np.deg2rad(needle_angle_deg)
    nx, ny = 0.85 * np.cos(needle_angle_rad), 0.85 * np.sin(needle_angle_rad)
    needle_color = (SUCCESS if mae_value <= target_mae else
                    CAUTION if mae_value <= 150 else DANGER)
    ax.annotate('', xy=(nx, ny), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=needle_color,
                                lw=2.5, mutation_scale=18), zorder=6)

    # Hub circle
    hub = plt.Circle((0, 0), 0.06, color=TEXT, zorder=7)
    ax.add_patch(hub)

    # MAE readout
    ax.text(0, -0.18, f'MAE = ${mae_value:,.0f}k',
            color=needle_color, fontsize=13, ha='center', va='center',
            fontweight='bold', zorder=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=BG, edgecolor=needle_color, lw=1.5))

    # Frame label
    ax.text(0, 1.15, frame_label, color=TEXT, fontsize=8,
            ha='center', va='center', alpha=0.8, zorder=8)

    # Title
    ax.text(0, 1.28, 'RealtyML · Data Quality Progress',
            color=TEXT, fontsize=9, ha='center', va='center', fontweight='bold', zorder=8)


def make_needle_gif(out_path: str, n_frames: int = 10) -> None:
    """Generate animated needle GIF (10 frames: 174k sweeping toward ~140k)."""
    start_mae = 174.0
    end_mae   = 140.0  # post-EDA cleaning estimate

    frames_pil = []
    for i in range(n_frames):
        t = i / (n_frames - 1)
        # Ease-in-out interpolation
        t_eased = t * t * (3 - 2 * t)
        mae = start_mae + (end_mae - start_mae) * t_eased

        if i == 0:
            label = f'Before EDA cleaning  (Portland deployment)'
        elif i == n_frames - 1:
            label = f'After EDA cleaning  (outliers + imputation fixed)'
        else:
            label = f'EDA cleaning in progress... ({int(t * 100)}%)'

        fig, ax = plt.subplots(figsize=(5, 3.2), facecolor=BG)
        draw_needle_frame(ax, mae, label)
        plt.tight_layout(pad=0.3)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, facecolor=BG, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        frames_pil.append(Image.open(buf).convert('RGBA'))

    # Save as GIF (200ms per frame, last frame holds 1s)
    durations = [200] * (n_frames - 1) + [1000]
    frames_pil[0].save(
        out_path,
        save_all=True,
        append_images=frames_pil[1:],
        optimize=False,
        duration=durations,
        loop=0,
    )
    print(f'[OK] Needle GIF -> {out_path}  ({n_frames} frames)')


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONSTRAINT DASHBOARD  (ch01-pandas-eda-progress-check.png)
# ─────────────────────────────────────────────────────────────────────────────

CONSTRAINTS = [
    ('#1 ACCURACY',         '<95k MAE',                 'partial',  '174k → ~140k (cleaning partial)'),
    ('#2 GENERALIZATION',   'CA → Portland → nationwide','partial',  'Distribution mismatch found, not yet fixed'),
    ('#3 DATA QUALITY',     'Fix all corruption',        'unlocked', 'Zero-fill reversed · IQR outliers flagged · KNN selected'),
    ('#4 AUDITABILITY',     'Document every fix',        'unlocked', 'Before/after MAE table · every change tracked'),
    ('#5 PRODUCTION-READY', '<100ms inference',          'blocked',  'No pipeline yet — addressed in Ch.3 + Regression track'),
]

STATUS_STYLE = {
    'unlocked': dict(bg=SUCCESS,  icon='UNLOCKED', text='#ffffff'),
    'partial':  dict(bg=CAUTION,  icon='PARTIAL',  text='#ffffff'),
    'blocked':  dict(bg=DANGER,   icon='BLOCKED',  text='#ffffff'),
}


def make_progress_check(out_path: str) -> None:
    """Generate a 5-constraint dashboard PNG."""
    fig_h = 0.9 + len(CONSTRAINTS) * 1.1
    fig, ax = plt.subplots(figsize=(10, fig_h), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, fig_h)
    ax.axis('off')

    # Title
    ax.text(5, fig_h - 0.45,
            'Ch.1 — Pandas & EDA  |  Constraint Progress',
            color=TEXT, fontsize=14, fontweight='bold', ha='center', va='center')
    ax.text(5, fig_h - 0.75,
            'RealtyML Fix · California Housing · Target: Portland MAE < 95k',
            color=MUTED, fontsize=9, ha='center', va='center', style='italic')

    # Column headers
    y_hdr = fig_h - 1.05
    for x, label in [(0.3, '#'), (1.2, 'Constraint'), (4.0, 'Target'),
                     (6.5, 'Status'), (8.5, 'This Chapter')]:
        ax.text(x, y_hdr, label, color=MUTED, fontsize=8,
                fontweight='bold', va='center', ha='left')
    ax.axhline(y=y_hdr - 0.18, xmin=0.02, xmax=0.98,
               color=MUTED, lw=0.5, alpha=0.5)

    # Rows
    for idx, (name, target, status, note) in enumerate(CONSTRAINTS):
        y = fig_h - 1.05 - (idx + 1) * 1.05
        style = STATUS_STYLE[status]

        # Row background (alternating)
        row_bg = '#1e2a45' if idx % 2 == 0 else '#162035'
        row = mpatches.FancyBboxPatch((0.05, y - 0.42), 9.9, 0.84,
                                       boxstyle='round,pad=0.05',
                                       facecolor=row_bg, edgecolor='none', zorder=1)
        ax.add_patch(row)

        # Status badge
        badge = mpatches.FancyBboxPatch((6.1, y - 0.28), 1.55, 0.56,
                                         boxstyle='round,pad=0.08',
                                         facecolor=style['bg'], edgecolor='none', zorder=2)
        ax.add_patch(badge)
        ax.text(6.875, y, style['icon'],
                color=style['text'], fontsize=7.5, fontweight='bold',
                ha='center', va='center', zorder=3)

        # Text columns
        ax.text(0.30, y, name,     color=TEXT,  fontsize=8.5, fontweight='bold', va='center')
        ax.text(4.0,  y, target,   color=TEXT,  fontsize=8,   va='center')
        ax.text(8.0,  y, note,     color=MUTED, fontsize=7,   va='center', wrap=True)

    # Footer
    ax.text(5, 0.25,
            'Generated by gen_scripts/gen_ch01.py · seed=42 · California Housing dataset',
            color=MUTED, fontsize=7, ha='center', va='center', style='italic')

    plt.tight_layout(pad=0.4)
    fig.savefig(out_path, dpi=120, facecolor=BG, bbox_inches='tight')
    plt.close(fig)
    print(f'[OK] Progress check PNG -> {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    needle_path  = os.path.join(IMG_DIR, 'ch01-pandas-eda-needle.gif')
    progress_path = os.path.join(IMG_DIR, 'ch01-pandas-eda-progress-check.png')

    print('Generating Ch.1 visual assets...')
    make_needle_gif(needle_path, n_frames=10)
    make_progress_check(progress_path)
    print('Done. Assets written to:', IMG_DIR)
