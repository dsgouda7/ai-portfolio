"""
gen_ch09.py — generate visual assets for Ch.9 From Sequences to Attention
─────────────────────────────────────────────────────────────────────────
Produces:
  img/ch09-sequences-to-attention-needle.gif   interpretability meter advancing
                                               as attention weights become visible
  img/ch09-sequences-to-attention-progress-check.png  UnifiedAI constraint dashboard

Usage:
  python gen_scripts/gen_ch09.py
from the chapter root
(notes/01-ml/03_neural_networks/ch09_sequences_to_attention/).
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyArrowPatch

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
IMG_DIR = SCRIPT_DIR.parent / "img"
IMG_DIR.mkdir(exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────
RNG = np.random.default_rng(42)

BG       = "#1a1a2e"
FG       = "#e2e8f0"
PRIMARY  = "#1e3a8a"
SUCCESS  = "#15803d"
CAUTION  = "#b45309"
DANGER   = "#b91c1c"
INFO     = "#1d4ed8"
NEEDLE_C = "#38bdf8"
ACCENT   = "#f472b6"

# ══════════════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════════════

def _fig_dark(figsize=(10, 5)):
    fig = plt.figure(figsize=figsize, facecolor=BG)
    return fig


def _ax_dark(ax):
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color(FG)
        spine.set_linewidth(0.6)
    ax.tick_params(colors=FG, labelsize=8)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    return ax


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


# ── attention data ─────────────────────────────────────────────────────────────
FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
            "Population", "AveOccup", "Latitude", "Longitude"]

# Manually crafted 8x8 attention weights for the high-value coastal district
# Row i = token i attending to column j
# Based on realistic correlations: MedInc & Latitude dominate for high-value
_raw = np.array([
    [4.0, 1.0, 1.2, 0.8, 0.5, 0.6, 3.0, 1.8],  # MedInc attends strongly to self + Latitude
    [1.2, 4.0, 1.5, 1.0, 0.8, 0.9, 1.3, 1.2],  # HouseAge self-dominant
    [1.4, 1.5, 4.0, 2.5, 1.0, 1.8, 1.0, 0.8],  # AveRooms attends to AveBedrms
    [1.0, 1.2, 2.8, 4.0, 0.8, 2.0, 0.8, 0.7],  # AveBedrms attends to AveRooms
    [0.7, 1.0, 1.2, 0.9, 4.0, 2.5, 0.8, 1.2],  # Population attends to AveOccup
    [0.8, 1.1, 1.8, 2.0, 2.5, 4.0, 0.9, 1.0],  # AveOccup attends to density
    [3.2, 1.2, 1.0, 0.8, 0.7, 0.8, 4.0, 2.8],  # Latitude attends to MedInc + Longitude
    [1.8, 1.0, 0.8, 0.7, 1.0, 0.9, 2.8, 4.0],  # Longitude attends to Latitude
])
ATTN_WEIGHTS = np.apply_along_axis(softmax, 1, _raw)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  NEEDLE GIF — interpretability meter advancing
# ══════════════════════════════════════════════════════════════════════════════

# Stage definitions: attention weights revealing progressively
STAGES = [
    {
        "label": "LSTM baseline",
        "subtitle": "Hidden state = opaque black box",
        "interp_score": 5.0,
        "alpha_revealed": 0,       # rows of attention matrix revealed (0 = none)
        "caption": "Ch.6 LSTM final hidden state:\nall 8 features compressed to one vector.\nInterpretability: zero.",
        "constraint_color": DANGER,
    },
    {
        "label": "Dot product scores",
        "subtitle": "Q·Kᵀ computed — similarities visible",
        "interp_score": 30.0,
        "alpha_revealed": 2,
        "caption": "Dot product scores for MedInc query:\nMedInc 1.00, Latitude 0.60, Longitude -0.80.\nDirections of influence emerging.",
        "constraint_color": CAUTION,
    },
    {
        "label": "Softmax weights",
        "subtitle": "α = softmax(Ŝ) — probability distribution",
        "interp_score": 60.0,
        "alpha_revealed": 5,
        "caption": "MedInc attention weights:\nMedInc 0.492 · Latitude 0.370 · Longitude 0.138\nFeature importance quantified.",
        "constraint_color": INFO,
    },
    {
        "label": "Full 8×8 attention matrix",
        "subtitle": "All features attend to all features",
        "interp_score": 88.0,
        "alpha_revealed": 8,
        "caption": "Complete attention matrix computed.\nTop drivers: MedInc (49%) + Latitude (37%)\n→ Constraint #4 INTERPRETABILITY partial ✓",
        "constraint_color": SUCCESS,
    },
]

NEEDLE_MIN   = 0.0
NEEDLE_MAX   = 100.0
N_TRANSITION = 20
N_HOLD       = 10
TOTAL_STAGES = len(STAGES)


def _interp_score_to_angle(score: float) -> float:
    """Map interpretability score 0–100 to needle angle: 220° (left) to −40° (right)."""
    frac = (score - NEEDLE_MIN) / (NEEDLE_MAX - NEEDLE_MIN)
    return 220.0 - frac * 260.0


def _build_needle_frame(
    fig, ax_gauge, ax_attn, ax_text,
    interp_score: float, alpha_revealed: int,
    stage_label: str, stage_subtitle: str,
    caption: str, constraint_color: str,
):
    """Draw one frame: gauge needle + partial attention heatmap + caption."""

    # ── gauge arc background ──────────────────────────────────────────────────
    ax_gauge.cla()
    ax_gauge.set_facecolor(BG)
    ax_gauge.set_xlim(-1.3, 1.3)
    ax_gauge.set_ylim(-0.5, 1.3)
    ax_gauge.set_aspect("equal")
    ax_gauge.axis("off")

    # coloured arc segments: red→amber→blue→green
    arc_colors   = [DANGER, CAUTION, INFO, SUCCESS]
    arc_segments = 4
    arc_theta    = np.linspace(np.radians(220), np.radians(-40), 200)

    for seg in range(arc_segments):
        seg_start = int(len(arc_theta) * seg / arc_segments)
        seg_end   = int(len(arc_theta) * (seg + 1) / arc_segments)
        seg_theta = arc_theta[seg_start:seg_end]
        ax_gauge.plot(
            np.cos(seg_theta), np.sin(seg_theta),
            color=arc_colors[seg], linewidth=8, alpha=0.7, solid_capstyle="round",
        )

    # tick marks
    for score_tick in [0, 25, 50, 75, 100]:
        ang = np.radians(_interp_score_to_angle(score_tick))
        ax_gauge.plot(
            [0.80 * np.cos(ang), 0.95 * np.cos(ang)],
            [0.80 * np.sin(ang), 0.95 * np.sin(ang)],
            color=FG, linewidth=1.2, alpha=0.8,
        )
        ax_gauge.text(
            1.05 * np.cos(ang), 1.05 * np.sin(ang),
            str(score_tick), ha="center", va="center",
            fontsize=7, color=FG, alpha=0.7,
        )

    # needle
    needle_angle = np.radians(_interp_score_to_angle(interp_score))
    ax_gauge.plot(
        [0.0, 0.75 * np.cos(needle_angle)],
        [0.0, 0.75 * np.sin(needle_angle)],
        color=NEEDLE_C, linewidth=3, solid_capstyle="round",
        zorder=5,
    )
    ax_gauge.add_patch(plt.Circle((0, 0), 0.06, color=NEEDLE_C, zorder=6))

    # score text
    ax_gauge.text(
        0, -0.30,
        f"{interp_score:.0f}",
        ha="center", va="center",
        fontsize=20, fontweight="bold", color=NEEDLE_C,
    )
    ax_gauge.text(
        0, -0.48,
        "INTERPRETABILITY",
        ha="center", va="center",
        fontsize=7, color=FG, alpha=0.6,
    )

    # stage label
    ax_gauge.text(
        0, 1.22,
        stage_label,
        ha="center", va="top",
        fontsize=9, fontweight="bold", color=constraint_color,
    )
    ax_gauge.text(
        0, 1.10,
        stage_subtitle,
        ha="center", va="top",
        fontsize=7, color=FG, alpha=0.7,
    )

    # ── attention heatmap ─────────────────────────────────────────────────────
    ax_attn.cla()
    ax_attn.set_facecolor(BG)

    if alpha_revealed > 0:
        display = np.full((8, 8), np.nan)
        display[:alpha_revealed, :] = ATTN_WEIGHTS[:alpha_revealed, :]
        # mask for revealed portion
        masked = np.ma.masked_invalid(display)
        cmap = plt.cm.Blues.copy()
        cmap.set_bad(color="#0f0f1e")
        im = ax_attn.imshow(
            masked, cmap=cmap, vmin=0.0, vmax=0.6,
            aspect="auto", interpolation="nearest",
        )
        # highlight max in each revealed row
        for r in range(alpha_revealed):
            c = np.argmax(ATTN_WEIGHTS[r])
            rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                  fill=False, edgecolor=SUCCESS, linewidth=1.5)
            ax_attn.add_patch(rect)
    else:
        ax_attn.imshow(
            np.zeros((8, 8)), cmap="Blues", vmin=0, vmax=1,
            aspect="auto", interpolation="nearest",
        )
        ax_attn.text(
            3.5, 3.5, "?",
            ha="center", va="center",
            fontsize=40, color=FG, alpha=0.2, fontweight="bold",
        )

    ax_attn.set_xticks(range(8))
    ax_attn.set_xticklabels(FEATURES, fontsize=6, color=FG, rotation=45, ha="right")
    ax_attn.set_yticks(range(8))
    ax_attn.set_yticklabels(FEATURES, fontsize=6, color=FG)
    ax_attn.set_title("Attention Matrix  A[i,j]", fontsize=8, color=FG, pad=4)
    for spine in ax_attn.spines.values():
        spine.set_color(FG)
        spine.set_linewidth(0.4)
    ax_attn.tick_params(colors=FG)

    # ── caption ───────────────────────────────────────────────────────────────
    ax_text.cla()
    ax_text.set_facecolor(BG)
    ax_text.axis("off")
    ax_text.text(
        0.5, 0.5, caption,
        ha="center", va="center",
        fontsize=8, color=FG, alpha=0.85,
        linespacing=1.6,
        transform=ax_text.transAxes,
    )


def build_needle_gif():
    out_path = IMG_DIR / "ch09-sequences-to-attention-needle.gif"

    fig = _fig_dark(figsize=(12, 5))
    gs  = fig.add_gridspec(2, 3,
                           left=0.02, right=0.98,
                           top=0.95, bottom=0.05,
                           wspace=0.35, hspace=0.3)
    ax_gauge = fig.add_subplot(gs[:, 0])
    ax_attn  = fig.add_subplot(gs[:, 1])
    ax_text  = fig.add_subplot(gs[:, 2])

    frames = []

    for stage_idx, stage in enumerate(STAGES):
        prev_score = STAGES[stage_idx - 1]["interp_score"] if stage_idx > 0 else NEEDLE_MIN
        prev_alpha = STAGES[stage_idx - 1]["alpha_revealed"] if stage_idx > 0 else 0

        # transition frames
        for t in range(N_TRANSITION):
            frac = t / N_TRANSITION
            interp_score = prev_score + frac * (stage["interp_score"] - prev_score)
            alpha_revealed = prev_alpha if frac < 0.5 else stage["alpha_revealed"]
            frames.append((
                interp_score, alpha_revealed,
                stage["label"], stage["subtitle"],
                stage["caption"], stage["constraint_color"],
            ))

        # hold frames
        for _ in range(N_HOLD):
            frames.append((
                stage["interp_score"], stage["alpha_revealed"],
                stage["label"], stage["subtitle"],
                stage["caption"], stage["constraint_color"],
            ))

    def animate(i):
        interp_score, alpha_rev, lbl, sub, cap, col = frames[i]
        _build_needle_frame(
            fig, ax_gauge, ax_attn, ax_text,
            interp_score, alpha_rev, lbl, sub, cap, col,
        )

    ani = FuncAnimation(fig, animate, frames=len(frames), interval=80)
    ani.save(str(out_path), writer=PillowWriter(fps=12))
    plt.close(fig)
    print(f"  ✓  {out_path.name}  ({out_path.stat().st_size // 1024} KB)")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  PROGRESS CHECK PNG
# ══════════════════════════════════════════════════════════════════════════════

CONSTRAINTS = [
    {
        "id": "#1",
        "name": "ACCURACY",
        "target": "≤$28k MAE + ≥95% acc",
        "status": "IN PROGRESS",
        "detail": "Mechanism defined; training\nwith learned W_Q,K,V in Ch.10",
        "pct": 55,
        "color": CAUTION,
    },
    {
        "id": "#2",
        "name": "GENERALIZATION",
        "target": "Unseen districts",
        "status": "IN PROGRESS",
        "detail": "Order-invariant processing\nreduces positional bias",
        "pct": 50,
        "color": CAUTION,
    },
    {
        "id": "#3",
        "name": "MULTI-TASK",
        "target": "Value + segment",
        "status": "IN PROGRESS",
        "detail": "Same attention layer\nserves both heads",
        "pct": 50,
        "color": CAUTION,
    },
    {
        "id": "#4",
        "name": "INTERPRETABILITY",
        "target": "Explainable attribution",
        "status": "PARTIAL ✓",
        "detail": "MedInc 49% · Latitude 37%\nexplain high-value prediction",
        "pct": 75,
        "color": SUCCESS,
    },
    {
        "id": "#5",
        "name": "PRODUCTION",
        "target": "<100ms inference",
        "status": "IN PROGRESS",
        "detail": "Parallel O(n²) confirmed;\nlatency benchmarking in Ch.10",
        "pct": 45,
        "color": CAUTION,
    },
]


def build_progress_check():
    out_path = IMG_DIR / "ch09-sequences-to-attention-progress-check.png"

    fig, axes = plt.subplots(1, 5, figsize=(16, 4), facecolor=BG)
    fig.suptitle(
        "Ch.9 — UnifiedAI Constraint Dashboard",
        fontsize=13, color=FG, fontweight="bold", y=0.98,
    )

    for ax, c in zip(axes, CONSTRAINTS):
        ax.set_facecolor(BG)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # outer border
        border = mpatches.FancyBboxPatch(
            (0.05, 0.05), 0.90, 0.90,
            boxstyle="round,pad=0.02",
            linewidth=1.5,
            edgecolor=c["color"],
            facecolor=BG,
        )
        ax.add_patch(border)

        # constraint ID badge
        badge = mpatches.FancyBboxPatch(
            (0.28, 0.78), 0.44, 0.16,
            boxstyle="round,pad=0.02",
            linewidth=0,
            facecolor=c["color"],
            alpha=0.9,
        )
        ax.add_patch(badge)
        ax.text(0.50, 0.86, c["id"], ha="center", va="center",
                fontsize=14, fontweight="bold", color="#ffffff",
                transform=ax.transAxes)

        # name
        ax.text(0.50, 0.70, c["name"], ha="center", va="center",
                fontsize=9, fontweight="bold", color=c["color"],
                transform=ax.transAxes)

        # target
        ax.text(0.50, 0.60, c["target"], ha="center", va="center",
                fontsize=7, color=FG, alpha=0.7,
                transform=ax.transAxes)

        # progress bar track
        bar_x, bar_y, bar_w, bar_h = 0.12, 0.45, 0.76, 0.06
        track = mpatches.FancyBboxPatch(
            (bar_x, bar_y), bar_w, bar_h,
            boxstyle="round,pad=0.005",
            facecolor="#1e293b", edgecolor="none",
        )
        ax.add_patch(track)

        filled_w = bar_w * c["pct"] / 100.0
        fill = mpatches.FancyBboxPatch(
            (bar_x, bar_y), filled_w, bar_h,
            boxstyle="round,pad=0.005",
            facecolor=c["color"], edgecolor="none", alpha=0.9,
        )
        ax.add_patch(fill)

        ax.text(bar_x + bar_w + 0.03, bar_y + bar_h / 2,
                f"{c['pct']}%", ha="left", va="center",
                fontsize=7, color=c["color"], fontweight="bold",
                transform=ax.transAxes)

        # status pill
        status_bg_color = SUCCESS if "✓" in c["status"] else CAUTION
        status_patch = mpatches.FancyBboxPatch(
            (0.10, 0.30), 0.80, 0.11,
            boxstyle="round,pad=0.01",
            facecolor=status_bg_color, edgecolor="none", alpha=0.85,
        )
        ax.add_patch(status_patch)
        ax.text(0.50, 0.355, c["status"], ha="center", va="center",
                fontsize=7.5, fontweight="bold", color="#ffffff",
                transform=ax.transAxes)

        # detail text
        ax.text(0.50, 0.17, c["detail"], ha="center", va="center",
                fontsize=6.5, color=FG, alpha=0.75, linespacing=1.5,
                transform=ax.transAxes)

    # attention weights mini-panel beneath constraint #4
    ax4 = axes[3]
    ax4.text(0.50, 0.06,
             "MedInc: α=0.492   Latitude: α=0.370",
             ha="center", va="bottom",
             fontsize=6, color=SUCCESS, alpha=0.9,
             transform=ax4.transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(str(out_path), dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {out_path.name}  ({out_path.stat().st_size // 1024} KB)")


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating Ch.9 visual assets …")
    build_needle_gif()
    build_progress_check()
    print("Done.")
