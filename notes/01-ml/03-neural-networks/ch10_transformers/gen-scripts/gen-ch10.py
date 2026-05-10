"""
gen_ch10.py — generate visual assets for Ch.10 Transformers & Attention
────────────────────────────────────────────────────────────────────────
Produces:
  img/ch10-transformers-needle.gif      UnifiedAI final chapter: all 5 constraints ✅
                                        MAE needle advancing to ≤$28k, accuracy ≥95%
  img/ch10-transformers-progress-check.png  Mission-complete constraint dashboard

Usage:
  python gen_scripts/gen_ch10.py
from the chapter root
(notes/01-ml/03_neural_networks/ch10_transformers/).
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
IMG_DIR    = SCRIPT_DIR.parent / "img"
IMG_DIR.mkdir(exist_ok=True)

# ── reproducibility ────────────────────────────────────────────────────────────
RNG = np.random.default_rng(42)

# ── palette ───────────────────────────────────────────────────────────────────
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

def _fig_dark(figsize=(12, 5)):
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


# ── stage definitions ─────────────────────────────────────────────────────────
# Each stage represents a transformer capability unlock, moving the needle
# from the Ch.9 single-head baseline toward the final transformer result.

FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
            "Population", "AveOccup", "Latitude", "Longitude"]

# Multi-head attention weights for h=4 heads (8×8 each)
# Head 1: income relationships | Head 2: geographic | Head 3: density | Head 4: stock
_raw_h1 = np.array([
    [4.0, 1.5, 2.0, 1.2, 0.5, 0.8, 1.0, 0.9],  # MedInc → rooms/age
    [1.8, 4.0, 1.2, 1.0, 0.6, 0.8, 0.9, 0.8],  # HouseAge self-dominant
    [2.2, 1.5, 4.0, 3.0, 0.8, 1.2, 0.7, 0.6],  # AveRooms → AveBedrms
    [1.5, 1.2, 3.2, 4.0, 0.7, 1.5, 0.6, 0.5],  # AveBedrms → AveRooms
    [0.6, 0.8, 1.0, 0.9, 4.0, 2.8, 0.7, 0.8],  # Population → AveOccup
    [0.9, 1.0, 1.4, 1.6, 3.0, 4.0, 0.8, 0.9],  # AveOccup → density
    [1.2, 1.0, 0.9, 0.8, 0.7, 0.9, 4.0, 1.5],  # Latitude
    [1.0, 0.9, 0.8, 0.7, 0.8, 0.9, 1.5, 4.0],  # Longitude
])
_raw_h2 = np.array([
    [2.0, 0.8, 0.9, 0.7, 0.5, 0.6, 3.5, 3.0],  # MedInc → Lat/Long
    [0.9, 2.5, 0.8, 0.7, 0.5, 0.6, 2.8, 2.5],  # HouseAge → geo
    [0.8, 0.9, 2.0, 1.2, 0.6, 0.7, 3.0, 2.8],  # AveRooms → geo
    [0.7, 0.8, 1.2, 2.0, 0.6, 0.8, 2.8, 2.5],  # AveBedrms → geo
    [0.6, 0.7, 0.8, 0.7, 2.5, 1.5, 3.0, 2.8],  # Population → geo
    [0.7, 0.8, 0.9, 0.9, 1.5, 2.5, 3.0, 2.8],  # AveOccup → geo
    [3.5, 2.8, 3.0, 2.8, 3.0, 3.0, 4.0, 4.5],  # Latitude attends geo+income
    [3.0, 2.5, 2.8, 2.5, 2.8, 2.8, 4.5, 4.0],  # Longitude attends geo+income
])
ATTN_H1 = np.apply_along_axis(softmax, 1, _raw_h1)
ATTN_H2 = np.apply_along_axis(softmax, 1, _raw_h2)

STAGES = [
    {
        "label": "Ch.9 baseline",
        "subtitle": "Single-head attention, identity projections",
        "mae": 34.0,
        "acc": 88.0,
        "head_shown": 0,         # 0 = no head, 1 = head1, 2 = head2
        "caption": "Single-head attention from Ch.9:\nMAE ≈ $34k · Accuracy ≈ 88%\nOne relationship type at a time.",
        "color": CAUTION,
        "all_green": False,
    },
    {
        "label": "Learned projections",
        "subtitle": "W_Q, W_K, W_V — optimal Q/K/V subspaces",
        "mae": 31.5,
        "acc": 90.5,
        "head_shown": 1,
        "caption": "Learned W_Q W_K W_V projections:\nHead 1 captures income relationships.\nMAE ≈ $31.5k · Accuracy ≈ 90.5%",
        "color": INFO,
        "all_green": False,
    },
    {
        "label": "Multi-head attention",
        "subtitle": "h=4 heads: income · geo · density · stock",
        "mae": 29.2,
        "acc": 93.0,
        "head_shown": 2,
        "caption": "Multi-head attention (h=4):\nHead 2 captures geographic proximity.\nMAE ≈ $29.2k · Accuracy ≈ 93%",
        "color": INFO,
        "all_green": False,
    },
    {
        "label": "Full transformer encoder",
        "subtitle": "Residuals · LayerNorm · N=6 blocks",
        "mae": 27.8,
        "acc": 95.3,
        "head_shown": 2,
        "caption":   "Full transformer (N=6, residuals, LN):\nMAE <= $28k [DONE] · Accuracy >= 95% [DONE]\nUnifiedAI mission: COMPLETE [DONE]",
        "color": SUCCESS,
        "all_green": True,
    },
]

MAE_TARGET  = 28.0
ACC_TARGET  = 95.0
N_TRANS     = 20
N_HOLD      = 12


def _needle_angle(value, vmin, vmax):
    """Map value in [vmin, vmax] to needle angle 220° (left) → -40° (right)."""
    frac = (value - vmin) / (vmax - vmin)
    return 220.0 - frac * 260.0


def _draw_gauge(ax, value, vmin, vmax, label, target, color, title):
    ax.cla()
    ax.set_facecolor(BG)
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-0.6, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")

    arc_theta = np.linspace(np.radians(220), np.radians(-40), 200)
    colors = [DANGER, CAUTION, INFO, SUCCESS]
    for s in range(4):
        sl = arc_theta[s * 50: (s + 1) * 50]
        ax.plot(np.cos(sl), np.sin(sl), color=colors[s], linewidth=9,
                alpha=0.75, solid_capstyle="round")

    # target line
    t_ang = np.radians(_needle_angle(target, vmin, vmax))
    ax.plot([0.70 * np.cos(t_ang), 0.98 * np.cos(t_ang)],
            [0.70 * np.sin(t_ang), 0.98 * np.sin(t_ang)],
            color=ACCENT, linewidth=2.5, zorder=7)
    ax.text(1.05 * np.cos(t_ang), 1.05 * np.sin(t_ang),
            f"target\n{target}", ha="center", va="center",
            fontsize=6, color=ACCENT, alpha=0.9)

    # tick marks
    for tick in np.linspace(vmin, vmax, 5):
        a = np.radians(_needle_angle(tick, vmin, vmax))
        ax.plot([0.78 * np.cos(a), 0.93 * np.cos(a)],
                [0.78 * np.sin(a), 0.93 * np.sin(a)],
                color=FG, linewidth=1.0, alpha=0.7)
        ax.text(1.03 * np.cos(a), 1.03 * np.sin(a),
                f"{tick:.0f}", ha="center", va="center",
                fontsize=6, color=FG, alpha=0.6)

    # needle
    ang = np.radians(_needle_angle(value, vmin, vmax))
    ax.plot([0, 0.76 * np.cos(ang)], [0, 0.76 * np.sin(ang)],
            color=NEEDLE_C, linewidth=3, solid_capstyle="round", zorder=8)
    ax.add_patch(plt.Circle((0, 0), 0.07, color=NEEDLE_C, zorder=9))

    ax.text(0, -0.35, f"{value:.1f}", ha="center", va="center",
            fontsize=18, fontweight="bold", color=color)
    ax.text(0, -0.52, label, ha="center", va="center",
            fontsize=7, color=FG, alpha=0.65)
    ax.text(0, 1.38, title, ha="center", va="top",
            fontsize=9, fontweight="bold", color=color)


def _draw_head_heatmap(ax, stage_idx):
    ax.cla()
    ax.set_facecolor(BG)
    if stage_idx == 0:
        ax.imshow(np.zeros((8, 8)), cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.text(3.5, 3.5, "single\nhead", ha="center", va="center",
                fontsize=11, color=FG, alpha=0.35, fontweight="bold")
    elif stage_idx == 1:
        ax.imshow(ATTN_H1, cmap="Blues", vmin=0, vmax=0.5, aspect="auto")
        ax.set_title("Head 1 — income", fontsize=7, color=FG, pad=3)
    else:
        combined = (ATTN_H1 + ATTN_H2) / 2
        ax.imshow(combined, cmap="Blues", vmin=0, vmax=0.45, aspect="auto")
        ax.set_title("Heads 1+2 combined", fontsize=7, color=FG, pad=3)

    ax.set_xticks(range(8))
    ax.set_xticklabels(FEATURES, fontsize=5, color=FG, rotation=45, ha="right")
    ax.set_yticks(range(8))
    ax.set_yticklabels(FEATURES, fontsize=5, color=FG)
    for sp in ax.spines.values():
        sp.set_color(FG)
        sp.set_linewidth(0.4)
    ax.tick_params(colors=FG)


def _draw_caption(ax, text, color):
    ax.cla()
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center",
            fontsize=8, color=FG, alpha=0.88,
            linespacing=1.7, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#0f0f1e",
                      edgecolor=color, linewidth=1.2, alpha=0.7))


# ══════════════════════════════════════════════════════════════════════════════
# 1. NEEDLE GIF
# ══════════════════════════════════════════════════════════════════════════════

def build_needle_gif():
    out = IMG_DIR / "ch10-transformers-needle.gif"

    fig = _fig_dark(figsize=(14, 5))
    gs  = fig.add_gridspec(1, 4, left=0.02, right=0.98,
                           top=0.92, bottom=0.06, wspace=0.3)
    ax_mae  = fig.add_subplot(gs[0])
    ax_acc  = fig.add_subplot(gs[1])
    ax_attn = fig.add_subplot(gs[2])
    ax_cap  = fig.add_subplot(gs[3])

    frames = []
    for idx, stage in enumerate(STAGES):
        prev_mae = STAGES[idx - 1]["mae"] if idx > 0 else STAGES[0]["mae"]
        prev_acc = STAGES[idx - 1]["acc"] if idx > 0 else STAGES[0]["acc"]
        prev_hs  = STAGES[idx - 1]["head_shown"] if idx > 0 else 0

        for t in range(N_TRANS):
            frac = t / N_TRANS
            frames.append({
                "mae":       prev_mae  + frac * (stage["mae"]  - prev_mae),
                "acc":       prev_acc  + frac * (stage["acc"]  - prev_acc),
                "head_shown": prev_hs if frac < 0.5 else stage["head_shown"],
                "label":     stage["label"],
                "caption":   stage["caption"],
                "color":     stage["color"],
            })
        for _ in range(N_HOLD):
            frames.append({
                "mae":       stage["mae"],
                "acc":       stage["acc"],
                "head_shown": stage["head_shown"],
                "label":     stage["label"],
                "caption":   stage["caption"],
                "color":     stage["color"],
            })

    def animate(i):
        f = frames[i]
        mae_col = SUCCESS if f["mae"] <= MAE_TARGET else f["color"]
        acc_col = SUCCESS if f["acc"] >= ACC_TARGET else f["color"]

        _draw_gauge(ax_mae, f["mae"], 70.0, 25.0,
                    "MAE  $k", MAE_TARGET, mae_col,
                    "Regression ↓")
        _draw_gauge(ax_acc, f["acc"], 85.0, 100.0,
                    "Accuracy %", ACC_TARGET, acc_col,
                    "Classification ↑")
        _draw_head_heatmap(ax_attn, f["head_shown"])
        _draw_caption(ax_cap, f["caption"], f["color"])

        # super-title
        fig.suptitle(
            f'Ch.10 — Transformers  ·  {f["label"]}',
            fontsize=11, fontweight="bold", color=f["color"], y=0.98,
        )

    ani = FuncAnimation(fig, animate, frames=len(frames), interval=80)
    ani.save(str(out), writer=PillowWriter(fps=12))
    plt.close(fig)
    print(f"  ✓  {out.name}  ({out.stat().st_size // 1024} KB)")


# ══════════════════════════════════════════════════════════════════════════════
# 2. PROGRESS CHECK PNG — mission complete dashboard
# ══════════════════════════════════════════════════════════════════════════════

CONSTRAINTS = [
    {
        "id": "#1", "name": "ACCURACY",
        "target": "≤$28k MAE\n≥95% acc",
        "status": "COMPLETE ✅",
        "detail": "MAE $27.8k · Acc 95.3%\nMulti-head (h=8) captures\nall 4 relationship types",
        "pct": 100, "color": SUCCESS,
    },
    {
        "id": "#2", "name": "GENERALIZATION",
        "target": "Unseen districts\n+ faces",
        "status": "COMPLETE ✅",
        "detail": "PE + dropout + residuals\nValidated on held-out\nCA districts & CelebA",
        "pct": 100, "color": SUCCESS,
    },
    {
        "id": "#3", "name": "MULTI-TASK",
        "target": "Value + segment",
        "status": "COMPLETE ✅",
        "detail": "Shared encoder backbone\nRegression head (mean-pool)\nClassification head (CLS)",
        "pct": 100, "color": SUCCESS,
    },
    {
        "id": "#4", "name": "INTERPRETABILITY",
        "target": "Explainable\nattribution",
        "status": "COMPLETE ✅",
        "detail": "Attention weights A[i,j]\nHead 1: MedInc→value\nHead 2: Lat/Long→location",
        "pct": 100, "color": SUCCESS,
    },
    {
        "id": "#5", "name": "PRODUCTION",
        "target": "<100ms\ninference",
        "status": "COMPLETE ✅",
        "detail": "45ms CPU / <10ms GPU\nFull parallelism achieved\nvs 180ms LSTM baseline",
        "pct": 100, "color": SUCCESS,
    },
]


def build_progress_check():
    out = IMG_DIR / "ch10-transformers-progress-check.png"

    fig, axes = plt.subplots(1, 5, figsize=(18, 5), facecolor=BG)
    fig.suptitle(
        "Ch.10 — UnifiedAI  ·  ALL 5 CONSTRAINTS COMPLETE  ✅",
        fontsize=14, color=SUCCESS, fontweight="bold", y=0.99,
    )

    for ax, c in zip(axes, CONSTRAINTS):
        ax.set_facecolor(BG)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # card background
        border = mpatches.FancyBboxPatch(
            (0.04, 0.04), 0.92, 0.92,
            boxstyle="round,pad=0.03",
            linewidth=2.5, edgecolor=c["color"],
            facecolor="#0d0d1e", alpha=0.9,
            transform=ax.transAxes, zorder=1,
        )
        ax.add_patch(border)

        # constraint ID badge
        ax.text(0.13, 0.88, c["id"], ha="center", va="center",
                fontsize=18, fontweight="bold", color=c["color"],
                transform=ax.transAxes, zorder=2)

        # constraint name
        ax.text(0.5, 0.80, c["name"], ha="center", va="center",
                fontsize=10, fontweight="bold", color=FG,
                transform=ax.transAxes, zorder=2)

        # target text
        ax.text(0.5, 0.66, c["target"], ha="center", va="center",
                fontsize=8, color=FG, alpha=0.7, linespacing=1.5,
                transform=ax.transAxes, zorder=2)

        # status badge
        status_bg = SUCCESS
        ax.text(0.5, 0.52, c["status"], ha="center", va="center",
                fontsize=9, fontweight="bold", color="#ffffff",
                transform=ax.transAxes, zorder=3,
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor=status_bg, alpha=0.9))

        # progress bar
        bar_w = 0.80
        bar_x = 0.10
        bar_y = 0.38
        bar_h = 0.06
        ax.add_patch(mpatches.Rectangle(
            (bar_x, bar_y), bar_w, bar_h,
            facecolor="#1e1e3e", transform=ax.transAxes, zorder=2,
        ))
        ax.add_patch(mpatches.Rectangle(
            (bar_x, bar_y), bar_w * c["pct"] / 100, bar_h,
            facecolor=c["color"], alpha=0.9,
            transform=ax.transAxes, zorder=3,
        ))
        ax.text(0.5, bar_y + bar_h / 2, f"{c['pct']}%",
                ha="center", va="center", fontsize=7,
                color="#ffffff", fontweight="bold",
                transform=ax.transAxes, zorder=4)

        # detail text
        ax.text(0.5, 0.20, c["detail"], ha="center", va="center",
                fontsize=7, color=FG, alpha=0.8, linespacing=1.6,
                transform=ax.transAxes, zorder=2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(out), dpi=140, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"  ✓  {out.name}  ({out.stat().st_size // 1024} KB)")


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Ch.10 — Generating visual assets …")
    build_needle_gif()
    build_progress_check()
    print("Done.")
