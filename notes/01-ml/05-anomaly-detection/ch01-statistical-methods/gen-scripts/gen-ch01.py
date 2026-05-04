"""Generate ch01 Statistical Methods visuals for FraudShield.

Produces
--------
../img/ch01-statistical-methods-needle.gif
    Animated gauge needle showing recall sweeping from 0% (no detector)
    to 45% (statistical baseline @ 0.5% FPR). Target zone at 80%.

../img/ch01-statistical-methods-progress-check.png
    5-constraint status dashboard for FraudShield after Ch.1.

Usage
-----
    python gen_scripts/gen_ch01.py

Run from anywhere — paths are resolved relative to this file.
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────
SEED = 42
rng = np.random.default_rng(SEED)

BG       = "#1a1a2e"   # dark navy background
PANEL_BG = "#16213e"   # slightly lighter panel
PRIMARY  = "#1e3a8a"   # deep blue
SUCCESS  = "#15803d"   # green
CAUTION  = "#b45309"   # amber
DANGER   = "#b91c1c"   # red
INFO     = "#1d4ed8"   # medium blue
TEXT     = "#e2e8f0"   # near-white text
MUTED    = "#64748b"   # muted grey
GRID     = "#2d3748"   # subtle grid lines

IMG_DIR = Path(__file__).resolve().parent.parent / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# ── Shared helpers ─────────────────────────────────────────────────────────────

def _ease_in_out(t: float) -> float:
    """Smooth cosine ease-in/ease-out interpolation (t in [0, 1])."""
    t = max(0.0, min(1.0, t))
    return 0.5 - 0.5 * math.cos(math.pi * t)


def _draw_needle_frame(ax, value: float, stages: list, metric_label: str,
                       title: str, subtitle: str, target: float = 80.0,
                       danger_end: float = 45.0):
    """Draw a single needle-gauge frame onto *ax*."""
    ax.clear()
    ax.set_facecolor(BG)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.28, 1.28)
    ax.set_aspect("equal")
    ax.axis("off")

    r = 0.90

    # ── Full arc background ──────────────────────────────────────────────────
    theta_full = np.linspace(np.pi, 0, 300)
    ax.plot(np.cos(theta_full) * r, np.sin(theta_full) * r,
            color=GRID, lw=8, solid_capstyle="round")

    # Target zone: target% → 100% highlighted in green
    t_ang = np.pi * (1 - target / 100)
    theta_tgt = np.linspace(t_ang, 0, 100)
    ax.plot(np.cos(theta_tgt) * r, np.sin(theta_tgt) * r,
            color=SUCCESS, lw=8, alpha=0.50, solid_capstyle="round")

    # Danger zone: 0% → danger_end% highlighted in red
    d_ang = np.pi * (1 - danger_end / 100)
    theta_dng = np.linspace(np.pi, d_ang, 100)
    ax.plot(np.cos(theta_dng) * r, np.sin(theta_dng) * r,
            color=DANGER, lw=8, alpha=0.22, solid_capstyle="round")

    # ── Tick marks & labels ──────────────────────────────────────────────────
    ticks = [0, 20, 40, 45, 60, 80, 100]
    for pct in ticks:
        angle = np.pi * (1 - pct / 100)
        r_in, r_out = 0.78, 0.95
        ax.plot([np.cos(angle) * r_in, np.cos(angle) * r_out],
                [np.sin(angle) * r_in, np.sin(angle) * r_out],
                color=MUTED, lw=1.2)
        r_lbl = 0.67
        label = f"{pct}%"
        col = SUCCESS if pct == 80 else (CAUTION if pct == 45 else TEXT)
        fw  = "bold" if pct in (80, 45) else "normal"
        ax.text(np.cos(angle) * r_lbl, np.sin(angle) * r_lbl,
                label, ha="center", va="center", fontsize=7,
                color=col, fontweight=fw)

    # ── Target line at 80% ───────────────────────────────────────────────────
    angle_80 = np.pi * (1 - 80 / 100)
    ax.plot([0, np.cos(angle_80) * r * 1.06],
            [0, np.sin(angle_80) * r * 1.06],
            color=SUCCESS, lw=1.5, linestyle="--", alpha=0.85)
    ax.text(np.cos(angle_80) * r * 1.15, np.sin(angle_80) * r * 1.14,
            "TARGET", ha="center", va="bottom", fontsize=7,
            color=SUCCESS, fontweight="bold")

    # ── Stage waypoint dots ──────────────────────────────────────────────────
    for s in stages:
        s_angle = np.pi * (1 - s["value"] / 100)
        dot_col = s.get("color", INFO)
        ax.plot(np.cos(s_angle) * (r + 0.06), np.sin(s_angle) * (r + 0.06),
                "o", color=dot_col, markersize=7, zorder=8)
        ax.text(np.cos(s_angle) * (r + 0.20), np.sin(s_angle) * (r + 0.18),
                s["label"], ha="center", va="center",
                fontsize=7.5, color=dot_col, fontweight="bold")

    # ── Needle ───────────────────────────────────────────────────────────────
    angle = np.pi * (1 - value / 100)
    needle_r = 0.82
    ax.annotate(
        "",
        xy=(np.cos(angle) * needle_r, np.sin(angle) * needle_r),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", lw=3.5, color=CAUTION,
                        mutation_scale=18),
    )
    ax.add_patch(plt.Circle((0, 0), 0.055, color=CAUTION, zorder=10))

    # ── Current value ─────────────────────────────────────────────────────────
    ax.text(0, -0.10, f"{value:.1f}%",
            ha="center", va="center", fontsize=26,
            fontweight="bold", color=CAUTION)
    ax.text(0, -0.22, metric_label,
            ha="center", va="center", fontsize=11, color=MUTED)

    # ── Title / subtitle ─────────────────────────────────────────────────────
    ax.text(0, 1.22, title,
            ha="center", va="center", fontsize=13,
            fontweight="bold", color=TEXT)
    ax.text(0, 1.10, subtitle,
            ha="center", va="center", fontsize=9, color=MUTED)


# ── 1. NEEDLE GIF ──────────────────────────────────────────────────────────────

def make_needle_gif():
    """Produce ch01-statistical-methods-needle.gif.

    Shows recall sweeping from 0% (no detector) to 45% (Z-score
    statistical baseline calibrated at 0.5% FPR).
    """
    out_path = IMG_DIR / "ch01-statistical-methods-needle.gif"

    stages_info = [
        {"value":  0.0, "label": "No detector\n0%",    "color": DANGER},
        {"value": 45.0, "label": "Statistical\n45%",   "color": CAUTION},
        {"value": 80.0, "label": "Target\n80%",        "color": SUCCESS},
    ]

    HOLD_START  = 20   # frames at 0%
    SWEEP_FRAMES = 60  # frames sweeping 0% → 45%
    HOLD_END    = 28   # frames at 45%
    total = HOLD_START + SWEEP_FRAMES + HOLD_END

    fig, ax = plt.subplots(figsize=(5.5, 3.5), facecolor=BG)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

    shown_stages = [stages_info[0], stages_info[2]]  # 0% and target dots

    def _val(frame: int) -> float:
        if frame < HOLD_START:
            return 0.0
        elif frame < HOLD_START + SWEEP_FRAMES:
            t = (frame - HOLD_START) / SWEEP_FRAMES
            return 0.0 + 45.0 * _ease_in_out(t)
        else:
            return 45.0

    def _animate(frame: int):
        v = _val(frame)
        # After sweep completes, show all 3 stage dots
        cur_stages = shown_stages if frame < HOLD_START + SWEEP_FRAMES else stages_info
        _draw_needle_frame(
            ax, v, cur_stages,
            metric_label="Recall @ 0.5% FPR",
            title="FraudShield — Statistical Baseline",
            subtitle="Ch.1: Z-score / IQR / Mahalanobis  →  45% recall",
            target=80.0,
            danger_end=45.0,
        )
        return (ax,)

    ani = animation.FuncAnimation(
        fig, _animate, frames=total,
        interval=60, blit=False, repeat=True,
    )
    ani.save(str(out_path), writer="pillow", fps=16)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── 2. PROGRESS-CHECK DASHBOARD ────────────────────────────────────────────────

def make_progress_check():
    """Produce ch01-statistical-methods-progress-check.png.

    5-constraint FraudShield status dashboard after Ch.1.
    """
    out_path = IMG_DIR / "ch01-statistical-methods-progress-check.png"

    constraints = [
        {
            "id": "#1",
            "label": "DETECTION",
            "target": "≥80% recall",
            "current": "45% recall",
            "status": "partial",
            "pct": 45,
            "note": "Need isolation forest to break 45% ceiling",
        },
        {
            "id": "#2",
            "label": "PRECISION",
            "target": "≤0.5% FPR",
            "current": "0.3% FPR @ τ=2.0",
            "status": "met",
            "pct": 100,
            "note": "Calibrated threshold satisfies FPR budget",
        },
        {
            "id": "#3",
            "label": "REAL-TIME",
            "target": "<100ms",
            "current": "~0.01ms",
            "status": "met",
            "pct": 100,
            "note": "Z-score: 2 ops per feature, microseconds",
        },
        {
            "id": "#4",
            "label": "ADAPTABILITY",
            "target": "Handle drift",
            "current": "Static stats",
            "status": "blocked",
            "pct": 0,
            "note": "No rolling-window retraining yet",
        },
        {
            "id": "#5",
            "label": "EXPLAINABILITY",
            "target": "Justify flags",
            "current": "z=8.1σ on Amount",
            "status": "met",
            "pct": 100,
            "note": "Standard-deviation units are human-readable",
        },
    ]

    STATUS_COLOR = {
        "met":     SUCCESS,
        "partial": CAUTION,
        "blocked": DANGER,
    }
    STATUS_ICON = {
        "met":     "✓",
        "partial": "~",
        "blocked": "✗",
    }

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(constraints) + 1.2)
    ax.axis("off")

    # Title
    ax.text(5, len(constraints) + 0.75,
            "FraudShield — Constraint Status after Ch.1",
            ha="center", va="center", fontsize=15,
            fontweight="bold", color=TEXT)
    ax.text(5, len(constraints) + 0.35,
            "Statistical Anomaly Detection  ·  Credit Card Fraud  ·  284,807 transactions",
            ha="center", va="center", fontsize=9, color=MUTED)

    bar_width_max = 4.8
    bar_x = 4.2

    for i, c in enumerate(constraints):
        y = len(constraints) - 1 - i   # top-to-bottom layout
        col = STATUS_COLOR[c["status"]]
        icon = STATUS_ICON[c["status"]]

        # Row background
        bg_rect = mpatches.FancyBboxPatch(
            (0.1, y + 0.05), 9.8, 0.82,
            boxstyle="round,pad=0.02",
            linewidth=0,
            facecolor=PANEL_BG, zorder=1,
        )
        ax.add_patch(bg_rect)

        # Constraint ID + label
        ax.text(0.45, y + 0.47, c["id"],
                ha="center", va="center", fontsize=11,
                fontweight="bold", color=col)
        ax.text(1.05, y + 0.47, c["label"],
                ha="left", va="center", fontsize=10,
                fontweight="bold", color=TEXT)

        # Target
        ax.text(2.50, y + 0.68, "Target",
                ha="left", va="center", fontsize=7.5, color=MUTED)
        ax.text(2.50, y + 0.34, c["target"],
                ha="left", va="center", fontsize=8.5,
                fontweight="bold", color=TEXT)

        # Progress bar background
        ax.add_patch(mpatches.FancyBboxPatch(
            (bar_x, y + 0.22), bar_width_max, 0.50,
            boxstyle="round,pad=0.02",
            linewidth=0, facecolor=GRID, zorder=2,
        ))

        # Progress bar fill
        fill_w = bar_width_max * c["pct"] / 100
        if fill_w > 0:
            ax.add_patch(mpatches.FancyBboxPatch(
                (bar_x, y + 0.22), fill_w, 0.50,
                boxstyle="round,pad=0.02",
                linewidth=0, facecolor=col, alpha=0.75, zorder=3,
            ))

        # Current value label inside / beside bar
        ax.text(bar_x + bar_width_max + 0.12, y + 0.47,
                c["current"],
                ha="left", va="center", fontsize=8.5,
                color=col, fontweight="bold")

        # Status icon
        ax.text(9.60, y + 0.47, icon,
                ha="center", va="center", fontsize=16,
                fontweight="bold", color=col)

    # Legend
    for status, label in [("met", "Met"), ("partial", "Partial"), ("blocked", "Blocked")]:
        lc = STATUS_COLOR[status]
        li = STATUS_ICON[status]

    # Bottom annotation
    ax.text(5, -0.30,
            "Statistical baseline: 45% recall @ 0.5% FPR  ·  Ch.2 Isolation Forest targets 72%",
            ha="center", va="center", fontsize=8.5,
            color=MUTED, style="italic")

    fig.tight_layout(pad=0.5)
    fig.savefig(str(out_path), dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating Ch.1 Statistical Methods visuals …")
    make_needle_gif()
    make_progress_check()
    print("Done.")
