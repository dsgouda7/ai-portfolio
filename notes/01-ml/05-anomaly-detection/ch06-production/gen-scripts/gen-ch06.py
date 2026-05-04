"""Gen script for Ch.6 — Production & Real-Time Inference.

Produces:
  img/ch06-production-needle.gif  — FraudShield mission complete, recall arc
  img/ch06-production-progress-check.png — all 5 constraints satisfied
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
rng = np.random.default_rng(SEED)

BG = "#1a1a2e"          # dark background
PRIMARY = "#1e3a8a"     # deep blue
SUCCESS = "#15803d"     # green
CAUTION = "#b45309"     # amber
DANGER  = "#b91c1c"     # red
INFO    = "#1d4ed8"     # lighter blue
LIGHT   = "#e2e8f0"
GREY    = "#64748b"

OUT_DIR = Path(__file__).resolve().parent.parent / "img"
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Needle GIF — recall arc: deploy 82% → drift to 71% → retrain → 81%
# ---------------------------------------------------------------------------

STAGES = [
    {
        "label": "Deployed (month 0)",
        "recall": 0.820,
        "psi": 0.09,
        "caption": "Model goes live. Recall 82% matches offline evaluation.",
    },
    {
        "label": "Drift emerging (month 1.5)",
        "recall": 0.780,
        "psi": 0.19,
        "caption": "Recall slips to 78%. PSI 0.19 shows slight feature drift — borderline.",
    },
    {
        "label": "Drift significant (month 2)",
        "recall": 0.710,
        "psi": 0.31,
        "caption": "TRIGGER: recall < 78% AND PSI > 0.25. Retraining pipeline fires.",
    },
    {
        "label": "Retrained + deployed (month 2.5)",
        "recall": 0.810,
        "psi": 0.07,
        "caption": "Blue-green cutover complete. Recall restored to 81%. All 5 constraints met.",
    },
]

RECALL_MIN = 0.60
RECALL_MAX = 0.85


def _ease(t: float) -> float:
    import math
    t = max(0.0, min(1.0, float(t)))
    return 0.5 - 0.5 * math.cos(math.pi * t)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def build_needle_gif() -> None:
    fig = plt.figure(figsize=(12, 7), facecolor=BG)
    fig.suptitle("FraudShield Ch.6 — Production: Drift Detection & Model Retraining",
                 fontsize=13, color=LIGHT, y=0.98)

    # Layout: left = recall gauge, right-top = PSI bar, bottom = history
    gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.0],
                          hspace=0.38, wspace=0.32,
                          left=0.07, right=0.96, top=0.92, bottom=0.10)
    ax_gauge = fig.add_subplot(gs[0, 0])
    ax_psi   = fig.add_subplot(gs[0, 1])
    ax_hist  = fig.add_subplot(gs[1, :])

    caption_txt = fig.text(0.5, 0.02, "", ha="center", va="bottom",
                            fontsize=10, color=LIGHT,
                            bbox=dict(boxstyle="round,pad=0.4", fc="#0f172a", ec=GREY))

    frames_per_seg = 12
    n_frames = (len(STAGES) - 1) * frames_per_seg + 6

    def _get_state(frame):
        n_segs = len(STAGES) - 1
        seg = min(frame // frames_per_seg, n_segs - 1)
        local = (frame % frames_per_seg) / max(frames_per_seg - 1, 1)
        local = _ease(local)
        s0, s1 = STAGES[seg], STAGES[seg + 1]
        return {
            "label": s0["label"] if local < 0.5 else s1["label"],
            "recall": _lerp(s0["recall"], s1["recall"], local),
            "psi":    _lerp(s0["psi"],    s1["psi"],    local),
            "caption": s0["caption"] if local < 0.5 else s1["caption"],
        }, seg

    def draw_gauge(ax, recall):
        ax.clear()
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(0, 1)
        ax.set_title("Recall", fontsize=12, color=LIGHT, pad=6)

        # track
        ax.hlines(0.45, 0.05, 0.95, color=GREY, linewidth=12, alpha=0.5)
        ratio = (recall - RECALL_MIN) / (RECALL_MAX - RECALL_MIN)
        ratio = max(0.0, min(1.0, ratio))
        fill_col = SUCCESS if recall >= 0.78 else (CAUTION if recall >= 0.70 else DANGER)
        ax.hlines(0.45, 0.05, 0.05 + 0.90 * ratio, color=fill_col, linewidth=12)
        nx = 0.05 + 0.90 * ratio
        ax.arrow(nx, 0.88, 0, -0.28, width=0.010, head_width=0.048, head_length=0.07,
                 color=DANGER, length_includes_head=True)

        ax.text(0.05, 0.18, f"{RECALL_MIN:.0%}", ha="left",  color=GREY, fontsize=9)
        ax.text(0.95, 0.18, f"{RECALL_MAX:.0%}", ha="right", color=GREY, fontsize=9)
        ax.text(0.50, 0.40, f"{RECALL_MIN + (RECALL_MAX-RECALL_MIN)*0.5:.0%}",
                ha="center", color=GREY, fontsize=8)

        ax.text(0.50, 0.72, f"Recall\n{recall:.1%}", ha="center", va="center",
                fontsize=14, fontweight="bold", color=fill_col,
                bbox=dict(boxstyle="round,pad=0.35", fc="#0f172a", ec=GREY))
        # 80% target line
        tgt = (0.80 - RECALL_MIN) / (RECALL_MAX - RECALL_MIN)
        tx = 0.05 + 0.90 * tgt
        ax.vlines(tx, 0.30, 0.62, color=CAUTION, linewidth=2, linestyle="--")
        ax.text(tx, 0.22, "80%\ntarget", ha="center", color=CAUTION, fontsize=8)

    def draw_psi(ax, psi):
        ax.clear()
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(0, 1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("PSI (Amount feature)", fontsize=12, color=LIGHT, pad=6)

        ax.hlines(0.45, 0.05, 0.95, color=GREY, linewidth=12, alpha=0.5)
        ratio = min(psi / 0.40, 1.0)
        fill_col = SUCCESS if psi < 0.10 else (CAUTION if psi < 0.25 else DANGER)
        ax.hlines(0.45, 0.05, 0.05 + 0.90 * ratio, color=fill_col, linewidth=12)
        nx = 0.05 + 0.90 * ratio
        ax.arrow(nx, 0.88, 0, -0.28, width=0.010, head_width=0.048, head_length=0.07,
                 color=DANGER, length_includes_head=True)

        ax.text(0.05, 0.18, "0.00", ha="left",  color=GREY, fontsize=9)
        ax.text(0.95, 0.18, "0.40+", ha="right", color=GREY, fontsize=9)

        # threshold lines
        for tval, label in [(0.10, "0.10"), (0.25, "0.25")]:
            tx = 0.05 + 0.90 * (tval / 0.40)
            ax.vlines(tx, 0.30, 0.62, color=CAUTION, linewidth=1.5, linestyle="--")
            ax.text(tx, 0.22, label, ha="center", color=CAUTION, fontsize=8)

        band = "no drift" if psi < 0.10 else ("slight drift" if psi < 0.25 else "SIGNIFICANT")
        ax.text(0.50, 0.72, f"PSI={psi:.3f}\n{band}", ha="center", va="center",
                fontsize=12, fontweight="bold", color=fill_col,
                bbox=dict(boxstyle="round,pad=0.35", fc="#0f172a", ec=GREY))

    def draw_history(ax, seg):
        ax.clear()
        ax.set_facecolor(BG)
        ax.tick_params(colors=LIGHT)
        for sp in ax.spines.values():
            sp.set_color(GREY)
        labels = [s["label"] for s in STAGES]
        recalls = [s["recall"] for s in STAGES]
        colors = [GREY] * len(STAGES)
        colors[min(seg + 1, len(STAGES) - 1)] = INFO
        bars = ax.bar(range(len(STAGES)), [r * 100 for r in recalls],
                      color=colors, edgecolor=GREY, alpha=0.85)
        for idx, bar in enumerate(bars):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{recalls[idx]:.1%}", ha="center", va="bottom",
                    fontsize=8, color=LIGHT)
        ax.axhline(80, color=CAUTION, linewidth=1.5, linestyle="--", label="80% target")
        ax.set_xticks(range(len(STAGES)))
        ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=8, color=LIGHT)
        ax.set_ylabel("Recall (%)", fontsize=9, color=LIGHT)
        ax.set_ylim(55, 90)
        ax.set_title("Recall across deployment stages", fontsize=10, color=LIGHT)
        ax.legend(fontsize=8, labelcolor=LIGHT, framealpha=0.1)
        ax.yaxis.label.set_color(LIGHT)

    def animate(frame):
        state, seg = _get_state(min(frame, n_frames - 1))
        draw_gauge(ax_gauge, state["recall"])
        draw_psi(ax_psi,     state["psi"])
        draw_history(ax_hist, seg)
        caption_txt.set_text(f"{state['label']}: {state['caption']}")
        return []

    # Save static PNG from final frame
    animate(n_frames - 1)
    png_path = OUT_DIR / "ch06-production-needle.png"
    fig.savefig(png_path, dpi=120, bbox_inches="tight", facecolor=BG)

    ani = animation.FuncAnimation(fig, animate, frames=n_frames,
                                  interval=280, blit=False, repeat=True)
    gif_path = OUT_DIR / "ch06-production-needle.gif"
    ani.save(gif_path, writer="pillow", fps=4, dpi=100)
    plt.close(fig)
    print(f"wrote {png_path}")
    print(f"wrote {gif_path}")


# ---------------------------------------------------------------------------
# 2. Progress-check PNG — all 5 FraudShield constraints
# ---------------------------------------------------------------------------

CONSTRAINTS = [
    ("1. DETECTION", "82% recall @ 0.5% FPR", "Ch.5 ensemble"),
    ("2. PRECISION", "<0.5% False Positive Rate", "Ch.5 threshold calibration"),
    ("3. EXPLAINABILITY", "SHAP attribution per transaction", "Ch.5 SHAP integration"),
    ("4. LATENCY", "<100ms (30+45+15=90ms)", "Ch.6 latency budget"),
    ("5. DRIFT RESILIENCE", "PSI + weekly recall trigger", "Ch.6 retrain pipeline"),
]


def build_progress_check() -> None:
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 10); ax.set_ylim(0, len(CONSTRAINTS) + 1.2)
    ax.axis("off")

    fig.suptitle("FraudShield — Mission Complete\nAll 5 Production Constraints Satisfied",
                 fontsize=15, fontweight="bold", color=LIGHT, y=0.97)

    row_h = 1.0
    for i, (label, detail, chapter) in enumerate(CONSTRAINTS):
        y = len(CONSTRAINTS) - i  # top to bottom

        # Background bar
        bar = mpatches.FancyBboxPatch(
            (0.2, y - 0.38), 9.6, 0.76,
            boxstyle="round,pad=0.04",
            facecolor=PRIMARY, edgecolor=SUCCESS, linewidth=1.5,
            transform=ax.transData, clip_on=False
        )
        ax.add_patch(bar)

        # Met badge
        ax.text(0.7, y, "MET", ha="center", va="center", fontsize=9,
                fontweight="bold", color=SUCCESS,
                bbox=dict(boxstyle="round,pad=0.25", fc="#052e16", ec=SUCCESS, lw=1.5))

        # Constraint name (bold)
        ax.text(1.2, y + 0.15, label, ha="left", va="center",
                fontsize=11, fontweight="bold", color=LIGHT)

        # Detail
        ax.text(1.2, y - 0.18, detail, ha="left", va="center",
                fontsize=9, color="#93c5fd")

        # Chapter badge
        ax.text(9.7, y, chapter, ha="right", va="center",
                fontsize=8, color=CAUTION,
                bbox=dict(boxstyle="round,pad=0.25", fc="#1e3a8a", ec=CAUTION, lw=1))

    # Bottom caption
    ax.text(5.0, 0.35, "FraudShield | Anomaly Detection Track · Chapter 6 of 6",
            ha="center", va="center", fontsize=9, color=GREY,
            style="italic")

    png_path = OUT_DIR / "ch06-production-progress-check.png"
    fig.savefig(png_path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"wrote {png_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    build_needle_gif()
    build_progress_check()
    print("gen_ch06.py complete.")
