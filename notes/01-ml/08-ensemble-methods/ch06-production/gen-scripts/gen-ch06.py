"""
gen_ch06.py — Generate visual assets for Ch.6 Production Ensembles.

Outputs (written to ../img/ relative to this script):
  ch06-production-needle.gif     — EnsembleAI production arc: offline accuracy ->
                                   latency validated -> drift handled -> all 5 constraints met
  ch06-production-progress-check.png — All 5 EnsembleAI constraints satisfied dashboard

Usage:
    python gen_ch06.py
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
IMG_DIR = SCRIPT_DIR.parent / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# shared animation renderer (same as all other chapters)
ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.animation_renderer import render_metric_story  # noqa: E402

# ── constants ───────────────────────────────────────────────────────────────
SEED = 42
BG      = "#1a1a2e"
PRIMARY = "#1e3a8a"
SUCCESS = "#15803d"
CAUTION = "#b45309"
DANGER  = "#b91c1c"
INFO    = "#1d4ed8"
LIGHT   = "#f1f5f9"
GREY    = "#64748b"

rng = np.random.default_rng(SEED)


# ═══════════════════════════════════════════════════════════════════════════
# 1 · NEEDLE GIF  — uses standard render_metric_story (regression style)
#     MAE in $k — lower is better
# ═══════════════════════════════════════════════════════════════════════════

STAGES = [
    {
        "label":   "Single XGBoost (34k)",
        "value":   34.0,
        "curve":   [0.12, -0.08, 0.95, 0.18],
        "margin":  0.62,
        "caption": "Single XGBoost baseline: MAE $34k. Ensembles beat this, but can we deploy in <50ms?",
    },
    {
        "label":   "Stack offline (20k)",
        "value":   20.0,
        "curve":   [0.06, -0.04, 0.88, 0.08],
        "margin":  0.38,
        "caption": "Stacked ensemble: MAE $20k — 41% better than single XGBoost. Now validate latency.",
    },
    {
        "label":   "Parallel serving (35ms)",
        "value":   20.4,
        "curve":   [0.05, -0.03, 0.87, 0.06],
        "margin":  0.32,
        "caption": "Parallel inference: max(1,15,20)+5+10 = 35ms < 50ms SLA. Constraint #3 LATENCY satisfied.",
    },
    {
        "label":   "Post-retrain (20.2k)",
        "value":   20.2,
        "curve":   [0.04, -0.02, 0.86, 0.04],
        "margin":  0.24,
        "caption": "PSI drift detected month 4. Retrain + shadow mode + cutover. All 5 constraints satisfied.",
    },
]

if __name__ == "__main__":
    # Needle GIF — MAE lower is better
    render_metric_story(
        SCRIPT_DIR,
        "ch06-production-needle",
        "Ch.6 — Production ensemble: MAE $34k -> $20k, latency 35ms < 50ms SLA",
        "Production MAE ($k)",
        STAGES,
        better="lower",
        style="regression",
    )

    # ─────────────────────────────────────────────────────────────────────
    # 2 · PROGRESS CHECK PNG
    # ─────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), facecolor=BG)
    fig.suptitle("EnsembleAI — All 5 Constraints Satisfied",
                 fontsize=16, fontweight="bold", color=LIGHT, y=0.97)

    constraint_data = [
        {
            "num": "#1", "name": "IMPROVEMENT",
            "detail": "MAE $20,000\nvs XGBoost $34,000\n\n-41% better",
        },
        {
            "num": "#2", "name": "DIVERSITY",
            "detail": "LR + RF + XGBoost\nError correlation\n< 0.4 pairwise",
        },
        {
            "num": "#3", "name": "LATENCY",
            "detail": "P50 = 33ms\nP99 = 38ms\nSLA = 50ms",
        },
        {
            "num": "#4", "name": "EXPLAINABILITY",
            "detail": "TreeSHAP per call\nTop-3 features\nin dollars",
        },
        {
            "num": "#5", "name": "MONITORING",
            "detail": "PSI > 0.20 triggers\nretraining pipeline\nMonth 4",
        },
    ]

    flat = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]
    ax_timeline = axes[1, 2]

    for ax, cd in zip(flat, constraint_data):
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_color(SUCCESS)
            sp.set_linewidth(2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])

        ax.text(0.12, 0.88, cd["num"], ha="center", va="center",
                fontsize=20, fontweight="bold", color=BG,
                bbox=dict(boxstyle="circle,pad=0.3", fc=SUCCESS, ec=SUCCESS))

        ax.text(0.5, 0.72, cd["name"], ha="center", va="center",
                fontsize=13, fontweight="bold", color=SUCCESS)

        ax.text(0.5, 0.40, cd["detail"], ha="center", va="center",
                fontsize=10, color=LIGHT, linespacing=1.6)

        ax.text(0.5, 0.08, "SATISFIED", ha="center", va="center",
                fontsize=11, fontweight="bold", color=BG,
                bbox=dict(boxstyle="round,pad=0.35", fc=SUCCESS, ec=SUCCESS))

    # Timeline panel
    ax_timeline.set_facecolor(BG)
    for sp in ax_timeline.spines.values():
        sp.set_color(GREY)
        sp.set_alpha(0.5)

    months = [0, 1, 2, 3, 4, 5, 6]
    maes   = [21.4, 21.1, 21.8, 24.8, 24.2, 23.8, 20.2]
    events = {
        0: ("Go-live", LIGHT),
        2: ("Full rollout", INFO),
        3: ("PSI alert", CAUTION),
        4: ("Retrain", DANGER),
        5: ("Shadow mode", CAUTION),
        6: ("Cutover", SUCCESS),
    }

    ax_timeline.plot(months, maes, color=PRIMARY, linewidth=2.5, marker="o",
                     markersize=7, markerfacecolor=LIGHT, zorder=3)
    ax_timeline.axhline(20, color=SUCCESS, linestyle="--", linewidth=1.5,
                        alpha=0.7, label="Target MAE $20k")
    ax_timeline.fill_between(months, maes, 20, alpha=0.15, color=CAUTION)

    for m, (label, col) in events.items():
        offset = 0.8 + (0.3 if m % 2 else 0)
        ax_timeline.annotate(label,
                             xy=(m, maes[m]),
                             xytext=(m, maes[m] + offset),
                             fontsize=7.5, color=col, ha="center",
                             arrowprops=dict(arrowstyle="-", color=col, lw=1))

    ax_timeline.set_title("6-Month Production MAE Timeline", color=LIGHT, fontsize=10, pad=6)
    ax_timeline.set_xlabel("Month", color=LIGHT, fontsize=9)
    ax_timeline.set_ylabel("MAE ($k)", color=LIGHT, fontsize=9)
    ax_timeline.set_ylim(18, 27)
    ax_timeline.tick_params(colors=LIGHT, labelsize=8)
    ax_timeline.legend(fontsize=8, labelcolor=LIGHT, framealpha=0.25)
    ax_timeline.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:.0f}k"))

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out = IMG_DIR / "ch06-production-progress-check.png"
    fig.savefig(str(out), dpi=130, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out}")

    print("Done.")


Outputs (written to ../img/ relative to this script):
  ch06-production-needle.gif     — EnsembleAI production arc: offline accuracy →
                                   latency validated → drift handled → all 5 constraints met
  ch06-production-progress-check.png — All 5 EnsembleAI constraints satisfied dashboard

Usage:
    python gen_ch06.py
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
IMG_DIR = SCRIPT_DIR.parent / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# Try to use shared animation renderer (falls back to local impl if unavailable)
ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

# ── constants ───────────────────────────────────────────────────────────────
SEED = 42
BG = "#1a1a2e"       # dark background (track convention)
PRIMARY = "#1e3a8a"  # deep blue
SUCCESS = "#15803d"  # green
CAUTION = "#b45309"  # amber
DANGER  = "#b91c1c"  # red
INFO    = "#1d4ed8"  # medium blue
LIGHT   = "#f1f5f9"
GREY    = "#64748b"

rng = np.random.default_rng(SEED)


# ═══════════════════════════════════════════════════════════════════════════
# 1 · NEEDLE GIF
# ═══════════════════════════════════════════════════════════════════════════

STAGES = [
    {
        "title":   "Ch.5 Outcome: Offline MAE = $20k",
        "subtitle":"Stacked ensemble beats single XGBoost by 41%",
        "mae":     20_000,
        "latency": None,
        "psi":     None,
        "constraints": [True, True, False, False, False],
        "note":    "Excellent offline accuracy — now can we serve it in <50ms?",
    },
    {
        "title":   "Parallel Serving: Latency = 35ms",
        "subtitle":"max(LR 1ms, RF 15ms, XGB 20ms) + meta 5ms + overhead 10ms = 35ms",
        "mae":     21_400,
        "latency": 35,
        "psi":     0.05,
        "constraints": [True, True, True, False, False],
        "note":    "Constraint #3 LATENCY ✅  35ms < 50ms SLA",
    },
    {
        "title":   "Month 3: Drift Detected (PSI = 0.17)",
        "subtitle":"California housing market +15% — PSI enters monitor zone",
        "mae":     24_800,
        "latency": 35,
        "psi":     0.17,
        "constraints": [True, True, True, True, False],
        "note":    "SHAP API live ✅  PSI alert fires → retraining pipeline starts",
    },
    {
        "title":   "Month 6: All 5 Constraints Satisfied",
        "subtitle":"Retrained model restores MAE $20k. EnsembleAI COMPLETE",
        "mae":     20_200,
        "latency": 35,
        "psi":     0.12,
        "constraints": [True, True, True, True, True],
        "note":    "Production MAE $20k | P99 38ms | SHAP per call | PSI monitor live",
    },
]

CONSTRAINT_LABELS = [
    "#1 IMPROVEMENT\nMAE $20k (-41%)",
    "#2 DIVERSITY\nLR+RF+XGB",
    "#3 LATENCY\n35ms < 50ms",
    "#4 EXPLAINABILITY\nSHAP API",
    "#5 MONITORING\nPSI + retrain",
]

FRAMES_PER_STAGE = 30
HOLD_FRAMES = 20
TOTAL_FRAMES = len(STAGES) * FRAMES_PER_STAGE + HOLD_FRAMES


def _ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 0.5 - 0.5 * np.cos(np.pi * t)


def _lerp(a, b, t):
    return a + (b - a) * t


def _get_stage_and_progress(frame):
    """Return (stage_idx, next_stage_idx, progress 0..1)."""
    if frame >= len(STAGES) * FRAMES_PER_STAGE:
        # hold on last stage
        return len(STAGES) - 1, len(STAGES) - 1, 1.0
    seg = frame // FRAMES_PER_STAGE
    local = (frame % FRAMES_PER_STAGE) / max(FRAMES_PER_STAGE - 1, 1)
    local = _ease(local)
    seg_next = min(seg + 1, len(STAGES) - 1)
    return seg, seg_next, local


def _draw_needle(ax, value_now, label, min_val, max_val, title):
    """Draw a gauge needle for a metric. lower is better for MAE."""
    ax.clear()
    ax.set_facecolor(BG)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    # Track bar
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1)
    ax.hlines(0.45, 0.05, 0.95, color=GREY, linewidth=12, alpha=0.5)

    # MAE: lower = better → needle moves right as value decreases
    ratio = (max_val - value_now) / (max_val - min_val)
    ratio = max(0.0, min(1.0, float(ratio)))

    color = SUCCESS if ratio > 0.6 else (CAUTION if ratio > 0.35 else DANGER)
    ax.hlines(0.45, 0.05, 0.05 + 0.9 * ratio, color=color, linewidth=12)

    needle_x = 0.05 + 0.9 * ratio
    ax.annotate("", xy=(needle_x, 0.17), xytext=(needle_x, 0.82),
                arrowprops=dict(arrowstyle="-|>", color=DANGER, lw=2.5))

    ax.text(0.05, 0.10, "worse", ha="left", va="center", fontsize=9, color=GREY)
    ax.text(0.95, 0.10, "better", ha="right", va="center", fontsize=9, color=LIGHT)

    ax.text(0.5, 0.78, f"{label}\n${value_now:,.0f}", ha="center", va="center",
            fontsize=13, fontweight="bold", color=LIGHT,
            bbox=dict(boxstyle="round,pad=0.4", fc=PRIMARY, ec=LIGHT, alpha=0.9))

    ax.set_title(title, fontsize=11, fontweight="bold", color=LIGHT, pad=8)


def _draw_latency_bar(ax, latency, sla=50):
    ax.clear()
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_color(GREY)
        sp.set_alpha(0.4)
    ax.set_facecolor(BG)

    if latency is None:
        ax.text(0.5, 0.5, "Latency\nnot yet\nmeasured",
                ha="center", va="center", fontsize=10, color=GREY,
                transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        return

    bar_color = SUCCESS if latency < sla * 0.8 else (CAUTION if latency < sla else DANGER)
    ax.barh(["Latency"], [latency], color=bar_color, alpha=0.9)
    ax.axvline(sla, color=DANGER, linestyle="--", linewidth=2, label=f"SLA {sla}ms")
    ax.set_xlim(0, 70)
    ax.set_xlabel("ms", color=LIGHT, fontsize=9)
    ax.tick_params(colors=LIGHT, labelsize=9)
    ax.set_title("P50 Latency", fontsize=10, color=LIGHT)
    ax.legend(fontsize=8, labelcolor=LIGHT, framealpha=0.3)
    ax.text(latency + 1, 0, f"{latency}ms", va="center", fontsize=9, color=LIGHT, fontweight="bold")


def _draw_constraints(ax, satisfied):
    ax.clear()
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(CONSTRAINT_LABELS) - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.set_title("EnsembleAI Constraints", fontsize=10, color=LIGHT, pad=6)

    for i, (label, done) in enumerate(zip(reversed(CONSTRAINT_LABELS), reversed(satisfied))):
        y = i
        color = SUCCESS if done else GREY
        symbol = "[x]" if done else "[ ]"
        ax.text(0.08, y, symbol, va="center", fontsize=10, fontfamily="monospace")
        ax.text(0.22, y, label, va="center", fontsize=8, color=color)


def _draw_note(ax, note):
    ax.clear()
    ax.set_facecolor(BG)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(0.5, 0.5, note, ha="center", va="center", fontsize=9,
            color=LIGHT, wrap=True,
            bbox=dict(boxstyle="round,pad=0.6", fc=INFO, ec=LIGHT, alpha=0.25),
            transform=ax.transAxes)


def make_needle_gif():
    fig = plt.figure(figsize=(11, 6), facecolor=BG)

    # Layout: top row (title), then [gauge | latency | constraints], then note
    gs = fig.add_gridspec(3, 3,
                          height_ratios=[0.12, 0.65, 0.23],
                          hspace=0.35, wspace=0.3,
                          left=0.04, right=0.98, top=0.95, bottom=0.04)

    ax_title  = fig.add_subplot(gs[0, :])
    ax_gauge  = fig.add_subplot(gs[1, 0])
    ax_lat    = fig.add_subplot(gs[1, 1])
    ax_con    = fig.add_subplot(gs[1, 2])
    ax_note   = fig.add_subplot(gs[2, :])

    ax_title.set_facecolor(BG)
    ax_title.set_xticks([])
    ax_title.set_yticks([])
    for sp in ax_title.spines.values():
        sp.set_visible(False)

    min_mae = 18_000
    max_mae = 35_000

    def update(frame):
        seg, seg_next, progress = _get_stage_and_progress(frame)
        s0 = STAGES[seg]
        s1 = STAGES[seg_next]

        # Interpolate MAE
        mae = _lerp(s0["mae"], s1["mae"], progress)

        # Title
        ax_title.clear()
        ax_title.set_facecolor(BG)
        ax_title.set_xticks([])
        ax_title.set_yticks([])
        for sp in ax_title.spines.values():
            sp.set_visible(False)
        stage_disp = s1 if progress > 0.5 else s0
        ax_title.text(0.5, 0.65, stage_disp["title"],
                      ha="center", va="center", fontsize=12,
                      fontweight="bold", color=LIGHT, transform=ax_title.transAxes)
        ax_title.text(0.5, 0.15, stage_disp["subtitle"],
                      ha="center", va="center", fontsize=9,
                      color=GREY, transform=ax_title.transAxes)

        _draw_needle(ax_gauge, mae, "Production MAE", min_mae, max_mae,
                     "MAE (lower = better)")

        lat_disp = s1["latency"] if progress > 0.5 else s0["latency"]
        _draw_latency_bar(ax_lat, lat_disp)

        # Constraints: light up progressively
        con_disp = s1["constraints"] if progress > 0.5 else s0["constraints"]
        _draw_constraints(ax_con, con_disp)

        _draw_note(ax_note, stage_disp["note"])

        return []

    ani = animation.FuncAnimation(fig, update, frames=TOTAL_FRAMES,
                                  interval=60, blit=False)

    out = IMG_DIR / "ch06-production-needle.gif"
    ani.save(str(out), writer="pillow", fps=16, dpi=110)
    plt.close(fig)
    print(f"  saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════
# 2 · PROGRESS CHECK PNG
# ═══════════════════════════════════════════════════════════════════════════

def make_progress_check():
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), facecolor=BG)
    fig.suptitle("EnsembleAI — All 5 Constraints SATISFIED",
                 fontsize=16, fontweight="bold", color=LIGHT, y=0.97)

    # ── top row: 5 constraint panels ──────────────────────────────────────
    constraint_data = [
        {
            "num": "#1", "name": "IMPROVEMENT",
            "detail": "MAE $20,000\nvs XGBoost $34,000\n\n−41% ✅",
            "value": 41, "unit": "% improvement",
        },
        {
            "num": "#2", "name": "DIVERSITY",
            "detail": "LR + RF + XGBoost\nError correlation\n< 0.4 pairwise ✅",
            "value": 3, "unit": "model families",
        },
        {
            "num": "#3", "name": "LATENCY",
            "detail": "P50 = 33ms\nP99 = 38ms\nSLA = 50ms ✅",
            "value": 35, "unit": "ms (vs 50ms SLA)",
        },
        {
            "num": "#4", "name": "EXPLAINABILITY",
            "detail": "TreeSHAP per call\nTop-3 features\nin dollars ✅",
            "value": 3, "unit": "ms SHAP latency",
        },
        {
            "num": "#5", "name": "MONITORING",
            "detail": "PSI > 0.20 triggers\nretraining pipeline\nMonth 4 ✅",
            "value": 0.20, "unit": "PSI threshold",
        },
    ]

    # Flatten axes
    flat = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]
    ax_timeline = axes[1, 2]

    for ax, cd in zip(flat, constraint_data):
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_color(SUCCESS)
            sp.set_linewidth(2)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])

        # Constraint number badge
        ax.text(0.12, 0.88, cd["num"], ha="center", va="center",
                fontsize=20, fontweight="bold", color=BG,
                bbox=dict(boxstyle="circle,pad=0.3", fc=SUCCESS, ec=SUCCESS))

        ax.text(0.5, 0.72, cd["name"], ha="center", va="center",
                fontsize=13, fontweight="bold", color=SUCCESS)

        ax.text(0.5, 0.40, cd["detail"], ha="center", va="center",
                fontsize=10, color=LIGHT, linespacing=1.6)

        ax.text(0.5, 0.08, "SATISFIED", ha="center", va="center",
                fontsize=11, fontweight="bold", color=BG,
                bbox=dict(boxstyle="round,pad=0.35", fc=SUCCESS, ec=SUCCESS))

    # ── timeline panel (bottom right) ─────────────────────────────────────
    ax_timeline.set_facecolor(BG)
    for sp in ax_timeline.spines.values():
        sp.set_color(GREY)
        sp.set_alpha(0.5)

    months = [0, 1, 2, 3, 4, 5, 6]
    maes   = [21.4, 21.1, 21.8, 24.8, 24.2, 23.8, 20.2]
    events = {
        0: ("Go-live", LIGHT),
        2: ("Full rollout", INFO),
        3: ("PSI alert", CAUTION),
        4: ("Retrain", DANGER),
        5: ("Shadow mode", CAUTION),
        6: ("Cutover ✅", SUCCESS),
    }

    ax_timeline.plot(months, maes, color=PRIMARY, linewidth=2.5, marker="o",
                     markersize=7, markerfacecolor=LIGHT, zorder=3)
    ax_timeline.axhline(20, color=SUCCESS, linestyle="--", linewidth=1.5,
                        alpha=0.7, label="Target MAE $20k")
    ax_timeline.fill_between(months, maes, 20, alpha=0.15, color=CAUTION)

    for m, (label, col) in events.items():
        ax_timeline.annotate(label,
                             xy=(m, maes[m]),
                             xytext=(m, maes[m] + 0.8 + (0.3 if m % 2 else 0)),
                             fontsize=7.5, color=col, ha="center",
                             arrowprops=dict(arrowstyle="-", color=col, lw=1))

    ax_timeline.set_title("6-Month Production MAE Timeline", color=LIGHT, fontsize=10, pad=6)
    ax_timeline.set_xlabel("Month", color=LIGHT, fontsize=9)
    ax_timeline.set_ylabel("MAE ($k)", color=LIGHT, fontsize=9)
    ax_timeline.set_ylim(18, 27)
    ax_timeline.tick_params(colors=LIGHT, labelsize=8)
    ax_timeline.legend(fontsize=8, labelcolor=LIGHT, framealpha=0.25)
    ax_timeline.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:.0f}k"))

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out = IMG_DIR / "ch06-production-progress-check.png"
    fig.savefig(str(out), dpi=130, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import matplotlib.ticker  # needed inside make_progress_check
    print("Generating Ch.6 Production Ensembles assets...")
    make_needle_gif()
    make_progress_check()
    print("Done.")
