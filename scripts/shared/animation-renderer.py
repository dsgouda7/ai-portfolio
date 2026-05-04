import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

DARK = "#1f2937"
BLUE = "#2563eb"
LIGHT_BLUE = "#93c5fd"
GREEN = "#16a34a"
AMBER = "#d97706"
RED = "#dc2626"
PURPLE = "#7c3aed"
GREY = "#cbd5e1"
LIGHT = "#f8fafc"


_rng = np.random.default_rng(42)
REG_X = np.linspace(-3.2, 3.2, 220)
REG_Y_TRUE = 0.42 * REG_X**2 + 0.15 * REG_X + 0.9
REG_Y_NOISY = REG_Y_TRUE + _rng.normal(0, 0.35, size=REG_X.shape)

CLS_NEG = _rng.normal(loc=[-1.4, -0.6], scale=[0.65, 0.7], size=(110, 2))
CLS_POS = _rng.normal(loc=[1.2, 0.8], scale=[0.55, 0.65], size=(90, 2))
CLS_RARE = _rng.normal(loc=[0.5, 1.6], scale=[0.22, 0.25], size=(18, 2))

CLUSTER_CENTERS = np.array([
    [-2.6, -1.8],
    [-1.4, 1.7],
    [0.4, -0.4],
    [2.2, 1.9],
    [2.8, -1.4],
])
CLUSTER_POINTS = np.vstack([
    _rng.normal(loc=center, scale=[0.35, 0.35], size=(42, 2))
    for center in CLUSTER_CENTERS
])

CLUSTER_COLORS = [BLUE, GREEN, AMBER, PURPLE, RED]


def _ease(t: float) -> float:
    t = max(0.0, min(1.0, float(t)))
    return 0.5 - 0.5 * math.cos(math.pi * t)


def _lerp(a, b, t):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a + (b - a) * t
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    return arr_a + (arr_b - arr_a) * t


def _get_frame_state(stages, frame, frames_per_transition):
    if len(stages) == 1:
        return stages[0], 0, 1.0
    seg = min(frame // frames_per_transition, len(stages) - 2)
    local = (frame % frames_per_transition) / max(frames_per_transition - 1, 1)
    local = _ease(local)
    s0, s1 = stages[seg], stages[seg + 1]

    state = {}
    for key in set(s0) | set(s1):
        if key not in s0:
            state[key] = s1[key]
        elif key not in s1:
            state[key] = s0[key]
        else:
            v0, v1 = s0[key], s1[key]
            if isinstance(v0, (int, float, list, tuple, np.ndarray)) and isinstance(v1, (int, float, list, tuple, np.ndarray)):
                try:
                    state[key] = _lerp(v0, v1, local)
                except Exception:
                    state[key] = v0 if local < 0.5 else v1
            else:
                state[key] = v0 if local < 0.5 else v1
    return state, seg, local


def _draw_gauge(ax, metric_name, value, min_value, max_value, better):
    ax.clear()
    ax.set_facecolor("white")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("Needle moved", fontsize=13, fontweight="bold", color=DARK, pad=10)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1)
    ax.hlines(0.45, 0.05, 0.95, color=GREY, linewidth=10, alpha=0.8)

    if max_value == min_value:
        ratio = 0.5
    else:
        if better == "lower":
            ratio = (max_value - value) / (max_value - min_value)
        else:
            ratio = (value - min_value) / (max_value - min_value)
    ratio = max(0.0, min(1.0, float(ratio)))

    ax.hlines(0.45, 0.05, 0.05 + 0.9 * ratio, color=GREEN, linewidth=10)
    needle_x = 0.05 + 0.9 * ratio
    ax.arrow(needle_x, 0.88, 0, -0.26, width=0.01, head_width=0.05, head_length=0.08,
             color=RED, length_includes_head=True)
    ax.text(0.05, 0.12, "worse", ha="left", va="center", fontsize=10, color=DARK)
    ax.text(0.95, 0.12, "better", ha="right", va="center", fontsize=10, color=DARK)
    ax.text(0.5, 0.72, f"{metric_name}\n{value:,.2f}", ha="center", va="center",
            fontsize=14, fontweight="bold", color=BLUE,
            bbox=dict(boxstyle="round,pad=0.35", fc=LIGHT, ec=GREY))


def _draw_history(ax, stages, metric_name, better, active_index):
    ax.clear()
    labels = [stage["label"] for stage in stages]
    values = [float(stage["value"]) for stage in stages]
    colors = [LIGHT_BLUE] * len(stages)
    colors[min(active_index + 1, len(stages) - 1)] = BLUE

    bars = ax.bar(range(len(stages)), values, color=colors, edgecolor=DARK, alpha=0.9)
    for idx, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{values[idx]:.2f}",
                ha="center", va="bottom", fontsize=8, color=DARK)
    ax.set_xticks(range(len(stages)), labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(metric_name, fontsize=9)
    trend = "↓ lower is better" if better == "lower" else "↑ higher is better"
    ax.set_title(f"Concept-by-concept progress ({trend})", fontsize=11, color=DARK)
    ax.grid(axis="y", alpha=0.2)


def _draw_regression(ax, state, title):
    ax.clear()
    curve = np.asarray(state.get("curve", [0.0, 0.0, 1.0, 0.0]), dtype=float)
    if curve.size == 3:
        a, b, c = curve
        wiggle = 0.0
    else:
        a, b, c, wiggle = curve[:4]

    y_pred = a * REG_X**2 + b * REG_X + c + wiggle * np.sin(2.4 * REG_X)
    ax.scatter(REG_X, REG_Y_NOISY, s=12, alpha=0.4, color=LIGHT_BLUE, label="housing samples")
    ax.plot(REG_X, REG_Y_TRUE, color=GREY, linestyle="--", linewidth=2, label="hidden signal")
    ax.plot(REG_X, y_pred, color=BLUE, linewidth=3, label=state.get("label", "model fit"))
    ax.fill_between(REG_X, y_pred, REG_Y_TRUE, color=AMBER, alpha=0.15)
    ax.set_title(title, fontsize=14, fontweight="bold", color=DARK)
    ax.set_xlabel("Feature space")
    ax.set_ylabel("Target")
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    ax.grid(alpha=0.2)


def _draw_classification(ax, state, title):
    ax.clear()
    m = float(state.get("m", 0.4))
    b = float(state.get("b", 0.0))
    bend = float(state.get("bend", 0.0))
    margin = float(state.get("margin", 0.55))

    ax.scatter(CLS_NEG[:, 0], CLS_NEG[:, 1], s=18, alpha=0.65, color=BLUE, label="negative")
    ax.scatter(CLS_POS[:, 0], CLS_POS[:, 1], s=18, alpha=0.7, color=AMBER, label="positive")
    ax.scatter(CLS_RARE[:, 0], CLS_RARE[:, 1], s=28, alpha=0.9, color=RED, label="rare positives")

    xs = np.linspace(-3.2, 3.2, 250)
    boundary = m * xs + b + bend * (xs**2 - 1.0)
    ax.plot(xs, boundary, color=GREEN, linewidth=3, label=state.get("label", "decision boundary"))
    ax.plot(xs, boundary + margin, color=GREEN, linewidth=1.2, linestyle="--", alpha=0.7)
    ax.plot(xs, boundary - margin, color=GREEN, linewidth=1.2, linestyle="--", alpha=0.7)

    ax.set_xlim(-3.4, 3.4)
    ax.set_ylim(-3.0, 3.4)
    ax.set_title(title, fontsize=14, fontweight="bold", color=DARK)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    ax.grid(alpha=0.2)


def _draw_threshold(ax, state, title):
    ax.clear()
    threshold = float(state.get("threshold", 0.5))
    x = np.linspace(0, 1, 300)
    neg = np.exp(-0.5 * ((x - 0.28) / 0.10) ** 2)
    pos = 0.9 * np.exp(-0.5 * ((x - 0.68) / 0.12) ** 2) + 0.35 * np.exp(-0.5 * ((x - 0.42) / 0.07) ** 2)

    ax.fill_between(x, neg, color=BLUE, alpha=0.45, label="negative scores")
    ax.fill_between(x, pos, color=AMBER, alpha=0.45, label="positive scores")
    ax.axvline(threshold, color=RED, linewidth=3, label=f"threshold={threshold:.2f}")
    ax.text(threshold, max(pos.max(), neg.max()) * 1.02, state.get("threshold_note", ""),
            ha="center", va="bottom", fontsize=9, color=RED)
    ax.set_title(title, fontsize=14, fontweight="bold", color=DARK)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Density")
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    ax.grid(alpha=0.2)


def _draw_clustering(ax, state, title):
    ax.clear()
    centers = np.asarray(state.get("centers", CLUSTER_CENTERS), dtype=float)
    pts = CLUSTER_POINTS

    dists = np.sqrt(((pts[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
    labels = dists.argmin(axis=1)
    for idx in range(len(centers)):
        cluster_pts = pts[labels == idx]
        if len(cluster_pts):
            ax.scatter(cluster_pts[:, 0], cluster_pts[:, 1], s=18, alpha=0.7,
                       color=CLUSTER_COLORS[idx % len(CLUSTER_COLORS)])
    ax.scatter(centers[:, 0], centers[:, 1], s=220, marker="X", color=DARK, label="centroids")

    for idx, center in enumerate(centers):
        ax.text(center[0] + 0.07, center[1] + 0.07, f"C{idx+1}", fontsize=9, color=DARK)

    ax.set_xlim(-3.5, 3.8)
    ax.set_ylim(-3.0, 3.2)
    ax.set_title(title, fontsize=14, fontweight="bold", color=DARK)
    ax.set_xlabel("Embedding 1")
    ax.set_ylabel("Embedding 2")
    ax.grid(alpha=0.2)


def render_metric_story(out_dir, slug, title, metric_name, stages, better="lower", style="regression"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    values = [float(stage["value"]) for stage in stages]
    min_value = min(values)
    max_value = max(values)

    frames_per_transition = 10
    n_frames = max(20, (len(stages) - 1) * frames_per_transition + 8)

    fig = plt.figure(figsize=(12, 7), facecolor="white")
    grid = fig.add_gridspec(2, 2, height_ratios=[2.2, 1.2], width_ratios=[2.2, 1.0])
    ax_main = fig.add_subplot(grid[0, 0])
    ax_gauge = fig.add_subplot(grid[0, 1])
    ax_hist = fig.add_subplot(grid[1, :])

    # Create caption text without bbox to avoid transform issues
    caption_box = fig.text(
        0.5, 0.02, "", ha="center", va="center", fontsize=11, color=DARK
    )

    draw_map = {
        "regression": _draw_regression,
        "classification": _draw_classification,
        "threshold": _draw_threshold,
        "clustering": _draw_clustering,
    }
    draw_func = draw_map[style]

    def animate(frame):
        state, seg, _ = _get_frame_state(stages, frame, frames_per_transition)
        draw_func(ax_main, state, title)
        _draw_gauge(ax_gauge, metric_name, float(state["value"]), min_value, max_value, better)
        _draw_history(ax_hist, stages, metric_name, better, seg)
        caption_box.set_text(f"{state.get('label', '')}: {state.get('caption', '')}")
        return []

    animate(n_frames - 1)
    png_path = out_dir / f"{slug}.png"
    gif_path = out_dir / f"{slug}.gif"
    fig.savefig(png_path, dpi=120, bbox_inches="tight")

    ani = animation.FuncAnimation(fig, animate, frames=n_frames, interval=260, blit=False, repeat=True)
    ani.save(gif_path, writer="pillow", fps=4, dpi=100)
    plt.close(fig)
    print(f"wrote {png_path}")
    print(f"wrote {gif_path}")
