"""Generate ch01 animation GIFs.

Outputs (saved next to this script):
    derivative_to_curve.gif   — zoom-in/zoom-out story: any curve is a
                                stitched-together quilt of tiny straight
                                tangent segments (shown first on a circle,
                                then on an arbitrary f(x)).
    gradient_descent_steps.gif — small-η convergence vs large-η overshoot

Run:  python gen_ch01_animations.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

BLUE = "#2E86C1"
GREEN = "#27AE60"
ORANGE = "#E67E22"
RED = "#C0392B"
DARK = "#2C3E50"
GREY = "#95A5A6"

HERE = Path(__file__).parent


# ──────────────────────────────────────────────────────────────────────────────
# Animation 1: "every curve is a quilt of tiny straight lines"
#
#   Narrative, in four zoom-driven acts (single panel, camera zooms in/out):
#     Act 1 — show a full circle. Pick a point on it.
#     Act 2 — zoom deep into that point until the arc looks like a straight
#             line (the tangent = the derivative at that point).
#     Act 3 — zoom back out while stamping tangent segments at regular angular
#             intervals around the circle. As the camera pulls out, the tiny
#             straight pieces stitch together into the circle.
#     Act 4 — repeat on an arbitrary smooth curve f(x). Zoom into one point
#             (locally straight), then zoom out while the tangent-segment
#             quilt reveals the full curve.
#
#   Visual intuition: like staring at a surface that looks blank, then
#   zooming in and finding microorganisms — each tangent segment has its own
#   slope (its own f'(x)), and the macro shape we call "the curve" is just
#   millions of these micro pieces laid end-to-end.
# ──────────────────────────────────────────────────────────────────────────────
def animate_derivative_to_curve() -> Path:
    # curve for Act 4 — same f(x) we used before (interpretable slope signs)
    f = lambda x: 0.25 * x ** 2 + np.sin(x)
    df = lambda x: 0.5 * x + np.cos(x)

    fig, ax = plt.subplots(figsize=(8.5, 6.4), facecolor="white")
    fig.suptitle(
        "Every curve is a quilt of tiny straight lines:  f(x+dx) ≈ f(x) + f'(x)·dx",
        fontsize=12, fontweight="bold", color=DARK,
    )

    # artists reused across acts
    (curve_line,)   = ax.plot([], [], color=GREY, lw=1.6, alpha=0.45, zorder=1)
    (segments,)     = ax.plot([], [], color=BLUE, lw=2.2, zorder=2)
    (tangent_line,) = ax.plot([], [], color=ORANGE, lw=3.0, zorder=4)
    (cursor_dot,)   = ax.plot([], [], "o", color=RED, ms=8,
                              markeredgecolor="white", markeredgewidth=1.2,
                              zorder=5)
    ax.grid(alpha=0.2)
    ax.set_xlabel("x"); ax.set_ylabel("y")

    caption = ax.text(0.5, 1.02, "", transform=ax.transAxes, ha="center",
                      va="bottom", fontsize=10.5, color=DARK, fontweight="bold")
    detail = ax.text(0.98, 0.03, "", transform=ax.transAxes, ha="right",
                     fontsize=9.5, color=DARK, family="monospace")

    # ── helpers ─────────────────────────────────────────────────────────────
    def set_view(cx, cy, half):
        """Center the camera at (cx, cy) with half-window `half` on each axis."""
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)

    def smoothstep(t):
        t = np.clip(t, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def log_interp(a, b, t):
        """Interpolate in log space — feels natural for camera zoom."""
        return float(np.exp(np.log(a) + (np.log(b) - np.log(a)) * t))

    # ── circle setup (Act 1-3) ──────────────────────────────────────────────
    thetas = np.linspace(0, 2 * np.pi, 400)
    circle_xy = np.column_stack([np.cos(thetas), np.sin(thetas)])
    theta_focus = np.pi / 3           # point of interest on the circle
    cx_focus = np.cos(theta_focus)
    cy_focus = np.sin(theta_focus)
    # unit tangent direction: (-sin θ, cos θ)
    tx = -np.sin(theta_focus); ty = np.cos(theta_focus)

    def circle_tangent_segment(theta_c, half_length):
        c = (np.cos(theta_c), np.sin(theta_c))
        t_dir = (-np.sin(theta_c), np.cos(theta_c))
        x0 = c[0] - half_length * t_dir[0]; y0 = c[1] - half_length * t_dir[1]
        x1 = c[0] + half_length * t_dir[0]; y1 = c[1] + half_length * t_dir[1]
        return (x0, x1), (y0, y1)

    # ── curve setup (Act 4) ─────────────────────────────────────────────────
    x_min, x_max = 0.0, 6.0
    xs_dense = np.linspace(x_min, x_max, 400)
    ys_dense = f(xs_dense)
    x_focus = 3.2
    y_focus = f(x_focus)
    slope_focus = df(x_focus)

    def curve_tangent_segment(xc, half_run):
        m = df(xc); yc = f(xc)
        return (xc - half_run, xc + half_run), (yc - m * half_run, yc + m * half_run)

    # ── frame schedule ──────────────────────────────────────────────────────
    # Each entry: dict with "kind" plus whatever state that frame needs.
    # Acts are kept short (≤ 25 frames each) so the GIF stays light.
    frames = []

    # Act 1 — show the circle, highlight the point. Static camera.
    for _ in range(10):
        frames.append(dict(kind="circle_hold"))

    # Act 2 — zoom IN on the focus point. At maximum zoom the arc is a line.
    ZOOM_FRAMES_IN = 22
    for i in range(ZOOM_FRAMES_IN):
        t = smoothstep(i / (ZOOM_FRAMES_IN - 1))
        frames.append(dict(kind="circle_zoom", half=log_interp(1.6, 0.035, t),
                            show_tangent=t > 0.25, t=t))
    # hold at max zoom
    for _ in range(6):
        frames.append(dict(kind="circle_zoom", half=0.035,
                            show_tangent=True, t=1.0))

    # Act 3 — zoom OUT while stamping tangent segments around the circle.
    N_STAMPS = 48
    ZOOM_FRAMES_OUT = 34
    for i in range(ZOOM_FRAMES_OUT):
        t = smoothstep(i / (ZOOM_FRAMES_OUT - 1))
        n_stamps = int(2 + t * N_STAMPS)
        frames.append(dict(kind="circle_stitch",
                            half=log_interp(0.035, 1.6, t),
                            n_stamps=n_stamps, t=t))
    # hold final view
    for _ in range(6):
        frames.append(dict(kind="circle_stitch", half=1.6,
                            n_stamps=N_STAMPS, t=1.0))

    # Act 4 — arbitrary curve f(x): zoom in on a point, then zoom out while
    # tangent stamps stitch into the full curve.
    for _ in range(6):
        frames.append(dict(kind="curve_hold"))
    ZOOM4_IN = 20
    for i in range(ZOOM4_IN):
        t = smoothstep(i / (ZOOM4_IN - 1))
        frames.append(dict(kind="curve_zoom", half=log_interp(3.6, 0.12, t),
                            show_tangent=t > 0.25, t=t))
    for _ in range(5):
        frames.append(dict(kind="curve_zoom", half=0.12,
                            show_tangent=True, t=1.0))
    N_STAMPS_CURVE = 70
    ZOOM4_OUT = 32
    for i in range(ZOOM4_OUT):
        t = smoothstep(i / (ZOOM4_OUT - 1))
        frames.append(dict(kind="curve_stitch",
                            half=log_interp(0.12, 3.6, t),
                            n_stamps=int(2 + t * N_STAMPS_CURVE), t=t))
    for _ in range(8):
        frames.append(dict(kind="curve_stitch", half=3.6,
                            n_stamps=N_STAMPS_CURVE, t=1.0))

    # ── renderer ────────────────────────────────────────────────────────────
    def build_circle_stamps(n):
        # n tangent segments at evenly spaced angles, each half-length chosen
        # so adjacent segments almost touch (so the "quilt" reads as a circle).
        if n < 2:
            return np.array([]), np.array([])
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        half = np.pi / n   # arc half-width per segment
        xs_all, ys_all = [], []
        for a in angles:
            (x0, x1), (y0, y1) = circle_tangent_segment(a, half)
            xs_all += [x0, x1, np.nan]
            ys_all += [y0, y1, np.nan]
        return np.array(xs_all), np.array(ys_all)

    def build_curve_stamps(n):
        if n < 2:
            return np.array([]), np.array([])
        xs_c = np.linspace(x_min, x_max, n)
        half_run = (x_max - x_min) / (2 * n)
        xs_all, ys_all = [], []
        for xc in xs_c:
            (x0, x1), (y0, y1) = curve_tangent_segment(xc, half_run)
            xs_all += [x0, x1, np.nan]
            ys_all += [y0, y1, np.nan]
        return np.array(xs_all), np.array(ys_all)

    def render(frame):
        kind = frame["kind"]

        # defaults each frame: hide tangent / cursor / segments until set
        tangent_line.set_data([], [])
        cursor_dot.set_data([], [])
        segments.set_data([], [])

        if kind in ("circle_hold", "circle_zoom", "circle_stitch"):
            curve_line.set_data(circle_xy[:, 0], circle_xy[:, 1])
            ax.set_aspect("equal", adjustable="box")

            if kind == "circle_hold":
                set_view(0, 0, 1.6)
                cursor_dot.set_data([cx_focus], [cy_focus])
                caption.set_text("Act 1 — A circle.  Pick any point on it.")
                detail.set_text("")

            elif kind == "circle_zoom":
                half = frame["half"]
                set_view(cx_focus, cy_focus, half)
                cursor_dot.set_data([cx_focus], [cy_focus])
                if frame["show_tangent"]:
                    (xs_t, ys_t) = circle_tangent_segment(theta_focus, 2 * half)
                    tangent_line.set_data(xs_t, ys_t)
                caption.set_text(
                    "Act 2 — Zoom into the point.  The arc is a tiny straight line:"
                    "  that is the derivative."
                )
                detail.set_text(f"zoom half-window = {half:.3f}")

            else:  # circle_stitch
                half = frame["half"]
                n = frame["n_stamps"]
                set_view(cx_focus, cy_focus, half)
                xs_s, ys_s = build_circle_stamps(n)
                segments.set_data(xs_s, ys_s)
                # hide the ghost circle mid-act so the quilt is the star;
                # fade it back in as we reach full zoom-out
                curve_line.set_alpha(0.15 + 0.35 * frame["t"])
                caption.set_text(
                    "Act 3 — Zoom back out.  Tiny tangent segments, stitched "
                    "edge-to-edge, re-form the circle."
                )
                detail.set_text(f"segments placed = {n}")

        else:  # curve acts
            curve_line.set_data(xs_dense, ys_dense)
            ax.set_aspect("auto")

            if kind == "curve_hold":
                set_view((x_min + x_max) / 2, (ys_dense.min() + ys_dense.max()) / 2, 3.6)
                cursor_dot.set_data([x_focus], [y_focus])
                caption.set_text(
                    "Act 4 — Any smooth curve behaves the same way.  Pick a point on f(x)."
                )
                detail.set_text(f"f(x)=0.25 x² + sin x    focus x={x_focus:0.2f}")

            elif kind == "curve_zoom":
                half = frame["half"]
                set_view(x_focus, y_focus, half)
                cursor_dot.set_data([x_focus], [y_focus])
                if frame["show_tangent"]:
                    (xs_t, ys_t) = curve_tangent_segment(x_focus, 2 * half)
                    tangent_line.set_data(xs_t, ys_t)
                caption.set_text(
                    "Zoom in far enough and every smooth curve looks straight. "
                    "That local slope is f'(x)."
                )
                detail.set_text(
                    f"zoom half-window = {half:.3f}    "
                    f"local slope f'(x) = {slope_focus:+0.2f}"
                )

            else:  # curve_stitch
                half = frame["half"]
                n = frame["n_stamps"]
                set_view((x_min + x_max) / 2, (ys_dense.min() + ys_dense.max()) / 2, half)
                xs_s, ys_s = build_curve_stamps(n)
                segments.set_data(xs_s, ys_s)
                curve_line.set_alpha(0.15 + 0.35 * frame["t"])
                caption.set_text(
                    "As we pull the camera back, all those tiny straight pieces "
                    "add up to the curve."
                )
                detail.set_text(f"segments placed = {n}")

    def update(k):
        render(frames[k])
        return []

    anim = FuncAnimation(
        fig, update, frames=len(frames), interval=70,
        blit=False, repeat=True,
    )
    out = HERE / "derivative_to_curve.gif"
    anim.save(out, writer=PillowWriter(fps=14))
    plt.close(fig)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Animation 2: gradient descent — small step vs too-large step
#   J(w) = (w - 3)² + 1.  Starting from w=-1.5.
#     left  : η=0.15  → smooth descent, lands at minimum
#     right : η=1.02  → overshoots, oscillates outward, never reaches it
# ──────────────────────────────────────────────────────────────────────────────
def animate_gradient_descent_steps() -> Path:
    J = lambda w: (w - 3.0) ** 2 + 1.0
    dJ = lambda w: 2.0 * (w - 3.0)

    w0 = -1.5
    n_iter = 30

    def run(eta: float):
        ws = [w0]
        for _ in range(n_iter):
            w = ws[-1]
            ws.append(w - eta * dJ(w))
        return np.array(ws)

    ws_small = run(0.15)
    ws_large = run(1.02)

    ww = np.linspace(-6, 10, 400)
    jj = J(ww)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), facecolor="white")
    fig.suptitle(
        "Gradient descent:  w ← w − η·J'(w).  Small steps converge; too-large steps overshoot.",
        fontsize=12, fontweight="bold", color=DARK,
    )

    panels = []
    for ax, ws, eta, title, colour in [
        (axes[0], ws_small, 0.15, "η = 0.15  (small step → converges)", GREEN),
        (axes[1], ws_large, 1.02, "η = 1.02  (too large → overshoots)", RED),
    ]:
        ax.plot(ww, jj, color=GREY, lw=1.8, alpha=0.8)
        ax.axvline(3.0, color=DARK, lw=0.8, ls=":", alpha=0.6)
        ax.text(3.0, J(3.0) - 4, "minimum\nw*=3", ha="center", fontsize=9,
                color=DARK)
        path_line, = ax.plot([], [], "-", color=colour, lw=1.6, alpha=0.7)
        path_pts, = ax.plot([], [], "o", color=colour, ms=5, alpha=0.55)
        ball, = ax.plot([], [], "o", color=colour, ms=13,
                        markeredgecolor="white", markeredgewidth=1.4, zorder=5)
        info = ax.text(0.03, 0.95, "", transform=ax.transAxes, va="top",
                       fontsize=9.5, family="monospace", color=DARK)
        ax.set_title(title, fontsize=11, color=DARK)
        ax.set_xlabel("w"); ax.set_ylabel("J(w)")
        ax.set_xlim(-6, 10)
        ax.set_ylim(-2, min(80, jj.max() + 5))
        ax.grid(alpha=0.25)
        panels.append((ws, eta, path_line, path_pts, ball, info))

    def update(k: int):
        arts = []
        for ws, eta, path_line, path_pts, ball, info in panels:
            k_c = min(k, len(ws) - 1)
            xs_p = ws[: k_c + 1]
            ys_p = J(xs_p)
            path_line.set_data(xs_p, ys_p)
            path_pts.set_data(xs_p, ys_p)
            ball.set_data([ws[k_c]], [J(ws[k_c])])
            info.set_text(
                f"iter {k_c:>2}\nη    = {eta:0.2f}\n"
                f"w    = {ws[k_c]:+0.3f}\nJ(w) = {J(ws[k_c]):0.3f}"
            )
            arts += [path_line, path_pts, ball, info]
        return arts

    anim = FuncAnimation(
        fig, update, frames=n_iter + 6, interval=220, blit=False, repeat=True,
    )
    out = HERE / "gradient_descent_steps.gif"
    anim.save(out, writer=PillowWriter(fps=6))
    plt.close(fig)
    return out


if __name__ == "__main__":
    for p in (animate_derivative_to_curve(), animate_gradient_descent_steps()):
        print(f"wrote {p.relative_to(HERE.parent.parent.parent.parent)}")
