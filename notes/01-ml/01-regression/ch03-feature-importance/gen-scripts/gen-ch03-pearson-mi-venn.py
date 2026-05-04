"""
Generates ch03-pearson-mi-venn.gif

Animated Venn diagram showing:
  Phase 1 — Two circles drift together (X and Y gaining shared information).
             A shared-area label "Shared Information" fades in.
  Phase 2 — Caption split: left circle labelled "Pearson sees:"
             with a narrow horizontal band highlighted in the overlap.
             Right caption "MI sees:" highlights the full overlap area.
  Phase 3 — Hold on final frame showing the contrast clearly.

Key message: Pearson counts only the linearly-aligned slice of overlap;
MI counts the entire overlap area regardless of shape.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyArrowPatch

DARK_BG  = "#1a1a2e"
CLR_X    = "#e94560"    # red for X circle
CLR_Y    = "#4ecdc4"    # teal for Y circle
CLR_OVL  = "#f7b731"    # amber for overlap
CLR_LINE = "#a29bfe"    # purple for the Pearson "ruler slice"
TEXT_CLR = "#e2e8f0"
FADE_CLR = "#ffffff"

APPROACH_F = 40   # circles drift together
LABEL_F    = 20   # labels fade in
SPLIT_F    = 30   # Pearson vs MI comparison appears
HOLD_F     = 35   # hold at end
TOTAL_F    = APPROACH_F + LABEL_F + SPLIT_F + HOLD_F

RADIUS = 0.30
Y_POS  = 0.50


def lerp(a, b, t):
    return a + (b - a) * np.clip(t, 0, 1)


def circle_overlap_x_range(cx1, cx2, r):
    """Return left and right x of the lens intersection."""
    d = abs(cx2 - cx1)
    if d >= 2 * r:
        return None
    x_mid = (cx1 + cx2) / 2
    half_chord = np.sqrt(r**2 - (d / 2)**2)
    return x_mid - half_chord, x_mid + half_chord


def make_animation():
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.set_title("Pearson vs Mutual Information — The Venn Intuition",
                 color=TEXT_CLR, fontsize=12, fontweight="bold", pad=12)

    # Starting positions (partially overlapping) and final positions (more overlap)
    cx1_start, cx1_end = 0.30, 0.38
    cx2_start, cx2_end = 0.70, 0.62

    # Draw circles (will update center in animation)
    circ_x = Circle((cx1_start, Y_POS), RADIUS, color=CLR_X, alpha=0.35,
                     zorder=2, lw=2)
    circ_y = Circle((cx2_start, Y_POS), RADIUS, color=CLR_Y, alpha=0.35,
                     zorder=2, lw=2)
    ax.add_patch(circ_x)
    ax.add_patch(circ_y)

    edge_x = Circle((cx1_start, Y_POS), RADIUS, fill=False, edgecolor=CLR_X,
                     lw=2.5, zorder=3)
    edge_y = Circle((cx2_start, Y_POS), RADIUS, fill=False, edgecolor=CLR_Y,
                     lw=2.5, zorder=3)
    ax.add_patch(edge_x)
    ax.add_patch(edge_y)

    # Labels for X and Y
    lbl_x = ax.text(cx1_start, Y_POS, "X", color=CLR_X, ha="center",
                    va="center", fontsize=20, fontweight="bold",
                    zorder=4, alpha=1)
    lbl_y = ax.text(cx2_start, Y_POS, "Y", color=CLR_Y, ha="center",
                    va="center", fontsize=20, fontweight="bold",
                    zorder=4, alpha=1)

    # Shared info label
    lbl_shared = ax.text(0.5, Y_POS + 0.05, "Shared\nInformation",
                         color=CLR_OVL, ha="center", va="center",
                         fontsize=9, fontweight="bold", alpha=0, zorder=5)

    # Bottom annotations
    ann_pearson = ax.text(
        0.25, 0.13,
        "Pearson: counts only\nlinearly-aligned overlap →",
        color=CLR_LINE, ha="center", va="center", fontsize=9, alpha=0,
        zorder=5
    )
    ann_mi = ax.text(
        0.75, 0.13,
        "← MI: counts the full\noverlap — any shape",
        color=CLR_OVL, ha="center", va="center", fontsize=9, alpha=0,
        zorder=5
    )

    # Ruler-slice rectangle (narrow horizontal band through overlap)
    ruler_rect = patches.FancyBboxPatch(
        (0.5 - 0.14, Y_POS - 0.018), 0.28, 0.036,
        boxstyle="round,pad=0.005",
        linewidth=1.5, edgecolor=CLR_LINE, facecolor=CLR_LINE,
        alpha=0, zorder=6
    )
    ax.add_patch(ruler_rect)

    # Overlap shading (approximate lens with a semi-transparent fill)
    # We'll draw a filled circle patch that represents overlap growing
    ovl_patch = Circle((0.5, Y_POS), 0.001, color=CLR_OVL, alpha=0,
                        zorder=1)
    ax.add_patch(ovl_patch)

    def set_positions(cx1, cx2):
        for circle, edge, cx in [(circ_x, edge_x, cx1),
                                  (circ_y, edge_y, cx2)]:
            circle.set_center((cx, Y_POS))
            edge.set_center((cx, Y_POS))
        lbl_x.set_position((cx1, Y_POS))
        lbl_y.set_position((cx2, Y_POS))

        # Update overlap circle size/position
        d = abs(cx2 - cx1)
        if d < 2 * RADIUS:
            ovl_r = RADIUS - d / 2 + 0.02   # rough proxy for lens radius
            ovl_patch.set_center((0.5, Y_POS))
            ovl_patch.set_radius(max(0.001, ovl_r))
            ovl_patch.set_alpha(0.4)
        else:
            ovl_patch.set_radius(0.001)
            ovl_patch.set_alpha(0)

    def animate(frame):
        # Phase 1: drift together
        if frame < APPROACH_F:
            t = frame / (APPROACH_F - 1)
            cx1 = lerp(cx1_start, cx1_end, t)
            cx2 = lerp(cx2_start, cx2_end, t)
            set_positions(cx1, cx2)
            lbl_shared.set_alpha(0)
            ann_pearson.set_alpha(0)
            ann_mi.set_alpha(0)
            ruler_rect.set_alpha(0)

        # Phase 2: labels fade in
        elif frame < APPROACH_F + LABEL_F:
            set_positions(cx1_end, cx2_end)
            t = (frame - APPROACH_F) / LABEL_F
            lbl_shared.set_alpha(t)
            ann_pearson.set_alpha(0)
            ann_mi.set_alpha(0)
            ruler_rect.set_alpha(0)

        # Phase 3: Pearson vs MI split
        elif frame < APPROACH_F + LABEL_F + SPLIT_F:
            set_positions(cx1_end, cx2_end)
            t = (frame - APPROACH_F - LABEL_F) / SPLIT_F
            lbl_shared.set_alpha(1)
            ann_pearson.set_alpha(t)
            ann_mi.set_alpha(t)
            ruler_rect.set_alpha(min(0.55, t * 0.9))

        # Phase 4: hold
        else:
            set_positions(cx1_end, cx2_end)
            lbl_shared.set_alpha(1)
            ann_pearson.set_alpha(1)
            ann_mi.set_alpha(1)
            ruler_rect.set_alpha(0.55)

        return (circ_x, circ_y, edge_x, edge_y, lbl_x, lbl_y,
                ovl_patch, lbl_shared, ann_pearson, ann_mi, ruler_rect)

    ani = animation.FuncAnimation(
        fig, animate,
        frames=TOTAL_F,
        interval=65,
        blit=True
    )

    out_path = "../img/ch03-pearson-mi-venn.gif"
    ani.save(out_path, writer="pillow", fps=15, dpi=110)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    make_animation()
