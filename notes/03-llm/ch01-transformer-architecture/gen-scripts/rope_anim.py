import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_rope_clean(sentence: str):
    tokens = sentence.split()
    num_tokens = len(tokens)

    if num_tokens < 2 or num_tokens > 10:
        raise ValueError(f"Sentence must have between 2 and 10 tokens. Found {num_tokens}.")

    dim = 6
    half_dim = dim // 2

    thetas = np.array([0.7, 0.3, 0.08])
    radii = [1.2, 0.8, 0.4]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blue, Orange, Green
    z_heights = [0.6, 0.3, 0.0]

    # --- 1. PLOT AND GRID ARCHITECTURE ---
    fig = plt.figure(figsize=(16, 8))
    grid = plt.GridSpec(2, num_tokens, height_ratios=[1.3, 1.2], hspace=0.4, wspace=0.3)

    # Top Row: 3D Token Towers
    axes_3d = []
    for m in range(num_tokens):
        ax = fig.add_subplot(grid[0, m], projection='3d')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-0.2, 1.0)

        # Color the subplot titles to match token activation states dynamically later
        ax.set_title(f"T{m}: \"{tokens[m]}\"", fontsize=12, fontweight='bold', pad=4)
        ax.view_init(elev=20, azim=-45)
        ax.axis('off')
        axes_3d.append(ax)

    # Bottom Row Left: Matrix Layout Base
    ax_matrix = fig.add_subplot(grid[1, :num_tokens-3])
    ax_matrix.axis('off')

    # Bottom Row Right: Legend Layout Base
    ax_legend = fig.add_subplot(grid[1, num_tokens-3:])
    ax_legend.axis('off')

    # --- 2. BUILD STATIC PLAIN-TEXT LEGEND ---
    ax_legend.text(0.0, 0.95, "=== ROPE LAYOUT LEGEND ===", fontfamily='monospace', fontsize=11, fontweight='bold', va='top')
    ax_legend.text(0.0, 0.82, "BLUE   (Top Disc) : Dims (q0, q3) | Fast (theta = 0.70)", fontfamily='monospace', fontsize=10, color=colors[0], fontweight='bold', va='top')
    ax_legend.text(0.0, 0.70, "ORANGE (Mid Disc) : Dims (q1, q4) | Med  (theta = 0.30)", fontfamily='monospace', fontsize=10, color=colors[1], fontweight='bold', va='top')
    ax_legend.text(0.0, 0.58, "GREEN  (Bot Disc) : Dims (q2, q5) | Slow (theta = 0.08)", fontfamily='monospace', fontsize=10, color=colors[2], fontweight='bold', va='top')

    strategy_text = (
        "Memory Slicing Strategy:\n"
        " Vector = [  q0,  q1,  q2  |  q3,  q4,  q5  ]\n"
        "          |--- Cosines ---|---  Sines  ---|\n"
        " Pairs  = (q0,q3), (q1,q4), (q2,q5)"
    )
    ax_legend.text(0.0, 0.44, strategy_text, fontfamily='monospace', fontsize=10, va='top')

    # --- 3. DRAW REFERENCE HOOPS ---
    for m in range(num_tokens):
        ax = axes_3d[m]
        ax.plot([0, 0], [0, 0], [-0.1, 0.8], color='gray', alpha=0.3, lw=2)
        for i, r in enumerate(radii):
            alpha_angles = np.linspace(0, 2 * np.pi, 100)
            cx = r * np.cos(alpha_angles)
            cy = r * np.sin(alpha_angles)
            cz = np.full_like(alpha_angles, z_heights[i])
            ax.plot(cx, cy, cz, color=colors[i], ls=':', alpha=0.3)

    # Graphic pointers
    vector_lines = []
    vector_dots = []
    for m in range(num_tokens):
        lines_at_m = []
        dots_at_m = []
        for i in range(half_dim):
            l, = axes_3d[m].plot([], [], [], lw=3, solid_capstyle='round')
            d, = axes_3d[m].plot([], [], [], 'o', ms=7)
            lines_at_m.append(l)
            dots_at_m.append(d)
        vector_lines.append(lines_at_m)
        vector_dots.append(dots_at_m)

    # Pre-allocate monolithic string rows to eliminate horizontal collisions
    header_1 = ax_matrix.text(0.0, 1.0, "PRODUCTION ROPE TENSOR VALUES", fontfamily='monospace', fontsize=11, fontweight='bold', va='top')
    header_2 = ax_matrix.text(0.0, 0.92, "", fontfamily='monospace', fontsize=10, va='top')

    row_text_objects = []
    for m in range(num_tokens):
        y_pos = 0.78 - (m * 0.11)
        t_obj = ax_matrix.text(0.0, y_pos, "", fontfamily='monospace', fontsize=10, va='top')
        row_text_objects.append(t_obj)

    frames_per_token = 50
    total_frames = num_tokens * frames_per_token

    # --- 4. RENDER CONTEXT METHOD ---
    def update(frame):
        active_token_idx = min(frame // frames_per_token, num_tokens - 1)
        local_progress = (frame % frames_per_token) / float(frames_per_token - 1)

        header_2.set_text(f"Attention Target Sequence Phase -> Focusing on Token T{active_token_idx}")
        updated_artists = [header_2]

        for m in range(num_tokens):
            row_values = np.zeros(dim)

            if m < active_token_idx:
                progress = 1.0
                status_str = "[Active]"
                # Match the 3D subplot title text to the processing state
                axes_3d[m].title.set_color('green')
            elif m == active_token_idx:
                progress = local_progress
                status_str = "[Spin  ]"
                axes_3d[m].title.set_color('darkorange')
            else:
                progress = 0.0
                status_str = "[Latent]"
                axes_3d[m].title.set_color('gray')

            for i in range(half_dim):
                target_angle = m * thetas[i]
                current_angle = target_angle * progress

                x = radii[i] * np.cos(current_angle)
                y = radii[i] * np.sin(current_angle)
                z = z_heights[i]

                row_values[i] = x / radii[i]
                row_values[i + half_dim] = y / radii[i]

                # Update 3D Canvas Geometry
                vector_lines[m][i].set_data([0, x], [0, y])
                vector_lines[m][i].set_3d_properties([z, z])
                vector_lines[m][i].set_color(colors[i] if progress > 0 or m == 0 else '#e0e0e0')
                vector_lines[m][i].set_alpha(1.0 if progress > 0 or m == 0 else 0.12)

                vector_dots[m][i].set_data([x], [y])
                vector_dots[m][i].set_3d_properties([z])
                vector_dots[m][i].set_color(colors[i] if progress > 0 or m == 0 else '#e0e0e0')
                vector_dots[m][i].set_alpha(1.0 if progress > 0 or m == 0 else 0.12)

            # Build clean, spaced string fragments to prevent layout shifting
            left_str = ", ".join([f"{val:5.2f}" for val in row_values[:half_dim]])
            right_str = ", ".join([f"{val:5.2f}" for val in row_values[half_dim:]])

            # Pad token names cleanly using standard string format parameters
            token_padded = f"\"{tokens[m]}\""
            full_row_line = f"{status_str:<9} T{m} ({token_padded:<10}) -> [ {left_str}  |  {right_str} ]"

            row_text_objects[m].set_text(full_row_line)

            # Apply uniform color indicating active block status to the text line
            if m == active_token_idx:
                row_text_objects[m].set_color('darkorange')
            elif m < active_token_idx:
                row_text_objects[m].set_color('black')
            else:
                row_text_objects[m].set_color('#b0b0b0')

            updated_artists.append(row_text_objects[m])

        return [item for sublist in vector_lines for item in sublist] + \
               [item for sublist in vector_dots for item in sublist] + updated_artists

    ani = FuncAnimation(fig, update, frames=total_frames, blit=True, interval=40, repeat=True)
    ani.save(r"img/rope_mechanism.gif", writer='pillow', fps=25)
    plt.show()

animate_rope_clean("Attention is all you need")
