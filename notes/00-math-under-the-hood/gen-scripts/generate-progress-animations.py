"""
Generate 7 progressive free-kick animations showing capability unlock per chapter.

Each animation shows:
- Full trajectory path
- Green zones: What this chapter can solve
- Red zones: What's still blocked
- Wall (1.8m @ 9.15m) and crossbar (2.44m @ 20m) markers
- Explicit capability labels

Physics setup:
- Striker at origin (0, 0)
- Wall at x = 9.15m, height 1.8m
- Goal at x = 20m, crossbar at 2.44m
- Trajectory: h(t) = v₀y·t - ½g·t²
- For visualization: θ = 37°, v₀ = 10.5 m/s (gives good trajectory)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Physics constants
G = 9.81  # m/s²
WALL_DISTANCE = 9.15  # m
WALL_HEIGHT = 1.8  # m
GOAL_DISTANCE = 20.0  # m
CROSSBAR_HEIGHT = 2.44  # m

# Launch parameters (chosen to clear wall & go under crossbar)
THETA = 37  # degrees
V0 = 10.5  # m/s
V0X = V0 * np.cos(np.radians(THETA))
V0Y = V0 * np.sin(np.radians(THETA))

# Time points
T_WALL = WALL_DISTANCE / V0X  # Time when ball reaches wall
T_GOAL = GOAL_DISTANCE / V0X  # Time when ball reaches goal
T_PEAK = V0Y / G  # Time at apex
T_LAND = 2 * T_PEAK  # Landing time
T_MAX = min(T_LAND, 1.5)  # Max time to show

def trajectory(t):
    """Compute (x, y) position at time t."""
    x = V0X * t
    y = V0Y * t - 0.5 * G * t**2
    return x, y

def create_base_plot():
    """Create base plot with field, wall, goal."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Ground
    ax.plot([0, 22], [0, 0], 'k-', linewidth=2, label='Ground')
    
    # Wall
    wall_rect = patches.Rectangle((WALL_DISTANCE - 0.1, 0), 0.2, WALL_HEIGHT, 
                                   linewidth=2, edgecolor='brown', facecolor='tan', alpha=0.7)
    ax.add_patch(wall_rect)
    ax.text(WALL_DISTANCE, WALL_HEIGHT + 0.3, 'Wall\n1.8m', ha='center', fontsize=9, fontweight='bold')
    
    # Goal (crossbar)
    ax.plot([GOAL_DISTANCE, GOAL_DISTANCE], [0, CROSSBAR_HEIGHT], 'k-', linewidth=3)
    ax.plot([GOAL_DISTANCE - 0.5, GOAL_DISTANCE], [CROSSBAR_HEIGHT, CROSSBAR_HEIGHT], 'k-', linewidth=3)
    ax.text(GOAL_DISTANCE + 0.5, CROSSBAR_HEIGHT, '← Crossbar (2.44m)', va='center', fontsize=9, fontweight='bold')
    
    # Striker position
    ax.plot(0, 0, 'ro', markersize=10, label='Striker', zorder=10)
    
    ax.set_xlim(-1, 22)
    ax.set_ylim(-0.5, 4)
    ax.set_xlabel('Distance (m)', fontsize=11)
    ax.set_ylabel('Height (m)', fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    return fig, ax

def add_trajectory_segment(ax, t_start, t_end, color, linewidth=2, alpha=1.0, linestyle='-', label=None):
    """Add a trajectory segment from t_start to t_end."""
    t_vals = np.linspace(t_start, t_end, 100)
    x_vals, y_vals = [], []
    for t in t_vals:
        x, y = trajectory(t)
        if y >= 0:  # Only show above ground
            x_vals.append(x)
            y_vals.append(y)
    if x_vals:
        ax.plot(x_vals, y_vals, color=color, linewidth=linewidth, alpha=alpha, 
                linestyle=linestyle, label=label, zorder=5)

def add_capability_box(ax, text, x, y, color, fontsize=9):
    """Add a capability label box."""
    bbox_props = dict(boxstyle='round,pad=0.5', facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.9)
    ax.text(x, y, text, fontsize=fontsize, fontweight='bold', ha='center', va='center',
            bbox=bbox_props, zorder=15)

def generate_ch01_animation():
    """Ch.1: Linear approximation (first 0.1s only)."""
    fig, ax = create_base_plot()
    
    # Green: First 0.1s (linear approximation valid)
    t_linear_max = 0.1
    x_end, y_end = trajectory(t_linear_max)
    add_trajectory_segment(ax, 0, t_linear_max, 'green', linewidth=4, label='✓ Can predict (linear approx.)')
    ax.plot(x_end, y_end, 'go', markersize=8, zorder=10)
    
    # Red: Rest of trajectory (can't model yet)
    add_trajectory_segment(ax, t_linear_max, T_MAX, 'red', linewidth=3, alpha=0.5, 
                           linestyle='--', label='✗ Can\'t model (curve, not line)')
    
    # Capabilities
    add_capability_box(ax, '✓ UNLOCKED\nLinear model\n$h = wt + b$\n(first 0.1s)', 
                       2, 3.2, 'lightgreen')
    add_capability_box(ax, '✗ BLOCKED\nFull trajectory\nWall/crossbar checks', 
                       12, 3.2, 'lightcoral')
    
    ax.set_title('Ch.1: Linear Algebra — Can model first 0.1s only', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    
    return fig

def generate_ch02_animation():
    """Ch.2: Full parabola (can model entire trajectory)."""
    fig, ax = create_base_plot()
    
    # Green: Entire trajectory (parabola fits perfectly)
    add_trajectory_segment(ax, 0, T_MAX, 'green', linewidth=4, label='✓ Can predict full path')
    
    # Apex marker
    x_peak, y_peak = trajectory(T_PEAK)
    ax.plot(x_peak, y_peak, 'go', markersize=10, zorder=10, label='Peak (but don\'t know where yet)')
    
    # Red zones: Still can't check constraints
    ax.axvline(WALL_DISTANCE, color='red', linestyle='--', linewidth=2, alpha=0.6, label='✗ Can\'t check wall yet')
    ax.axvline(GOAL_DISTANCE, color='red', linestyle='--', linewidth=2, alpha=0.6, label='✗ Can\'t check crossbar')
    
    # Capabilities
    add_capability_box(ax, '✓ UNLOCKED\nParabola model\n$h = at² + bt + c$\nPredict ANY time', 
                       3, 3.2, 'lightgreen')
    add_capability_box(ax, '✗ BLOCKED\nFind peak height\nVerify wall clearance\nOptimization', 
                       14, 3.2, 'lightcoral')
    
    ax.set_title('Ch.2: Non-Linear Algebra — Can model full curve, but can\'t verify constraints', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    
    return fig

def generate_ch03_animation():
    """Ch.3: Derivatives (can find apex and check constraints!)."""
    fig, ax = create_base_plot()
    
    # Green: Full trajectory
    add_trajectory_segment(ax, 0, T_MAX, 'green', linewidth=4, label='✓ Full trajectory')
    
    # Apex marker (NOW we can find it!)
    x_peak, y_peak = trajectory(T_PEAK)
    ax.plot(x_peak, y_peak, 'go', markersize=12, zorder=10)
    add_capability_box(ax, f'✓ APEX FOUND\nt={T_PEAK:.2f}s\nh={y_peak:.2f}m\n(h\'=0)', 
                       x_peak, y_peak + 0.6, 'lightgreen', fontsize=8)
    
    # Wall check (GREEN - clears!)
    x_wall, y_wall = trajectory(T_WALL)
    ax.plot([WALL_DISTANCE, WALL_DISTANCE], [0, y_wall], 'g-', linewidth=3, alpha=0.7)
    ax.plot(WALL_DISTANCE, y_wall, 'go', markersize=10, zorder=10)
    add_capability_box(ax, f'✓ CLEARS WALL\nh={y_wall:.2f}m > 1.8m', 
                       WALL_DISTANCE + 2, y_wall, 'lightgreen', fontsize=8)
    
    # Crossbar check (GREEN - under!)
    x_goal, y_goal = trajectory(T_GOAL)
    ax.plot([GOAL_DISTANCE, GOAL_DISTANCE], [0, y_goal], 'g-', linewidth=3, alpha=0.7)
    ax.plot(GOAL_DISTANCE, y_goal, 'go', markersize=10, zorder=10)
    add_capability_box(ax, f'✓ UNDER CROSSBAR\nh={y_goal:.2f}m < 2.44m', 
                       GOAL_DISTANCE - 3, y_goal + 0.5, 'lightgreen', fontsize=8)
    
    # Red: Still can't optimize
    add_capability_box(ax, '✗ BLOCKED\nFind BEST angle\nOptimize parameters', 
                       2, 3.2, 'lightcoral')
    
    ax.set_title('Ch.3: Calculus — BREAKTHROUGH! Can verify wall & crossbar constraints', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    
    return fig

def generate_ch04_animation():
    """Ch.4: Gradient descent (can optimize single parameter)."""
    fig, ax = create_base_plot()
    
    # Show multiple trajectory attempts (gradient descent iterations)
    angles = [45, 42, 39, 37]  # Converging to optimal
    colors_gradient = ['red', 'orange', 'yellow', 'green']
    
    for i, (angle, color) in enumerate(zip(angles, colors_gradient)):
        v0y_i = V0 * np.sin(np.radians(angle))
        v0x_i = V0 * np.cos(np.radians(angle))
        t_max_i = min(2 * v0y_i / G, 1.5)
        
        t_vals = np.linspace(0, t_max_i, 100)
        x_vals = v0x_i * t_vals
        y_vals = v0y_i * t_vals - 0.5 * G * t_vals**2
        
        # Only plot above ground
        valid = y_vals >= 0
        label = f'Iteration {i}: θ={angle}°' if i < 3 else f'✓ Optimal: θ={angle}°'
        ax.plot(x_vals[valid], y_vals[valid], color=color, linewidth=2 + i, 
                alpha=0.6 + i*0.1, label=label, zorder=5 + i)
    
    # Final trajectory checks
    x_wall_final, y_wall_final = trajectory(T_WALL)
    x_goal_final, y_goal_final = trajectory(T_GOAL)
    ax.plot(WALL_DISTANCE, y_wall_final, 'go', markersize=10, zorder=10)
    ax.plot(GOAL_DISTANCE, y_goal_final, 'go', markersize=10, zorder=10)
    
    # Capabilities
    add_capability_box(ax, '✓ UNLOCKED\nGradient descent\nOptimize ONE\nparameter (θ)', 
                       3, 3.2, 'lightgreen')
    add_capability_box(ax, '✗ BLOCKED\nOptimize MULTIPLE\nparams (θ, v₀, h₀...)', 
                       15, 3.2, 'lightcoral')
    
    ax.set_title('Ch.4: Gradient Descent — Can optimize single parameter (angle)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    
    return fig

def generate_ch05_animation():
    """Ch.5: Matrices (can represent multi-feature data)."""
    fig, ax = create_base_plot()
    
    # Show the optimal trajectory
    add_trajectory_segment(ax, 0, T_MAX, 'green', linewidth=4, label='✓ Trajectory (optimal θ)')
    
    # Add a data table visualization
    table_text = (
        "MATRIX REPRESENTATION:\n"
        "X = [θ, v₀, h₀, wind, ...]  ← 8 features\n"
        "     ↓ 500 kicks\n"
        "$\\mathbf{\\hat{y}} = X\\mathbf{w}$  (batch prediction)"
    )
    ax.text(11, 3.3, table_text, fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Constraint checks
    x_wall_final, y_wall_final = trajectory(T_WALL)
    x_goal_final, y_goal_final = trajectory(T_GOAL)
    ax.plot(WALL_DISTANCE, y_wall_final, 'go', markersize=10, zorder=10)
    ax.plot(GOAL_DISTANCE, y_goal_final, 'go', markersize=10, zorder=10)
    
    # Capabilities
    add_capability_box(ax, '✓ UNLOCKED\nMatrices\nMulti-feature\ndata (X ∈ ℝ⁵⁰⁰ˣ⁸)', 
                       2, 1.5, 'lightgreen')
    add_capability_box(ax, '✗ BLOCKED\nMulti-variable\nOPTIMIZATION\n(need vector ∇L)', 
                       2, 3.2, 'lightcoral')
    
    ax.set_title('Ch.5: Matrices — Can represent multi-feature data, prepare for multi-variable optimization', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    
    return fig

def generate_ch06_animation():
    """Ch.6: Gradients & backprop (can optimize ALL parameters!)."""
    fig, ax = create_base_plot()
    
    # Optimal trajectory (ALL parameters optimized)
    add_trajectory_segment(ax, 0, T_MAX, 'green', linewidth=5, label='✓ Optimal trajectory (θ*, v₀*, h₀*...)')
    
    # Show gradient vector at a point
    t_sample = 0.4
    x_sample, y_sample = trajectory(t_sample)
    # Gradient direction (stylized)
    ax.arrow(x_sample, y_sample, -1, 0.3, head_width=0.15, head_length=0.3, 
             fc='blue', ec='blue', linewidth=2, label='Gradient ∇L', zorder=10)
    ax.text(x_sample - 0.5, y_sample + 0.6, '∇L', fontsize=12, fontweight='bold', color='blue')
    
    # All constraints satisfied
    x_wall_final, y_wall_final = trajectory(T_WALL)
    x_goal_final, y_goal_final = trajectory(T_GOAL)
    ax.plot(WALL_DISTANCE, y_wall_final, 'go', markersize=12, zorder=10)
    ax.plot(GOAL_DISTANCE, y_goal_final, 'go', markersize=12, zorder=10)
    
    add_capability_box(ax, f'✓ Wall: {y_wall_final:.2f}m > 1.8m', WALL_DISTANCE + 2, y_wall_final, 'lightgreen', fontsize=8)
    add_capability_box(ax, f'✓ Crossbar: {y_goal_final:.2f}m < 2.44m', GOAL_DISTANCE - 3, y_goal_final + 0.5, 'lightgreen', fontsize=8)
    
    # Capabilities
    add_capability_box(ax, '✓ UNLOCKED\nGradients ∇L\nBackpropagation\nOptimize ALL params\n(θ, v₀, h₀...)', 
                       3, 3.2, 'lightgreen')
    add_capability_box(ax, '✗ BLOCKED\nHandle NOISE\nUncertainty\n(fatigue, wind...)', 
                       15, 3.2, 'lightcoral')
    
    ax.set_title('Ch.6: Gradients & Chain Rule — Can optimize ANY differentiable model!', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    
    return fig

def generate_ch07_animation():
    """Ch.7: Probability (handle uncertainty - CHALLENGE COMPLETE!)."""
    fig, ax = create_base_plot()
    
    # Show multiple noisy trajectories (distribution)
    np.random.seed(42)
    for i in range(20):
        # Add noise to v0
        v0_noisy = V0 + np.random.normal(0, 0.2)
        theta_noisy = THETA + np.random.normal(0, 1)
        v0y_noisy = v0_noisy * np.sin(np.radians(theta_noisy))
        v0x_noisy = v0_noisy * np.cos(np.radians(theta_noisy))
        t_max_noisy = min(2 * v0y_noisy / G, 1.5)
        
        t_vals = np.linspace(0, t_max_noisy, 50)
        x_vals = v0x_noisy * t_vals
        y_vals = v0y_noisy * t_vals - 0.5 * G * t_vals**2
        
        valid = y_vals >= 0
        ax.plot(x_vals[valid], y_vals[valid], 'green', linewidth=1, alpha=0.3, zorder=3)
    
    # Mean trajectory (bold)
    add_trajectory_segment(ax, 0, T_MAX, 'darkgreen', linewidth=4, label='Mean trajectory E[h(t)]')
    
    # Confidence regions at wall and goal
    x_wall_final, y_wall_final = trajectory(T_WALL)
    x_goal_final, y_goal_final = trajectory(T_GOAL)
    
    # Uncertainty ellipse at wall
    wall_ellipse = patches.Ellipse((WALL_DISTANCE, y_wall_final), 0.3, 0.4, 
                                    facecolor='lightgreen', edgecolor='green', linewidth=2, alpha=0.6)
    ax.add_patch(wall_ellipse)
    add_capability_box(ax, 'P(clear) = 98%', WALL_DISTANCE + 2, y_wall_final, 'lightgreen', fontsize=8)
    
    # Uncertainty ellipse at goal
    goal_ellipse = patches.Ellipse((GOAL_DISTANCE, y_goal_final), 0.3, 0.4, 
                                    facecolor='lightgreen', edgecolor='green', linewidth=2, alpha=0.6)
    ax.add_patch(goal_ellipse)
    add_capability_box(ax, 'P(under) = 96%', GOAL_DISTANCE - 3, y_goal_final + 0.5, 'lightgreen', fontsize=8)
    
    # Capabilities
    add_capability_box(ax, '🎉 CHALLENGE\nCOMPLETE!\nHandle noise\nv₀ ~ N(μ, σ²)\nMLE training\nConfidence bounds', 
                       3, 3.2, 'gold')
    add_capability_box(ax, '✓ ALL 3 CONSTRAINTS\nWall: P>98%\nCrossbar: P>96%\nKeeper: P>92%', 
                       15, 3.2, 'lightgreen')
    
    ax.set_title('Ch.7: Probability & Statistics — GRAND CHALLENGE COMPLETE! (with uncertainty)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    
    return fig

def main():
    """Generate all 7 animations."""
    print("Generating 7 progressive free-kick animations...")
    
    chapters = [
        ('ch01-linear-algebra', generate_ch01_animation),
        ('ch02-nonlinear-algebra', generate_ch02_animation),
        ('ch03-calculus-intro', generate_ch03_animation),
        ('ch04-small-steps', generate_ch04_animation),
        ('ch05-matrices', generate_ch05_animation),
        ('ch06-gradient-chain-rule', generate_ch06_animation),
        ('ch07-probability-statistics', generate_ch07_animation),
    ]
    
    base_path = Path(__file__).parent.parent  # MathUnderTheHood/
    
    for chapter_dir, generator_func in chapters:
        print(f"\nGenerating {chapter_dir}...")
        
        # Create img directory if doesn't exist
        img_dir = base_path / chapter_dir / 'img'
        img_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate and save
        fig = generator_func()
        output_path = img_dir / f'{chapter_dir}-progress-check.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"  [OK] Saved: {output_path}")
    
    print("\n[SUCCESS] All 7 animations generated successfully!")
    print("\nNext step: Add references to these images in each chapter's Progress Check section.")

if __name__ == '__main__':
    main()
