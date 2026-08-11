"""Compose cinematic free-kick animations over a real match photograph.

The photograph is Michael Barera's "Detroit City FC v. San Antonio FC 2023 20
(free kick)", licensed CC BY-SA 4.0. The ball motion and every displayed value
come from the same projectile model used in the notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import gc
import math
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "images"
SOURCE = IMAGE_DIR / "free-kick-stadium-source.jpg"
WIDTH, HEIGHT = 1600, 900

# The same physical world used by the notebook.
G = 9.81
V0 = 20.0
BALL_RADIUS = 0.11
WALL_X = 9.15
WALL_H = 1.8
GOAL_X = 20.0
CROSS_H = 2.44
NET_X = 21.5
TARGET_H = 1.10
GOAL_BOTTOM = BALL_RADIUS
GOAL_TOP = CROSS_H - BALL_RADIUS

# Camera calibration for the selected photograph after its 1920x1080 crop.
STRIKE_PX = (146, 574)
NET_PX_X = 1482
GROUND_START_Y = 574
GROUND_END_Y = 560
HEIGHT_SCALE = 58.0

WHITE = (247, 248, 243)
INK = (13, 23, 30)
MUTED = (178, 195, 198)
GOLD = (246, 190, 72)
AMBER = (238, 129, 55)
CYAN = (69, 205, 220)
GREEN = (75, 214, 142)
RED = (238, 91, 77)
NAVY = (5, 17, 27)


def load_font(size: int, *, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    candidates = []
    if mono:
        candidates.extend([Path(r"C:\Windows\Fonts\consolab.ttf"), Path(r"C:\Windows\Fonts\consola.ttf")])
    elif bold:
        candidates.extend([Path(r"C:\Windows\Fonts\bahnschrift.ttf"), Path(r"C:\Windows\Fonts\segoeuib.ttf")])
    else:
        candidates.extend([Path(r"C:\Windows\Fonts\segoeui.ttf"), Path(r"C:\Windows\Fonts\bahnschrift.ttf")])
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default(size=size)


FONT_HERO = load_font(48, bold=True)
FONT_TITLE = load_font(28, bold=True)
FONT_LABEL = load_font(17, bold=True)
FONT_BODY = load_font(22)
FONT_SMALL = load_font(15)
FONT_MONO = load_font(22, mono=True)
FONT_MONO_BIG = load_font(35, mono=True)


@dataclass(frozen=True)
class LossState:
    theta: float
    h_wall: float
    h_goal: float
    h_net: float
    wall_gap: float
    goal_low_gap: float
    goal_high_gap: float
    net_error: float
    loss: float
    gradient: float


def height(x: np.ndarray | float, theta_deg: float) -> np.ndarray | float:
    theta = np.radians(theta_deg)
    return x * np.tan(theta) - G * np.asarray(x) ** 2 / (2 * V0**2 * np.cos(theta) ** 2)


def state_at(x: float, theta_deg: float) -> dict[str, float]:
    theta = math.radians(theta_deg)
    vx = V0 * math.cos(theta)
    t = x / vx
    vy = V0 * math.sin(theta) - G * t
    return {
        "x": x,
        "t": t,
        "y": float(height(x, theta_deg)),
        "vx": vx,
        "vy": vy,
        "speed": math.hypot(vx, vy),
    }


def dh_dtheta(x: float, theta_deg: float) -> float:
    theta = math.radians(theta_deg)
    sec2 = 1.0 / math.cos(theta) ** 2
    a = G * x**2 / (2 * V0**2)
    return (x * sec2 - 2 * a * sec2 * math.tan(theta)) * math.pi / 180.0


def loss_parts(theta_deg: float) -> tuple[float, float, float, float]:
    h_wall = float(height(WALL_X, theta_deg))
    h_goal = float(height(GOAL_X, theta_deg))
    h_net = float(height(NET_X, theta_deg))
    return (
        max(0.0, WALL_H + BALL_RADIUS - h_wall),
        max(0.0, GOAL_BOTTOM - h_goal),
        max(0.0, h_goal - GOAL_TOP),
        h_net - TARGET_H,
    )


def kick_loss(theta_deg: float) -> float:
    wall_gap, goal_low_gap, goal_high_gap, net_error = loss_parts(theta_deg)
    return wall_gap**2 + 2 * goal_low_gap**2 + 2 * goal_high_gap**2 + 0.25 * net_error**2


def analytic_gradient(theta_deg: float) -> float:
    wall_gap, goal_low_gap, goal_high_gap, net_error = loss_parts(theta_deg)
    gradient = 0.5 * net_error * dh_dtheta(NET_X, theta_deg)
    if wall_gap > 0:
        gradient -= 2 * wall_gap * dh_dtheta(WALL_X, theta_deg)
    if goal_low_gap > 0:
        gradient -= 4 * goal_low_gap * dh_dtheta(GOAL_X, theta_deg)
    if goal_high_gap > 0:
        gradient += 4 * goal_high_gap * dh_dtheta(GOAL_X, theta_deg)
    return gradient


def numerical_gradient(theta_deg: float, epsilon: float = 1e-4) -> float:
    return (kick_loss(theta_deg + epsilon) - kick_loss(theta_deg - epsilon)) / (2 * epsilon)


def describe(theta_deg: float) -> LossState:
    wall_gap, goal_low_gap, goal_high_gap, net_error = loss_parts(theta_deg)
    return LossState(
        theta=theta_deg,
        h_wall=float(height(WALL_X, theta_deg)),
        h_goal=float(height(GOAL_X, theta_deg)),
        h_net=float(height(NET_X, theta_deg)),
        wall_gap=wall_gap,
        goal_low_gap=goal_low_gap,
        goal_high_gap=goal_high_gap,
        net_error=net_error,
        loss=kick_loss(theta_deg),
        gradient=analytic_gradient(theta_deg),
    )


def optimize(start: float = 35.0, learning_rate: float = 2.0, steps: int = 36) -> list[LossState]:
    theta = start
    history = [describe(theta)]
    for _ in range(steps):
        theta = float(np.clip(theta - learning_rate * analytic_gradient(theta), 8.0, 45.0))
        history.append(describe(theta))
    return history


def verify_model(history: list[LossState]) -> None:
    errors = [abs(analytic_gradient(theta) - numerical_gradient(theta)) for theta in np.linspace(14, 36, 12)]
    final = history[-1]
    if max(errors) > 1e-5:
        raise RuntimeError(f"analytic gradient mismatch: {max(errors):.3e}")
    if final.loss >= history[0].loss * 0.001:
        raise RuntimeError("gradient descent did not reduce the loss enough")
    if not (final.h_wall > WALL_H + BALL_RADIUS and GOAL_BOTTOM < final.h_goal < GOAL_TOP):
        raise RuntimeError("optimized trajectory is not scoreable")
    if abs(final.net_error) > 0.08:
        raise RuntimeError("optimized trajectory misses its target")
    print(
        f"verified theta {history[0].theta:.2f} -> {final.theta:.2f} deg; "
        f"loss {history[0].loss:.4f} -> {final.loss:.6f}; "
        f"gradient error {max(errors):.2e}"
    )


def smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3 - 2 * value)


def world_to_screen(x: float, y: float) -> tuple[int, int]:
    progress = x / NET_X
    px = STRIKE_PX[0] + progress * (NET_PX_X - STRIKE_PX[0])
    ground = GROUND_START_Y + progress * (GROUND_END_Y - GROUND_START_Y)
    perspective_scale = HEIGHT_SCALE * (1.0 - 0.12 * progress)
    py = ground - y * perspective_scale
    return int(px), int(py)


def trajectory_points(theta: float, end_x: float = NET_X, samples: int = 180) -> list[tuple[int, int]]:
    xs = np.linspace(0, end_x, samples)
    return [world_to_screen(float(x), max(-0.15, float(height(x, theta)))) for x in xs]


def make_vignette() -> Image.Image:
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    dx = (xx - WIDTH / 2) / (WIDTH / 2)
    dy = (yy - HEIGHT / 2) / (HEIGHT / 2)
    radius = np.sqrt(dx**2 + dy**2)
    alpha = np.clip((radius - 0.28) / 0.78, 0, 1) ** 1.7 * 180
    image = Image.new("RGBA", (WIDTH, HEIGHT), NAVY + (0,))
    image.putalpha(Image.fromarray(alpha.astype(np.uint8), mode="L"))
    return image


def prepare_plate() -> tuple[Image.Image, Image.Image]:
    source = Image.open(SOURCE).convert("RGB")
    source = source.crop((0, 100, 1920, 1180)).resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    source = ImageEnhance.Color(source).enhance(0.82)
    source = ImageEnhance.Contrast(source).enhance(1.18)
    source = ImageEnhance.Brightness(source).enhance(0.70)

    # Cool shadows and bloom the real stadium lights.
    cool = Image.new("RGB", source.size, (8, 35, 52))
    source = Image.blend(source, cool, 0.14)
    grayscale = ImageOps.grayscale(source)
    highlights = grayscale.point(lambda pixel: max(0, (pixel - 178) * 4))
    bloom = Image.new("RGBA", source.size, (255, 210, 135, 0))
    bloom.putalpha(highlights.filter(ImageFilter.GaussianBlur(18)))
    base = Image.alpha_composite(source.convert("RGBA"), bloom)
    base = Image.alpha_composite(base, make_vignette())

    # Extract the real match ball as the moving sprite.
    raw = Image.open(SOURCE).convert("RGB")
    crop = raw.crop((154, 742, 196, 784)).resize((48, 48), Image.Resampling.LANCZOS)
    mask = Image.new("L", crop.size, 0)
    ImageDraw.Draw(mask).ellipse((4, 4, 44, 44), fill=255)
    sprite = crop.convert("RGBA")
    sprite.putalpha(mask.filter(ImageFilter.GaussianBlur(1.2)))
    return base, sprite


BASE_PLATE, BALL_SPRITE = prepare_plate()


def glass_panel(frame: Image.Image, box: tuple[int, int, int, int], alpha: int = 185) -> ImageDraw.ImageDraw:
    overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle(box, radius=22, fill=NAVY + (alpha,), outline=(168, 202, 205, 85), width=2)
    frame.alpha_composite(overlay)
    return ImageDraw.Draw(frame)


def draw_header(frame: Image.Image, kicker: str, title: str, chapter: str) -> None:
    draw = ImageDraw.Draw(frame)
    draw.text((62, 46), kicker.upper(), font=FONT_LABEL, fill=GOLD)
    draw.text((62, 73), title, font=FONT_HERO, fill=WHITE)
    badge_box = (1327, 47, 1538, 91)
    draw.rounded_rectangle(badge_box, radius=20, fill=NAVY + (180,), outline=CYAN + (130,), width=2)
    draw.text((1432, 69), chapter.upper(), anchor="mm", font=FONT_SMALL, fill=WHITE)


def draw_ball(frame: Image.Image, position: tuple[int, int], angle: float, trail: list[tuple[int, int]]) -> None:
    glow = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    if len(trail) > 1:
        glow_draw.line(trail, fill=GOLD + (120,), width=12, joint="curve")
        glow_draw.line(trail, fill=WHITE + (175,), width=3, joint="curve")
    glow = glow.filter(ImageFilter.GaussianBlur(5))
    frame.alpha_composite(glow)
    if len(trail) > 1:
        ImageDraw.Draw(frame).line(trail, fill=GOLD + (190,), width=3, joint="curve")

    x, y = position
    for index, (tx, ty) in enumerate(trail[-7:-1]):
        radius = 3 + index
        ImageDraw.Draw(frame).ellipse((tx - radius, ty - radius, tx + radius, ty + radius), fill=WHITE + (20 + index * 16,))
    sprite = BALL_SPRITE.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)
    frame.alpha_composite(sprite, (x - sprite.width // 2, y - sprite.height // 2))


def draw_velocity_arrow(frame: Image.Image, state: dict[str, float], length: float = 112.0) -> None:
    position = world_to_screen(state["x"], state["y"])
    sample_time = 0.01
    sample = world_to_screen(
        state["x"] + state["vx"] * sample_time,
        state["y"] + state["vy"] * sample_time,
    )
    screen_dx = sample[0] - position[0]
    screen_dy = sample[1] - position[1]
    magnitude = math.hypot(screen_dx, screen_dy)
    if magnitude == 0:
        return

    unit_x = screen_dx / magnitude
    unit_y = screen_dy / magnitude
    start = (position[0] + unit_x * 23, position[1] + unit_y * 23)
    edge_limits = [length]
    if unit_x > 0:
        edge_limits.append((WIDTH - 24 - start[0]) / unit_x)
    elif unit_x < 0:
        edge_limits.append((24 - start[0]) / unit_x)
    if unit_y > 0:
        edge_limits.append((HEIGHT - 24 - start[1]) / unit_y)
    elif unit_y < 0:
        edge_limits.append((24 - start[1]) / unit_y)
    visible_length = max(48.0, min(edge_limits))
    tip = (start[0] + unit_x * visible_length, start[1] + unit_y * visible_length)
    normal_x, normal_y = -unit_y, unit_x
    arrowhead = [
        tip,
        (tip[0] - unit_x * 20 + normal_x * 9, tip[1] - unit_y * 20 + normal_y * 9),
        (tip[0] - unit_x * 20 - normal_x * 9, tip[1] - unit_y * 20 - normal_y * 9),
    ]

    overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.line((start, tip), fill=CYAN + (235,), width=6)
    draw.polygon(arrowhead, fill=CYAN + (245,))
    label_x = start[0] + unit_x * 34 + normal_x * 19
    label_y = start[1] + unit_y * 34 + normal_y * 19
    label_box = draw.textbbox((label_x, label_y), "VELOCITY · TANGENT", font=FONT_SMALL, anchor="lm")
    label_x += min(0, WIDTH - 16 - label_box[2]) + max(0, 16 - label_box[0])
    label_y += min(0, HEIGHT - 16 - label_box[3]) + max(0, 16 - label_box[1])
    label_box = draw.textbbox((label_x, label_y), "VELOCITY · TANGENT", font=FONT_SMALL, anchor="lm")
    draw.rounded_rectangle(
        (label_box[0] - 8, label_box[1] - 5, label_box[2] + 8, label_box[3] + 5),
        radius=8,
        fill=NAVY + (205,),
    )
    draw.text((label_x, label_y), "VELOCITY · TANGENT", font=FONT_SMALL, fill=CYAN, anchor="lm")
    frame.alpha_composite(overlay)


def value_cell(draw: ImageDraw.ImageDraw, x: int, y: int, label: str, value: str, color=WHITE) -> None:
    draw.text((x, y), label.upper(), font=FONT_SMALL, fill=MUTED)
    draw.text((x, y + 24), value, font=FONT_MONO, fill=color)


def save_frames(frames: list[Image.Image], target: Path, duration: int = 65) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    paletted = [frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=192) for frame in frames]
    paletted[0].save(
        target,
        save_all=True,
        append_images=paletted[1:],
        loop=0,
        duration=duration,
        disposal=2,
        optimize=True,
    )
    with Image.open(target) as image:
        print(target.name, image.size, f"frames={image.n_frames}", f"{target.stat().st_size / 1_048_576:.1f} MiB")
    del paletted
    del frames
    gc.collect()


def forward_phase(x: float, state: dict[str, float]) -> tuple[str, str]:
    if x < 0.8:
        return "STRIKE", "One choice sets the whole flight in motion."
    if x < WALL_X + 0.8:
        if x < WALL_X - 0.4:
            return "RISING", "The tangent points upward while vertical velocity is positive."
        return "WALL CLEARED", "The first constraint is satisfied."
    if x < GOAL_X + 0.5:
        if x >= GOAL_X - 0.7:
            return "INSIDE THE FRAME", "The ball crosses the goal plane legally."
        slope = state["vy"] / state["vx"]
        if abs(slope) < 0.025:
            return "APEX", "The tangent is nearly flat where rising becomes falling."
        if state["vy"] > 0:
            return "RISING", "The tangent points upward while vertical velocity is positive."
        return "DESCENDING", "The tangent points downward after vertical velocity turns negative."
    return "TARGET HIT", "The final error is only +0.009 m."


def render_forward(theta: float) -> None:
    frames: list[Image.Image] = []
    total = 84
    for frame_index in range(total):
        frame = BASE_PLATE.copy()
        draw_header(frame, "Forward pass", "THE KICK BECOMES CONSEQUENCES", "01 / SEE IT")
        if frame_index <= 22:
            x = WALL_X * smoothstep(frame_index / 22)
        elif frame_index <= 30:
            x = WALL_X
        elif frame_index <= 52:
            x = WALL_X + (GOAL_X - WALL_X) * smoothstep((frame_index - 30) / 22)
        elif frame_index <= 60:
            x = GOAL_X
        elif frame_index <= 73:
            x = GOAL_X + (NET_X - GOAL_X) * smoothstep((frame_index - 60) / 13)
        else:
            x = NET_X
        state = state_at(x, theta)
        phase, insight = forward_phase(x, state)
        trail_x = np.linspace(max(0.0, x - 4.2), x, 32)
        trail = [world_to_screen(float(value), max(-0.15, float(height(value, theta)))) for value in trail_x]
        draw_ball(frame, world_to_screen(x, max(-0.15, state["y"])), frame_index * 22, trail)
        draw_velocity_arrow(frame, state)

        panel_draw = glass_panel(frame, (54, 698, 1546, 842))
        panel_draw.text((82, 720), phase, font=FONT_TITLE, fill=GREEN if phase == "TARGET HIT" else GOLD)
        panel_draw.text((82, 763), insight, font=FONT_BODY, fill=WHITE)
        value_cell(panel_draw, 612, 718, "time", f"{state['t']:.3f} s")
        value_cell(panel_draw, 803, 718, "position", f"{state['x']:05.2f} m / {state['y']:+05.2f} m", CYAN)
        slope = state["vy"] / state["vx"]
        value_cell(panel_draw, 1094, 718, "speed / slope", f"{state['speed']:05.2f} m/s / {slope:+.2f}")
        net_error = float(height(NET_X, theta)) - TARGET_H
        value_cell(panel_draw, 1323, 718, "final error", f"{net_error:+.3f} m", GREEN)

        # Constraint checkpoints remain dim until the ball reaches them, then lock on in color.
        checkpoints = ImageDraw.Draw(frame)
        checkpoint_values = [
            ("WALL", x >= WALL_X, float(height(WALL_X, theta)) - WALL_H - BALL_RADIUS),
            ("GOAL", x >= GOAL_X, float(height(GOAL_X, theta))),
            ("TARGET", x >= NET_X, net_error),
        ]
        for index, (label, reached, value) in enumerate(checkpoint_values):
            top = 160 + index * 58
            color = GREEN if reached else (111, 132, 138)
            checkpoints.rounded_rectangle((1270, top, 1538, top + 43), radius=18,
                                           fill=NAVY + (190,), outline=color + (180,), width=2)
            checkpoints.text((1290, top + 11), label, font=FONT_SMALL, fill=color)
            display = f"{value:+.3f}m" if label != "GOAL" else f"{value:.3f}m"
            checkpoints.text((1518, top + 21), display, anchor="rm", font=FONT_MONO, fill=WHITE)
        frames.append(frame)
    save_frames(frames, IMAGE_DIR / "free-kick-forward-telemetry.gif", duration=70)


def attempt_message(state: LossState) -> tuple[str, str, tuple[int, int, int]]:
    if state.goal_high_gap > 0:
        return "TOO HIGH", "Reduce the launch angle", RED
    if state.wall_gap > 0:
        return "WALL HIT", "Increase the launch angle", RED
    if state.net_error > 0.08:
        return "HIGH IN THE NET", "Reduce the angle a little", AMBER
    if state.net_error < -0.08:
        return "LOW IN THE NET", "Increase the angle a little", AMBER
    return "TARGET LOCKED", "The correction has almost vanished", GREEN


def render_descent(history: list[LossState]) -> None:
    selected = np.unique(np.round(np.linspace(0, len(history) - 1, 9)).astype(int))
    frames: list[Image.Image] = []
    frames_per_attempt = 10
    main_frames = len(selected) * frames_per_attempt
    for frame_index in range(main_frames + 8):
        frame = BASE_PLATE.copy()
        draw_header(frame, "Learn by missing", "EVERY MISS CONTAINS A DIRECTION", "02 / CORRECT IT")
        slot = min(len(selected) - 1, frame_index // frames_per_attempt)
        history_index = int(selected[slot])
        state = history[history_index]
        if frame_index >= main_frames:
            local_progress = 1.0
        else:
            local_progress = smoothstep(min(1.0, (frame_index % frames_per_attempt) / 5.0))
        x = NET_X * local_progress

        # Earlier attempts remain as restrained light traces: the learner sees memory accumulating.
        ghost = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        ghost_draw = ImageDraw.Draw(ghost)
        for old_index in selected[:slot]:
            old_points = trajectory_points(history[int(old_index)].theta)
            ghost_draw.line(old_points, fill=(155, 190, 194, 42), width=2)
        frame.alpha_composite(ghost)

        trail_x = np.linspace(max(0.0, x - 4.0), x, 30)
        trail = [world_to_screen(float(value), max(-0.15, float(height(value, state.theta)))) for value in trail_x]
        draw_ball(frame, world_to_screen(x, max(-0.15, float(height(x, state.theta)))), frame_index * 24, trail)
        outcome, correction, outcome_color = attempt_message(state)

        draw = glass_panel(frame, (58, 147, 445, 408), alpha=198)
        draw.text((86, 175), f"ATTEMPT {history_index:02d}", font=FONT_LABEL, fill=MUTED)
        draw.text((86, 215), f"{state.theta:06.3f}°", font=FONT_MONO_BIG, fill=WHITE)
        draw.text((86, 272), outcome, font=FONT_TITLE, fill=outcome_color)
        draw.text((86, 315), correction, font=FONT_BODY, fill=WHITE)
        arrow = "↓" if state.gradient > 0.01 else "↑" if state.gradient < -0.01 else "·"
        draw.text((86, 360), f"NEXT  {arrow}  2.00 × {abs(state.gradient):.4f}", font=FONT_MONO, fill=CYAN)

        ticker = glass_panel(frame, (640, 728, 1540, 842), alpha=188)
        value_cell(ticker, 674, 749, "wall", f"{state.h_wall:05.2f} m", GREEN if state.wall_gap == 0 else RED)
        value_cell(ticker, 890, 749, "goal plane", f"{state.h_goal:05.2f} m", GREEN if state.goal_low_gap == state.goal_high_gap == 0 else RED)
        value_cell(ticker, 1140, 749, "back net", f"{state.h_net:05.2f} m", GOLD)
        value_cell(ticker, 1345, 749, "wrongness", f"{state.loss:07.4f}", outcome_color)
        frames.append(frame)
    save_frames(frames, IMAGE_DIR / "gradient-descent-free-kick.gif", duration=90)


def draw_chip(frame: Image.Image, position: tuple[int, int], label: str, value: str, active: bool) -> None:
    x, y = position
    overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    color = GREEN if active else (115, 137, 143)
    draw.rounded_rectangle((x, y, x + 266, y + 86), radius=18, fill=NAVY + ((225 if active else 165),),
                           outline=color + ((240 if active else 110),), width=3)
    draw.text((x + 18, y + 14), label.upper(), font=FONT_SMALL, fill=MUTED)
    draw.text((x + 18, y + 40), value if active else "waiting...", font=FONT_MONO,
              fill=WHITE if active else MUTED)
    if active:
        halo = overlay.filter(ImageFilter.GaussianBlur(12))
        frame.alpha_composite(halo)
    frame.alpha_composite(overlay)


def render_backprop(theta: float) -> None:
    state = describe(theta)
    frames: list[Image.Image] = []
    total = 78
    forward_end = 34
    impact_end = 45
    path = trajectory_points(theta)
    nodes = [
        ((1165, 595), "impact", f"{state.h_net:.3f} m"),
        ((872, 245), "miss", f"{state.net_error:+.3f} m"),
        ((546, 182), "wrongness", f"{state.loss:.5f}"),
        ((174, 228), "angle responsibility", f"{state.gradient:+.5f}"),
    ]

    for frame_index in range(total):
        frame = ImageEnhance.Brightness(BASE_PLATE).enhance(0.82)
        draw_header(frame, "Reverse pass", "START AT THE IMPACT. TRACE THE CAUSE.", "03 / TRACE IT")
        forward_progress = smoothstep(min(1.0, frame_index / forward_end))
        x = NET_X * forward_progress
        trail_count = max(2, int(len(path) * forward_progress))
        draw_ball(frame, world_to_screen(x, max(-0.15, float(height(x, theta)))), frame_index * 20, path[:trail_count])

        reverse_progress = smoothstep(max(0.0, (frame_index - impact_end) / (total - impact_end - 1)))
        active_nodes = 0
        if frame_index > forward_end:
            active_nodes = 1 + int(reverse_progress * (len(nodes) - 1) + 0.001)
            for node_index, (position, label, value) in enumerate(nodes):
                draw_chip(frame, position, label, value, node_index < active_nodes)

        if reverse_progress > 0:
            pulse = Image.new("RGBA", frame.size, (0, 0, 0, 0))
            pulse_draw = ImageDraw.Draw(pulse)
            reverse_path = list(reversed(path))
            count = max(2, int(len(reverse_path) * reverse_progress))
            pulse_draw.line(reverse_path[:count], fill=GREEN + (215,), width=8, joint="curve")
            pulse_draw.line(reverse_path[:count], fill=WHITE + (220,), width=2, joint="curve")
            frame.alpha_composite(pulse.filter(ImageFilter.GaussianBlur(5)))
            frame.alpha_composite(pulse)

        draw = glass_panel(frame, (58, 700, 1542, 842), alpha=205)
        if frame_index <= forward_end:
            headline = "WHAT HAPPENED?"
            sentence = "The chosen angle produced this exact flight and impact."
            color = GOLD
        elif frame_index <= impact_end:
            headline = "HOW WRONG WAS IT?"
            sentence = f"The ball arrived {state.net_error:+.3f} m above its target."
            color = RED
        else:
            headline = "WHICH WAY SHOULD THE CAUSE MOVE?"
            sentence = "A steeper angle raises the impact. The next update lowers the angle."
            color = GREEN
        draw.text((86, 721), headline, font=FONT_TITLE, fill=color)
        draw.text((86, 767), sentence, font=FONT_BODY, fill=WHITE)
        if active_nodes == len(nodes):
            draw.text((1040, 754), "0.5e × impact sensitivity", font=FONT_SMALL, fill=MUTED)
            draw.text((1040, 782), f"= {state.gradient:+.5f} loss / degree", font=FONT_MONO, fill=GREEN)
        elif frame_index > forward_end:
            draw.text((1040, 754), "ACTIVE BRANCH", font=FONT_SMALL, fill=MUTED)
            draw.text((1040, 782), "TARGET  (wall + goal satisfied)", font=FONT_MONO, fill=CYAN)
        frames.append(frame)
    save_frames(frames, IMAGE_DIR / "backprop-free-kick.gif", duration=75)


def render_activation_gates() -> None:
    """Show three distinct ways a model can constrain or gate a signal."""
    frames: list[Image.Image] = []
    total = 84
    cards = [
        (54, "HARD CLIP", "A rigid boundary", AMBER),
        (548, "RELU GATE", "Negative blocked; positive passes", CYAN),
        (1042, "SIGMOID", "Everything compressed to 0–1", GREEN),
    ]

    for frame_index in range(total):
        frame = ImageEnhance.Brightness(BASE_PLATE.filter(ImageFilter.GaussianBlur(5))).enhance(0.64)
        draw_header(frame, "Constraint gates", "THREE WAYS TO CONTROL A SIGNAL", "04 / BOUND IT")

        sweep = smoothstep(min(1.0, frame_index / 70))
        raw_value = -3.0 + 6.0 * sweep
        outputs = [
            float(np.clip(raw_value, 0.0, 1.0)),
            max(0.0, raw_value),
            1.0 / (1.0 + math.exp(-raw_value)),
        ]
        gradients = [
            1.0 if 0.0 < raw_value < 1.0 else 0.0,
            1.0 if raw_value > 0.0 else 0.0,
            outputs[2] * (1.0 - outputs[2]),
        ]
        explanations = [
            "Outside the legal range, the value sticks to a wall.",
            "The gate sleeps for negative evidence and wakes for positive evidence.",
            "No hard wall: the output approaches the limits smoothly.",
        ]

        for card_index, (x, title, subtitle, accent) in enumerate(cards):
            panel = Image.new("RGBA", frame.size, (0, 0, 0, 0))
            panel_draw = ImageDraw.Draw(panel)
            panel_draw.rounded_rectangle(
                (x, 158, x + 454, 706),
                radius=28,
                fill=NAVY + (214,),
                outline=accent + (185,),
                width=3,
            )
            panel_draw.text((x + 30, 188), title, font=FONT_TITLE, fill=accent)
            panel_draw.text((x + 30, 232), subtitle, font=FONT_SMALL, fill=MUTED)

            panel_draw.text((x + 30, 296), "RAW SIGNAL", font=FONT_SMALL, fill=MUTED)
            panel_draw.text((x + 30, 323), f"z = {raw_value:+.2f}", font=FONT_MONO_BIG, fill=WHITE)

            # Signal enters from the left and meets the gate in the centre.
            rail_y = 430
            panel_draw.line((x + 38, rail_y, x + 414, rail_y), fill=(116, 139, 145, 170), width=5)
            gate_x = x + 226
            panel_draw.line((gate_x, rail_y - 62, gate_x, rail_y + 62), fill=accent + (220,), width=8)
            input_x = int(x + 50 + sweep * 152)
            output_scale = outputs[card_index]
            if card_index == 1:
                output_scale = min(1.0, output_scale / 3.0)
            output_x = int(gate_x + 22 + output_scale * 160)
            panel_draw.ellipse((input_x - 13, rail_y - 13, input_x + 13, rail_y + 13), fill=GOLD + (255,))
            panel_draw.ellipse((output_x - 15, rail_y - 15, output_x + 15, rail_y + 15),
                               fill=accent + (255,), outline=WHITE + (210,), width=2)
            panel_draw.text((x + 30, 490), "OUTPUT", font=FONT_SMALL, fill=MUTED)
            panel_draw.text((x + 30, 518), f"{outputs[card_index]:.3f}", font=FONT_MONO_BIG, fill=accent)
            panel_draw.text((x + 252, 490), "LOCAL GRADIENT", font=FONT_SMALL, fill=MUTED)
            panel_draw.text((x + 252, 523), f"{gradients[card_index]:.3f}", font=FONT_MONO, fill=WHITE)
            wrapped = "\n".join(textwrap.wrap(explanations[card_index], width=41))
            panel_draw.multiline_text((x + 30, 592), wrapped, font=FONT_SMALL, fill=WHITE, spacing=5)
            frame.alpha_composite(panel)

        footer = glass_panel(frame, (54, 738, 1496, 844), alpha=205)
        if raw_value < 0:
            footer_title = "NEGATIVE INPUT"
            footer_text = "Clip and ReLU both stop it. Sigmoid still returns a small, non-zero signal."
        elif raw_value <= 1:
            footer_title = "INSIDE THE ACTIVE REGION"
            footer_text = "Clip and ReLU pass the signal with full local sensitivity; sigmoid changes smoothly."
        else:
            footer_title = "LARGE POSITIVE INPUT"
            footer_text = "Clip saturates at 1, ReLU keeps growing, and sigmoid approaches 1 gradually."
        footer.text((82, 758), footer_title, font=FONT_LABEL, fill=GOLD)
        footer.text((82, 792), footer_text, font=FONT_BODY, fill=WHITE)
        frames.append(frame)

    save_frames(frames, IMAGE_DIR / "activation-functions-gates.gif", duration=80)


def main() -> None:
    history = optimize()
    verify_model(history)
    render_forward(history[-1].theta)
    render_descent(history)
    render_backprop(history[-1].theta + 0.65)
    render_activation_gates()


if __name__ == "__main__":
    main()
