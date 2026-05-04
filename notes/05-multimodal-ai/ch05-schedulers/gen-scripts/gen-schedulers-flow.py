from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.flow_animation import FlowStage, render_flow_animation

if __name__ == "__main__":
    chapter_dir = Path(__file__).resolve().parents[1]
    gif_path, png_path = render_flow_animation(
        chapter_title="Schedulers - Data Flow",
        flow_caption="same UNet weights -> different sampler trajectories",
        stages=[
            FlowStage("xT", "noise start"),
            FlowStage("DDPM Path", "many short steps"),
            FlowStage("DDIM Path", "deterministic jumps"),
            FlowStage("DPM++ Path", "high-order solver"),
            FlowStage("x0_hat", "faster sample"),
        ],
        output_stem="schedulers-flow",
        chapter_dir=chapter_dir,
    )
    print(f"Saved {gif_path}")
    print(f"Saved {png_path}")

