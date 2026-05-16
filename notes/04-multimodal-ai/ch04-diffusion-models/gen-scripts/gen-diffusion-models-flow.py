from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.flow_animation import FlowStage, render_flow_animation

if __name__ == "__main__":
    chapter_dir = Path(__file__).resolve().parents[1]
    gif_path, png_path = render_flow_animation(
        chapter_title="Diffusion Models - Data Flow",
        flow_caption="x0 -> forward noise chain -> reverse denoise -> sampled image",
        stages=[
            FlowStage("Input x0", "clean image"),
            FlowStage("Forward q", "add noise"),
            FlowStage("xT", "pure Gaussian"),
            FlowStage("Reverse p_theta", "iterative denoise"),
            FlowStage("Sample x0_hat", "generated output"),
        ],
        output_stem="diffusion-models-flow",
        chapter_dir=chapter_dir,
    )
    print(f"Saved {gif_path}")
    print(f"Saved {png_path}")

