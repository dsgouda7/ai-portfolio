from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.flow_animation import FlowStage, render_flow_animation

if __name__ == "__main__":
    chapter_dir = Path(__file__).resolve().parents[1]
    gif_path, png_path = render_flow_animation(
        chapter_title="Local Diffusion Lab - Data Flow",
        flow_caption="train mini DDPM -> sample -> compare schedulers",
        stages=[
            FlowStage("Dataset", "local training set"),
            FlowStage("Train", "mini DDPM"),
            FlowStage("Sample", "noise to image"),
            FlowStage("Benchmark", "DDPM/DDIM/DPM"),
            FlowStage("Deploy", "optimized local run"),
        ],
        output_stem="local-diffusion-lab-flow",
        chapter_dir=chapter_dir,
    )
    print(f"Saved {gif_path}")
    print(f"Saved {png_path}")

