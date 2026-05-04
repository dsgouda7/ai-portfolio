from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.flow_animation import FlowStage, render_flow_animation

if __name__ == "__main__":
    chapter_dir = Path(__file__).resolve().parents[1]
    gif_path, png_path = render_flow_animation(
        chapter_title="Generative Evaluation - Data Flow",
        flow_caption="generated + reference sets -> metrics -> preference signal",
        stages=[
            FlowStage("Generated Set", "model outputs"),
            FlowStage("Reference Set", "real samples"),
            FlowStage("Metrics", "FID/CLIP/HPS"),
            FlowStage("Aggregate", "quality profile"),
            FlowStage("Decision", "ship or tune"),
        ],
        output_stem="generative-evaluation-flow",
        chapter_dir=chapter_dir,
    )
    print(f"Saved {gif_path}")
    print(f"Saved {png_path}")

