from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.flow_animation import FlowStage, render_flow_animation

if __name__ == "__main__":
    chapter_dir = Path(__file__).resolve().parents[1]
    gif_path, png_path = render_flow_animation(
        chapter_title="Text-to-Video - Data Flow",
        flow_caption="prompt + temporal blocks -> latent frames -> decode",
        stages=[
            FlowStage("Prompt", "text condition"),
            FlowStage("Temporal Blocks", "space-time attention"),
            FlowStage("Latent Frames", "multi-frame tensor"),
            FlowStage("Video Decoder", "frame synthesis"),
            FlowStage("Clip", "final video"),
        ],
        output_stem="text-to-video-flow",
        chapter_dir=chapter_dir,
    )
    print(f"Saved {gif_path}")
    print(f"Saved {png_path}")

