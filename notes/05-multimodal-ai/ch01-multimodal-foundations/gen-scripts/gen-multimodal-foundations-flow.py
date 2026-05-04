from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.flow_animation import FlowStage, render_flow_animation

if __name__ == "__main__":
    chapter_dir = Path(__file__).resolve().parents[1]
    gif_path, png_path = render_flow_animation(
        chapter_title="Multimodal Foundations - Data Flow",
        flow_caption="raw signal -> tensor -> tokens -> shared embedding",
        stages=[
            FlowStage("Raw Signal", "image/audio/video"),
            FlowStage("Tensorize", "numeric arrays"),
            FlowStage("Normalize", "stable scale"),
            FlowStage("Tokenize", "sequence form"),
            FlowStage("Shared Space", "aligned embedding"),
        ],
        output_stem="multimodal-foundations-flow",
        chapter_dir=chapter_dir,
    )
    print(f"Saved {gif_path}")
    print(f"Saved {png_path}")

