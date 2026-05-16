from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.flow_animation import FlowStage, render_flow_animation

if __name__ == "__main__":
    chapter_dir = Path(__file__).resolve().parents[1]
    gif_path, png_path = render_flow_animation(
        chapter_title="Vision Transformers - Data Flow",
        flow_caption="image -> patches -> embeddings -> MHSA -> CLS head",
        stages=[
            FlowStage("Image", "RGB tensor"),
            FlowStage("Patchify", "16x16 tiles"),
            FlowStage("Embed", "patch vectors"),
            FlowStage("Transformer", "MHSA blocks"),
            FlowStage("CLS Output", "semantic embedding"),
        ],
        output_stem="vision-transformers-flow",
        chapter_dir=chapter_dir,
    )
    print(f"Saved {gif_path}")
    print(f"Saved {png_path}")

