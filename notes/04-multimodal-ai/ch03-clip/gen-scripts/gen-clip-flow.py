from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.flow_animation import FlowStage, render_flow_animation

if __name__ == "__main__":
    chapter_dir = Path(__file__).resolve().parents[1]
    gif_path, png_path = render_flow_animation(
        chapter_title="CLIP - Data Flow",
        flow_caption="image/text encoders -> projection -> cosine similarity",
        stages=[
            FlowStage("Image", "visual input"),
            FlowStage("ViT Encoder", "image features"),
            FlowStage("Text Encoder", "text features"),
            FlowStage("Project", "shared 512-dim"),
            FlowStage("Similarity", "contrastive score"),
        ],
        output_stem="clip-flow",
        chapter_dir=chapter_dir,
    )
    print(f"Saved {gif_path}")
    print(f"Saved {png_path}")

