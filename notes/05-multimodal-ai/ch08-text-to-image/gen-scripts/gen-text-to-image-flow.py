from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.flow_animation import FlowStage, render_flow_animation

if __name__ == "__main__":
    chapter_dir = Path(__file__).resolve().parents[1]
    gif_path, png_path = render_flow_animation(
        chapter_title="Text-to-Image - Data Flow",
        flow_caption="prompt -> text encoder -> denoise loop -> decode",
        stages=[
            FlowStage("Prompt", "natural language"),
            FlowStage("Tokenizer", "token ids"),
            FlowStage("Text Encoder", "context vectors"),
            FlowStage("UNet Loop", "guided denoise"),
            FlowStage("Decode", "image render"),
        ],
        output_stem="text-to-image-flow",
        chapter_dir=chapter_dir,
    )
    print(f"Saved {gif_path}")
    print(f"Saved {png_path}")

