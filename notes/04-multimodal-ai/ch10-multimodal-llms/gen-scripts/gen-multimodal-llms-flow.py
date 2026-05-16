from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.flow_animation import FlowStage, render_flow_animation

if __name__ == "__main__":
    chapter_dir = Path(__file__).resolve().parents[1]
    gif_path, png_path = render_flow_animation(
        chapter_title="Multimodal LLMs - Data Flow",
        flow_caption="image encoder -> projector/Q-former -> LLM tokens -> answer",
        stages=[
            FlowStage("Image", "visual prompt"),
            FlowStage("Vision Encoder", "patch features"),
            FlowStage("Adapter", "Q-former/projector"),
            FlowStage("LLM Decode", "token reasoning"),
            FlowStage("Answer", "text output"),
        ],
        output_stem="multimodal-llms-flow",
        chapter_dir=chapter_dir,
    )
    print(f"Saved {gif_path}")
    print(f"Saved {png_path}")

