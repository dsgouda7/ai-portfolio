from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.flow_animation import FlowStage, render_flow_animation

if __name__ == "__main__":
    chapter_dir = Path(__file__).resolve().parents[1]
    gif_path, png_path = render_flow_animation(
        chapter_title="Guidance Conditioning - Data Flow",
        flow_caption="prompt + null branch -> CFG combine -> guided denoise",
        stages=[
            FlowStage("Prompt", "text tokens"),
            FlowStage("Cond UNet", "eps_cond"),
            FlowStage("Uncond UNet", "eps_uncond"),
            FlowStage("CFG Mix", "eps_u + w*(eps_c-eps_u)"),
            FlowStage("Denoise", "prompt-aligned step"),
        ],
        output_stem="guidance-conditioning-flow",
        chapter_dir=chapter_dir,
    )
    print(f"Saved {gif_path}")
    print(f"Saved {png_path}")

