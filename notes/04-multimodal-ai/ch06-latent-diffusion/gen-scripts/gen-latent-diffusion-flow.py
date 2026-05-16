from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.flow_animation import FlowStage, render_flow_animation

if __name__ == "__main__":
    chapter_dir = Path(__file__).resolve().parents[1]
    gif_path, png_path = render_flow_animation(
        chapter_title="Latent Diffusion - Data Flow",
        flow_caption="pixels -> VAE latent -> denoise loop -> decode",
        stages=[
            FlowStage("Pixels", "512x512x3"),
            FlowStage("VAE Encode", "compress"),
            FlowStage("Latent z", "64x64x4"),
            FlowStage("UNet Denoise", "latent steps"),
            FlowStage("VAE Decode", "image output"),
        ],
        output_stem="latent-diffusion-flow",
        chapter_dir=chapter_dir,
    )
    print(f"Saved {gif_path}")
    print(f"Saved {png_path}")

