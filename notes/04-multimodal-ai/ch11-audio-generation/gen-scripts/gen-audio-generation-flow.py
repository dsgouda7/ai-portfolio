from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.flow_animation import FlowStage, render_flow_animation

if __name__ == "__main__":
    chapter_dir = Path(__file__).resolve().parents[1]
    gif_path, png_path = render_flow_animation(
        chapter_title="Audio Generation - Data Flow",
        flow_caption="text -> tokenizer -> acoustic model -> waveform",
        stages=[
            FlowStage("Text", "prompt/caption"),
            FlowStage("Tokenizer", "phoneme/token ids"),
            FlowStage("Acoustic Model", "mel features"),
            FlowStage("Vocoder", "waveform synthesis"),
            FlowStage("Audio", "playable output"),
        ],
        output_stem="audio-generation-flow",
        chapter_dir=chapter_dir,
    )
    print(f"Saved {gif_path}")
    print(f"Saved {png_path}")

