import torch
import torchaudio
from pathlib import Path
from speechbrain.pretrained import SpectralMaskEnhancement
import warnings
warnings.filterwarnings('ignore')


class AudioProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Audio processor device: {self.device}")

        self.model_config = {
            'gpu': {
                'name': 'speechbrain/metricgan-plus-voicebank',
                'loaded': False,
                'model': None
            },
            'cpu': {
                'name': 'speechbrain/metricgan-plus-voicebank',
                'loaded': False,
                'model': None
            }
        }

    def load_model(self):
        mode = 'gpu' if self.device == 'cuda' else 'cpu'
        config = self.model_config[mode]

        if config['loaded']:
            return

        print(f"Loading {mode.upper()} audio model: {config['name']}")

        try:
            config['model'] = SpectralMaskEnhancement.from_hparams(
                source=config['name'],
                savedir="pretrained_models/metricgan-plus",
                run_opts={"device": self.device}
            )
            config['loaded'] = True
            print(f"{mode.upper()} audio model loaded")
        except Exception as e:
            print(f"Error loading audio model: {e}")
            raise

    def enhance_audio(self, input_path: str, output_path: str):
        if not self.model_config['gpu']['loaded'] and not self.model_config['cpu']['loaded']:
            self.load_model()

        mode = 'gpu' if self.device == 'cuda' else 'cpu'
        config = self.model_config[mode]

        try:
            print(f"Enhancing audio: {input_path}")

            enhanced = config['model'].enhance_file(input_path)

            torchaudio.save(output_path, enhanced.unsqueeze(0).cpu(), 16000)

            print(f"Audio enhanced: {output_path}")
            return output_path
        except Exception as e:
            print(f"Audio enhancement error: {e}")
            import shutil
            shutil.copy(input_path, output_path)
            print(f"Copied original audio to: {output_path}")
            return output_path


_audio_processor = None

def get_audio_processor():
    global _audio_processor
    if _audio_processor is None:
        _audio_processor = AudioProcessor()
    return _audio_processor
