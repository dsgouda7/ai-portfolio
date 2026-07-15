import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, pipeline
import warnings
warnings.filterwarnings('ignore')


class ModelManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"{self.device}")

        self.models_config = {
            # speech to text model
            'stt': {
                'name': 'openai/whisper-tiny.en',
                'type': 'automatic-speech-recognition',
                'loaded': False,
                'pipeline': None
            },
            # llm to process raw text
            'llm': {
                'name': 'microsoft/DialoGPT-small',
                'type': 'conversational',
                'loaded': False,
                'tokenizer': None,
                'model': None
            },
            # text to speech model
            'tts': {
                'name': 'microsoft/speecht5_tts',
                'type': 'text-to-speech',
                'loaded': False,
                'pipeline': None,
                'processor': None,
                'vocoder_name': 'microsoft/speecht5_hifigan'
            }
        }

    def download_model(self, model_type):
        if model_type not in self.models_config:
            raise ValueError(f"Unknown model: {model_type}")

        config = self.models_config[model_type]
        print(f"{model_type.upper()}: {config['name']}")

        try:
            if model_type == 'stt':
                config['pipeline'] = pipeline(
                    config['type'],
                    model=config['name'],
                    device=0 if self.device == "cuda" else -1
                )
                config['loaded'] = True
                print(f"{model_type.upper()}")

            elif model_type == 'llm':
                config['tokenizer'] = AutoTokenizer.from_pretrained(config['name'])
                config['model'] = AutoModelForCausalLM.from_pretrained(
                    config['name'],
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                ).to(self.device)
                config['tokenizer'].pad_token = config['tokenizer'].eos_token
                config['loaded'] = True
                print(f"{model_type.upper()}")

            elif model_type == 'tts':
                config['processor'] = AutoProcessor.from_pretrained(config['name'])
                config['pipeline'] = pipeline(
                    "text-to-speech",
                    model=config['name'],
                    device=0 if self.device == "cuda" else -1
                )
                config['loaded'] = True
                print(f"{model_type.upper()}")

        except Exception as e:
            print(f"{model_type}: {e}")
            raise

    def download_all_models(self):
        print("Loading models...")
        for model_type in self.models_config.keys():
            self.download_model(model_type)
        print("Models loaded")

    def speech_to_text(self, audio_bytes):
        if not self.models_config['stt']['loaded']:
            self.download_model('stt')

        try:
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_file_path = tmp_file.name

            try:
                result = self.models_config['stt']['pipeline'](tmp_file_path)
                return result['text']
            finally:
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)
        except Exception as e:
            print(f"STT error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_text_response(self, input_text, max_length=100):
        if not self.models_config['llm']['loaded']:
            self.download_model('llm')

        config = self.models_config['llm']

        try:
            inputs = config['tokenizer'].encode(
                input_text + config['tokenizer'].eos_token,
                return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                outputs = config['model'].generate(
                    inputs,
                    max_length=max_length,
                    pad_token_id=config['tokenizer'].eos_token_id,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.7
                )

            response = config['tokenizer'].decode(
                outputs[0][inputs.shape[-1]:],
                skip_special_tokens=True
            )

            return response.strip() if response else "No response generated."

        except Exception as e:
            print(f"LLM: {e}")
            return "Error generating response."

    def text_to_speech(self, text):
        if not self.models_config['tts']['loaded']:
            self.download_model('tts')

        try:
            speech = self.models_config['tts']['pipeline'](text)
            return speech['audio'], speech['sampling_rate']
        except Exception as e:
            print(f"TTS: {e}")
            return None, None

    def get_model_status(self):
        status = {}
        for model_type, config in self.models_config.items():
            status[model_type] = {
                'name': config['name'],
                'loaded': config['loaded'],
                'device': self.device
            }
        return status


_model_manager = None

def get_model_manager():
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
