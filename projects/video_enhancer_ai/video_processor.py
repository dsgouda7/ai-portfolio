import torch
import cv2
import numpy as np
from pathlib import Path
from transformers import Swin2SRForImageSuperResolution, AutoImageProcessor
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


class VideoProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Video processor device: {self.device}")

        self.model_config = {
            'gpu': {
                'name': 'caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr',
                'loaded': False,
                'model': None,
                'processor': None
            },
            'cpu': {
                'name': 'caidas/swin2SR-classical-sr-x4-64',
                'loaded': False,
                'model': None,
                'processor': None
            }
        }

    def load_model(self):
        mode = 'gpu' if self.device == 'cuda' else 'cpu'
        config = self.model_config[mode]

        if config['loaded']:
            return

        print(f"Loading {mode.upper()} model: {config['name']}")

        try:
            config['model'] = Swin2SRForImageSuperResolution.from_pretrained(
                config['name']
            ).to(self.device)
            config['processor'] = AutoImageProcessor.from_pretrained(
                config['name']
            )
            config['loaded'] = True
            print(f"{mode.upper()} model loaded")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def upscale_frame(self, frame: np.ndarray) -> np.ndarray:
        if not self.model_config['gpu']['loaded'] and not self.model_config['cpu']['loaded']:
            self.load_model()

        mode = 'gpu' if self.device == 'cuda' else 'cpu'
        config = self.model_config[mode]

        try:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = config['processor'](img, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = config['model'](**inputs)

            output = outputs.reconstruction.squeeze().permute(1, 2, 0).cpu().numpy()
            output = (output * 255).clip(0, 255).astype(np.uint8)

            return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Frame upscaling error: {e}")
            return frame

    def process_video(self, input_path: str, output_path: str, target_resolution=(3840, 2160)):
        print(f"Processing video: {input_path}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Input: {width}x{height}, {fps} FPS, {total_frames} frames")
        print(f"Target: {target_resolution[0]}x{target_resolution[1]}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, target_resolution)

        processed = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            upscaled = self.upscale_frame(frame)

            if upscaled.shape[:2][::-1] != target_resolution:
                upscaled = cv2.resize(upscaled, target_resolution, interpolation=cv2.INTER_LANCZOS4)

            out.write(upscaled)
            processed += 1

            if processed % 30 == 0:
                progress = (processed / total_frames) * 100
                print(f"Progress: {processed}/{total_frames} ({progress:.1f}%)", end='\r')

        cap.release()
        out.release()
        print(f"\nCompleted: {output_path}")
        return output_path


_video_processor = None

def get_video_processor():
    global _video_processor
    if _video_processor is None:
        _video_processor = VideoProcessor()
    return _video_processor
