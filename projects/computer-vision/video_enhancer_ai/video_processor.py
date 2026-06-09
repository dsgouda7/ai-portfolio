import torch
import cv2
import numpy as np
from pathlib import Path
from transformers import Swin2SRForImageSuperResolution, AutoImageProcessor
from PIL import Image
import warnings
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings('ignore')


class VideoProcessor:
    def __init__(self, config_path='config.yaml'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Video processor device: {self.device}")

        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.processing_config = config['processing']
        video_models = config['video_models']

        self.model_config = {
            'gpu': {
                'name': video_models['gpu']['name'],
                'description': video_models['gpu']['description'],
                'url': video_models['gpu']['huggingface_url'],
                'loaded': False,
                'model': None,
                'processor': None
            },
            'cpu': {
                'name': video_models['cpu']['name'],
                'description': video_models['cpu']['description'],
                'url': video_models['cpu']['huggingface_url'],
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

        try:
            config['model'] = Swin2SRForImageSuperResolution.from_pretrained(
                config['name']
            ).to(self.device)
            config['processor'] = AutoImageProcessor.from_pretrained(
                config['name']
            )
            config['loaded'] = True
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
            return frame

    def process_chunk(self, frames, chunk_id, total_chunks, target_resolution):
        """Process a chunk of frames in parallel"""
        processed_frames = []
        for i, frame in enumerate(frames):
            upscaled = self.upscale_frame(frame)
            if upscaled.shape[:2][::-1] != target_resolution:
                upscaled = cv2.resize(upscaled, target_resolution, interpolation=cv2.INTER_LANCZOS4)
            processed_frames.append(upscaled)
        return processed_frames

    def process_video(self, input_path: str, output_path: str, target_resolution=None):
        if target_resolution is None:
            target_resolution = tuple(self.processing_config['target_resolution'])

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Read all frames into memory (for parallel processing)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        # Split into chunks for parallel processing
        chunk_size = self.processing_config.get('chunk_size', 30)
        chunks = [frames[i:i + chunk_size] for i in range(0, len(frames), chunk_size)]

        # Process chunks in parallel
        processed_frames = []
        max_workers = self.processing_config.get('parallel_workers', 4)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for i, chunk in enumerate(chunks):
                future = executor.submit(self.process_chunk, chunk, i, len(chunks), target_resolution)
                futures[future] = i

            # Collect results in order
            chunk_results = {}
            for future in as_completed(futures):
                chunk_id = futures[future]
                chunk_results[chunk_id] = future.result()

            # Reconstruct frames in order
            for i in range(len(chunks)):
                processed_frames.extend(chunk_results[i])

        # Write output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, target_resolution)

        for frame in processed_frames:
            out.write(frame)

        out.release()
        return output_path

    def get_model_info(self):
        mode = 'gpu' if self.device == 'cuda' else 'cpu'
        config = self.model_config[mode]
        return {
            'name': config['name'],
            'description': config['description'],
            'url': config['url']
        }


_video_processor = None

def get_video_processor():
    global _video_processor
    if _video_processor is None:
        _video_processor = VideoProcessor()
    return _video_processor
