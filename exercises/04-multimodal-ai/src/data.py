"""
Data loading and preprocessing for multimodal inputs
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import librosa
from transformers import CLIPProcessor, WhisperProcessor, BlipProcessor


class MultimodalDataset(Dataset):
    """
    Dataset for multimodal inputs (text, image, audio).
    """
    
    def __init__(
        self,
        data: List[Dict],
        modalities: List[str] = ["text", "image", "audio"],
        processors: Optional[Dict] = None
    ):
        """
        Args:
            data: List of dictionaries containing modality data
            modalities: List of modalities to include
            processors: Dictionary of preprocessors for each modality
        """
        self.data = data
        self.modalities = modalities
        self.processors = processors or {}
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        processed = {}
        
        for modality in self.modalities:
            if modality in item and modality in self.processors:
                processed[modality] = self.processors[modality](item[modality])
            elif modality in item:
                processed[modality] = item[modality]
        
        return processed


class ImagePreprocessor:
    """
    Preprocess images for multimodal models.
    """
    
    def __init__(self, processor: Optional[CLIPProcessor] = None, target_size: int = 224):
        """
        Args:
            processor: CLIP processor
            target_size: Target image size
        """
        self.processor = processor
        self.target_size = target_size
    
    def __call__(self, image: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Preprocess image.
        
        Args:
            image: Image as path, PIL Image, or numpy array
        
        Returns:
            Preprocessed image tensor
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Use processor if available
        if self.processor is not None:
            return self.processor(images=image, return_tensors="pt")
        
        # Default preprocessing
        if image.size != (self.target_size, self.target_size):
            image = image.resize((self.target_size, self.target_size), Image.Resampling.LANCZOS)
        
        # Convert to tensor and normalize
        image_array = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        
        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        
        return image_tensor


class AudioPreprocessor:
    """
    Preprocess audio for multimodal models.
    """
    
    def __init__(
        self,
        processor: Optional[WhisperProcessor] = None,
        sample_rate: int = 16000,
        max_duration: float = 30.0
    ):
        """
        Args:
            processor: Whisper processor
            sample_rate: Target sample rate
            max_duration: Maximum audio duration in seconds
        """
        self.processor = processor
        self.sample_rate = sample_rate
        self.max_duration = max_duration
    
    def __call__(self, audio: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Preprocess audio.
        
        Args:
            audio: Audio as path or waveform
        
        Returns:
            Preprocessed audio tensor
        """
        # Load audio if path
        if isinstance(audio, (str, Path)):
            audio, sr = librosa.load(audio, sr=self.sample_rate)
        elif isinstance(audio, torch.Tensor):
            audio = audio.numpy()
        
        # Trim to max duration
        max_samples = int(self.max_duration * self.sample_rate)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        
        # Use processor if available
        if self.processor is not None:
            return self.processor(audio, sampling_rate=self.sample_rate, return_tensors="pt")
        
        # Default: return as tensor
        return torch.from_numpy(audio).float()


class TextPreprocessor:
    """
    Preprocess text for multimodal models.
    """
    
    def __init__(self, processor: Optional[CLIPProcessor] = None, max_length: int = 77):
        """
        Args:
            processor: CLIP processor
            max_length: Maximum text length
        """
        self.processor = processor
        self.max_length = max_length
    
    def __call__(self, text: str) -> Dict:
        """
        Preprocess text.
        
        Args:
            text: Input text
        
        Returns:
            Preprocessed text tokens
        """
        if self.processor is not None:
            return self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
        
        # Simple tokenization fallback
        return {"text": text}


def create_multimodal_dataloader(
    data: List[Dict],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    processors: Optional[Dict] = None
) -> DataLoader:
    """
    Create DataLoader for multimodal data.
    
    Args:
        data: List of data dictionaries
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        processors: Dictionary of preprocessors
    
    Returns:
        DataLoader
    """
    dataset = MultimodalDataset(data=data, processors=processors)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader
