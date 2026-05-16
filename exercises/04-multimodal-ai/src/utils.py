"""
Utility functions for PixelSmith
"""

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
from PIL import Image
import numpy as np
import soundfile as sf


def setup_logger(name: str = "pixelsmith", level: str = "INFO") -> logging.Logger:
    """
    Set up logger with consistent formatting.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def ensure_dir(path: str) -> Path:
    """
    Ensure directory exists, create if not.
    
    Args:
        path: Directory path
    
    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def validate_image_file(file_path: str, max_size_mb: int = 10) -> bool:
    """
    Validate image file format and size.
    
    Args:
        file_path: Path to image file
        max_size_mb: Maximum file size in MB
    
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > max_size_mb:
            return False
        
        # Try to open with PIL
        with Image.open(file_path) as img:
            img.verify()
        
        return True
    except Exception:
        return False


def resize_image(image: Image.Image, max_width: int = 512, max_height: int = 512) -> Image.Image:
    """
    Resize image maintaining aspect ratio.
    
    Args:
        image: PIL Image object
        max_width: Maximum width
        max_height: Maximum height
    
    Returns:
        Resized image
    """
    # Calculate new dimensions
    width, height = image.size
    aspect = width / height
    
    if width > max_width or height > max_height:
        if aspect > 1:
            new_width = max_width
            new_height = int(max_width / aspect)
        else:
            new_height = max_height
            new_width = int(max_height * aspect)
        
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return image


def validate_audio_file(file_path: str, max_duration_sec: int = 30, max_size_mb: int = 50) -> bool:
    """
    Validate audio file format, duration, and size.
    
    Args:
        file_path: Path to audio file
        max_duration_sec: Maximum duration in seconds
        max_size_mb: Maximum file size in MB
    
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > max_size_mb:
            return False
        
        # Check duration
        info = sf.info(file_path)
        if info.duration > max_duration_sec:
            return False
        
        return True
    except Exception:
        return False


def load_audio(file_path: str, target_sr: int = 16000) -> np.ndarray:
    """
    Load audio file and resample to target sample rate.
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate
    
    Returns:
        Audio waveform as numpy array
    """
    import librosa
    audio, sr = librosa.load(file_path, sr=target_sr)
    return audio


def safe_filename(filename: str) -> str:
    """
    Generate safe filename by removing special characters.
    
    Args:
        filename: Original filename
    
    Returns:
        Safe filename
    """
    import re
    # Remove special characters
    safe_name = re.sub(r'[^\w\s.-]', '', filename)
    # Replace spaces with underscores
    safe_name = safe_name.replace(' ', '_')
    return safe_name
