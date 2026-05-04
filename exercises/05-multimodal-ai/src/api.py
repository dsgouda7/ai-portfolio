"""
Flask API for PixelSmith Multimodal AI System
"""

import os
import io
import time
from pathlib import Path
from typing import Dict, Any
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

from .models.clip import CLIPModel
from .models.whisper import WhisperModel
from .models.image_gen import ImageGenerator
from .models.image_caption import ImageCaptioner
from .features import MultimodalFeatureExtractor
from .evaluate import compute_clip_score
from .monitoring import MetricsCollector, monitor_performance, start_metrics_server
from .utils import (
    setup_logger,
    load_config,
    ensure_dir,
    validate_image_file,
    validate_audio_file,
    resize_image,
    safe_filename
)


logger = setup_logger(__name__)


# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load configuration
config = load_config()

# Configure app
app.config['MAX_CONTENT_LENGTH'] = config['api']['max_content_length']
app.config['UPLOAD_FOLDER'] = config['api']['upload_folder']
ensure_dir(app.config['UPLOAD_FOLDER'])

# Initialize models (lazy loading)
models = {
    "clip": None,
    "whisper": None,
    "image_gen": None,
    "captioner": None,
    "feature_extractor": None
}

# Initialize metrics collector
metrics = MetricsCollector(
    mlflow_uri=config['monitoring']['mlflow_uri'],
    experiment_name=config['monitoring']['experiment_name']
)


def get_clip_model() -> CLIPModel:
    """Get or initialize CLIP model."""
    if models["clip"] is None:
        start = time.time()
        models["clip"] = CLIPModel(
            model_name=config['models']['clip']['model_name'],
            device=config['models']['clip']['device'],
            cache_dir=config['models']['clip']['cache_dir']
        )
        metrics.log_model_load("clip", time.time() - start)
    return models["clip"]


def get_whisper_model() -> WhisperModel:
    """Get or initialize Whisper model."""
    if models["whisper"] is None:
        start = time.time()
        models["whisper"] = WhisperModel(
            model_name=config['models']['whisper']['model_name'],
            device=config['models']['whisper']['device'],
            cache_dir=config['models']['whisper']['cache_dir'],
            language=config['models']['whisper']['language']
        )
        metrics.log_model_load("whisper", time.time() - start)
    return models["whisper"]


def get_image_generator() -> ImageGenerator:
    """Get or initialize Image Generator."""
    if models["image_gen"] is None:
        start = time.time()
        models["image_gen"] = ImageGenerator(
            model_name=config['models']['stable_diffusion']['model_name'],
            device=config['models']['stable_diffusion']['device'],
            cache_dir=config['models']['stable_diffusion']['cache_dir']
        )
        metrics.log_model_load("stable_diffusion", time.time() - start)
    return models["image_gen"]


def get_captioner() -> ImageCaptioner:
    """Get or initialize Image Captioner."""
    if models["captioner"] is None:
        start = time.time()
        models["captioner"] = ImageCaptioner(
            model_name=config['models']['blip']['model_name'],
            device=config['models']['blip']['device'],
            cache_dir=config['models']['blip']['cache_dir']
        )
        metrics.log_model_load("blip", time.time() - start)
    return models["captioner"]


def get_feature_extractor() -> MultimodalFeatureExtractor:
    """Get or initialize Feature Extractor."""
    if models["feature_extractor"] is None:
        models["feature_extractor"] = MultimodalFeatureExtractor(
            clip_model=get_clip_model(),
            whisper_model=get_whisper_model()
        )
    return models["feature_extractor"]


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "pixelsmith"})


@app.route('/similarity', methods=['POST'])
@monitor_performance(endpoint="similarity", modality="text_image")
def compute_similarity():
    """
    Compute text-image similarity using CLIP.
    
    Expected input:
    - text: string
    - image: file upload
    
    Returns:
    - similarity: float score between 0 and 1
    """
    try:
        # Validate inputs
        if 'text' not in request.form:
            return jsonify({"error": "Missing 'text' field"}), 400
        
        if 'image' not in request.files:
            return jsonify({"error": "Missing 'image' file"}), 400
        
        text = request.form['text']
        image_file = request.files['image']
        
        # Save and validate image
        filename = secure_filename(image_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename(filename))
        image_file.save(filepath)
        
        if not validate_image_file(filepath):
            return jsonify({"error": "Invalid image file"}), 400
        
        # Load and process image
        image = Image.open(filepath).convert("RGB")
        image = resize_image(
            image,
            config['image']['max_width'],
            config['image']['max_height']
        )
        
        # Compute similarity
        clip_model = get_clip_model()
        similarity = clip_model.compute_similarity(text, image)
        
        # Clean up
        os.remove(filepath)
        
        return jsonify({
            "similarity": float(similarity),
            "text": text,
            "threshold": config['thresholds']['clip_score_min']
        })
    
    except Exception as e:
        logger.error(f"Error in similarity endpoint: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/transcribe', methods=['POST'])
@monitor_performance(endpoint="transcribe", modality="audio")
def transcribe_audio():
    """
    Transcribe audio to text using Whisper.
    
    Expected input:
    - audio: file upload
    
    Returns:
    - text: transcribed text
    - duration: audio duration
    - language: detected language
    """
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "Missing 'audio' file"}), 400
        
        audio_file = request.files['audio']
        
        # Save and validate audio
        filename = secure_filename(audio_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename(filename))
        audio_file.save(filepath)
        
        if not validate_audio_file(
            filepath,
            config['audio']['max_duration_sec'],
            config['audio']['max_file_size_mb']
        ):
            return jsonify({"error": "Invalid audio file"}), 400
        
        # Transcribe
        whisper_model = get_whisper_model()
        result = whisper_model.transcribe(filepath)
        
        # Clean up
        os.remove(filepath)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in transcribe endpoint: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/generate', methods=['POST'])
@monitor_performance(endpoint="generate", modality="text_to_image")
def generate_image():
    """
    Generate image from text prompt using Stable Diffusion.
    
    Expected input (JSON):
    - prompt: string
    - negative_prompt: optional string
    - num_inference_steps: optional int
    - guidance_scale: optional float
    - width: optional int
    - height: optional int
    - seed: optional int
    
    Returns:
    - image: base64 encoded image or file
    """
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing 'prompt' field"}), 400
        
        prompt = data['prompt']
        negative_prompt = data.get('negative_prompt', None)
        num_steps = data.get('num_inference_steps', config['models']['stable_diffusion']['num_inference_steps'])
        guidance_scale = data.get('guidance_scale', config['models']['stable_diffusion']['guidance_scale'])
        width = data.get('width', config['image']['generation']['default_width'])
        height = data.get('height', config['image']['generation']['default_height'])
        seed = data.get('seed', None)
        
        # Generate image
        start_time = time.time()
        generator = get_image_generator()
        images = generator.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images=1,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            seed=seed
        )
        
        generation_time = time.time() - start_time
        
        # Compute CLIP score
        clip_model = get_clip_model()
        clip_score = clip_model.compute_similarity(prompt, images[0])
        
        # Log metrics
        metrics.log_generation(generation_time, clip_score)
        
        # Save image
        filename = f"generated_{int(time.time())}.png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        images[0].save(filepath)
        
        # Return image file
        return send_file(filepath, mimetype='image/png')
    
    except Exception as e:
        logger.error(f"Error in generate endpoint: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/caption', methods=['POST'])
@monitor_performance(endpoint="caption", modality="image_to_text")
def caption_image():
    """
    Generate caption for image using BLIP.
    
    Expected input:
    - image: file upload
    
    Returns:
    - caption: generated caption
    """
    try:
        if 'image' not in request.files:
            return jsonify({"error": "Missing 'image' file"}), 400
        
        image_file = request.files['image']
        
        # Save and validate image
        filename = secure_filename(image_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename(filename))
        image_file.save(filepath)
        
        if not validate_image_file(filepath):
            return jsonify({"error": "Invalid image file"}), 400
        
        # Load image
        image = Image.open(filepath).convert("RGB")
        
        # Generate caption
        captioner = get_captioner()
        caption = captioner.caption(image)
        
        # Clean up
        os.remove(filepath)
        
        return jsonify({"caption": caption})
    
    except Exception as e:
        logger.error(f"Error in caption endpoint: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/search', methods=['POST'])
@monitor_performance(endpoint="search", modality="multimodal")
def multimodal_search():
    """
    Multimodal search using text, image, or audio query.
    
    Expected input (multipart/form-data):
    - text: optional string
    - image: optional file
    - audio: optional file
    - database: JSON list of items to search
    
    Returns:
    - results: list of matching items with scores
    """
    try:
        # Parse query
        query = {}
        
        if 'text' in request.form:
            query['text'] = request.form['text']
        
        if 'image' in request.files:
            image_file = request.files['image']
            filename = secure_filename(image_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename(filename))
            image_file.save(filepath)
            query['image'] = Image.open(filepath).convert("RGB")
        
        if 'audio' in request.files:
            audio_file = request.files['audio']
            filename = secure_filename(audio_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename(filename))
            audio_file.save(filepath)
            query['audio'] = filepath
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        # For demo, return mock results
        # In production, would search actual database
        results = [
            {"id": 1, "score": 0.85, "type": "image"},
            {"id": 2, "score": 0.78, "type": "text"},
            {"id": 3, "score": 0.72, "type": "audio"}
        ]
        
        return jsonify({"results": results, "query_modalities": list(query.keys())})
    
    except Exception as e:
        logger.error(f"Error in search endpoint: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Start Prometheus metrics server
    start_metrics_server(config['monitoring']['prometheus_port'])
    
    # Run Flask app
    app.run(
        host=config['api']['host'],
        port=config['api']['port'],
        debug=config['api']['debug']
    )
