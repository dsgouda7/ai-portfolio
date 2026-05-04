"""Flask API for Object Detection

Provides REST API endpoints for image object detection inference.
"""

import os
import io
import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify
from PIL import Image
from typing import Dict, List
from pydantic import BaseModel, Field, validator
from src.utils import setup_logger, load_config
from src.models import ModelRegistry
from src.features import ImagePreprocessor
from src.monitoring import monitor_api_request, metrics_collector


# Initialize Flask app
app = Flask(__name__)

# Configuration
config = load_config()
logger = setup_logger(log_file=config['monitoring']['log_file'])

# Initialize model and preprocessor
model_registry = ModelRegistry(config)
preprocessor = ImagePreprocessor(config)

# Load model
model_path = os.path.join(config['paths']['model_dir'], 'compressed_model.pth')
if os.path.exists(model_path):
    model = model_registry.load_model(model_path)
    logger.info(f"Loaded model from {model_path}")
else:
    logger.warning(f"Model not found at {model_path}, creating baseline model")
    model = model_registry.create_model()

model.eval()
device = next(model.parameters()).device


class DetectionRequest(BaseModel):
    """Request schema for object detection."""
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_detections: int = Field(default=100, ge=1, le=500)
    
    @validator('confidence_threshold')
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('confidence_threshold must be between 0 and 1')
        return v


class BoundingBox(BaseModel):
    """Bounding box schema."""
    x: float
    y: float
    width: float
    height: float
    confidence: float
    label: str
    class_id: int


class DetectionResponse(BaseModel):
    """Response schema for object detection."""
    success: bool
    num_detections: int
    detections: List[BoundingBox]
    inference_time_ms: float
    image_shape: List[int]


@app.route('/health', methods=['GET'])
@monitor_api_request
def health_check():
    """
    Health check endpoint.
    
    Returns:
        JSON response with health status
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device)
    }), 200


@app.route('/metrics', methods=['GET'])
def metrics():
    """
    Prometheus metrics endpoint.
    
    Returns:
        Prometheus metrics in exposition format
    """
    return metrics_collector.get_metrics()


@app.route('/detect', methods=['POST'])
@monitor_api_request
def detect_objects():
    """
    Object detection endpoint.
    
    Expects:
        - multipart/form-data with 'image' file
        - Optional: confidence_threshold, max_detections as form fields
    
    Returns:
        JSON response with detected objects
    """
    try:
        # Validate request
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400
        
        image_file = request.files['image']
        
        # Validate file size
        max_size_mb = config['api']['max_upload_size_mb']
        image_file.seek(0, os.SEEK_END)
        file_size_mb = image_file.tell() / (1024 * 1024)
        image_file.seek(0)
        
        if file_size_mb > max_size_mb:
            return jsonify({
                'success': False,
                'error': f'File size exceeds {max_size_mb}MB limit'
            }), 400
        
        # Parse parameters
        confidence_threshold = float(request.form.get('confidence_threshold', 0.5))
        max_detections = int(request.form.get('max_detections', 100))
        
        # Validate parameters
        try:
            params = DetectionRequest(
                confidence_threshold=confidence_threshold,
                max_detections=max_detections
            )
        except ValueError as e:
            return jsonify({
                'success': False,
                'error': f'Invalid parameters: {str(e)}'
            }), 400
        
        # Load and preprocess image
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image)
        
        # Perform detection
        detections, inference_time = perform_detection(
            image_np,
            params.confidence_threshold,
            params.max_detections
        )
        
        # Build response
        response = DetectionResponse(
            success=True,
            num_detections=len(detections),
            detections=detections,
            inference_time_ms=inference_time,
            image_shape=list(image_np.shape)
        )
        
        logger.info(f"Detected {len(detections)} objects in {inference_time:.2f}ms")
        
        return jsonify(response.dict()), 200
    
    except Exception as e:
        logger.error(f"Detection error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def perform_detection(
    image: np.ndarray,
    confidence_threshold: float = 0.5,
    max_detections: int = 100
) -> tuple:
    """
    Perform object detection on image.
    
    Args:
        image: Input image (numpy array)
        confidence_threshold: Minimum confidence for detections
        max_detections: Maximum number of detections to return
        
    Returns:
        Tuple of (detections list, inference time in ms)
    """
    import time
    
    # Preprocess
    image_tensor = preprocessor.preprocess(image, normalize=True)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Inference
    start_time = time.time()
    
    with torch.no_grad():
        predictions = model([image_tensor])[0]
    
    inference_time = (time.time() - start_time) * 1000  # Convert to ms
    
    # Post-process predictions
    detections = []
    
    if 'boxes' in predictions:
        boxes = predictions['boxes'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        
        # Filter by confidence
        mask = scores >= confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]
        
        # Limit number of detections
        if len(boxes) > max_detections:
            indices = np.argsort(scores)[::-1][:max_detections]
            boxes = boxes[indices]
            scores = scores[indices]
            labels = labels[indices]
        
        # Convert to response format
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            
            detection = BoundingBox(
                x=float(x1),
                y=float(y1),
                width=float(x2 - x1),
                height=float(y2 - y1),
                confidence=float(score),
                label=get_label_name(int(label)),
                class_id=int(label)
            )
            
            detections.append(detection)
    
    return detections, inference_time


def get_label_name(class_id: int) -> str:
    """
    Get label name for class ID.
    
    Args:
        class_id: COCO class ID
        
    Returns:
        Label name
    """
    # COCO class names (simplified for demonstration)
    coco_classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    if 0 <= class_id < len(coco_classes):
        return coco_classes[class_id]
    else:
        return f'class_{class_id}'


@app.route('/info', methods=['GET'])
@monitor_api_request
def model_info():
    """
    Get model information.
    
    Returns:
        JSON response with model details
    """
    from src.utils import get_model_size_mb
    
    try:
        info = {
            'model_architecture': config['model']['architecture'],
            'num_classes': config['model']['num_classes'],
            'target_size_mb': config['model']['target_size_mb'],
            'target_map': config['model']['target_map'],
            'target_inference_ms': config['model']['target_inference_ms'],
            'device': str(device),
            'compression_enabled': {
                'distillation': config['compression']['distillation']['enabled'],
                'pruning': config['compression']['pruning']['enabled'],
                'quantization': config['compression']['quantization']['enabled']
            }
        }
        
        # Add actual model size if available
        if os.path.exists(model_path):
            info['actual_model_size_mb'] = get_model_size_mb(model_path)
        
        return jsonify(info), 200
    
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    host = config['api']['host']
    port = config['api']['port']
    debug = config['api']['debug']
    
    logger.info(f"Starting ProductionCV API on {host}:{port}")
    app.run(host=host, port=port, debug=debug)
