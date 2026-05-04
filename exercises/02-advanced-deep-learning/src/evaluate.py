"""Model Evaluation and Benchmarking

Implements mAP calculation, inference latency benchmarking, and model size tracking.
"""

import os
import time
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import mlflow
from src.utils import setup_logger, timing_decorator, get_model_size_mb


logger = setup_logger()


class CVEvaluator:
    """
    Comprehensive evaluator for object detection and instance segmentation models.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize evaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_map = config['model']['target_map']
        self.target_inference_ms = config['model']['target_inference_ms']
        self.target_size_mb = config['model']['target_size_mb']
        
        logger.info("Initialized CVEvaluator")
    
    @timing_decorator
    def calculate_map(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        iou_threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Calculate mean Average Precision (mAP).
        
        Args:
            model: Detection model
            dataloader: Validation/test data loader
            iou_threshold: IoU threshold for positive detection
            
        Returns:
            Dictionary with mAP metrics
        """
        logger.info(f"Calculating mAP @ IoU={iou_threshold}...")
        
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                images = [img['image'].to(self.device) for img in batch]
                targets = [{k: v.to(self.device) for k, v in t.items() if k != 'image'} 
                          for t in batch]
                
                # Get predictions
                predictions = model(images)
                
                all_predictions.extend(predictions)
                all_targets.extend(targets)
        
        # Calculate AP for each class
        ap_per_class = self._compute_ap_per_class(
            all_predictions,
            all_targets,
            iou_threshold
        )
        
        # Calculate mAP
        map_value = np.mean([ap for ap in ap_per_class.values() if ap >= 0])
        
        metrics = {
            'mAP': map_value,
            'mAP@0.5': map_value,
            'num_classes': len(ap_per_class)
        }
        
        logger.info(f"mAP: {map_value:.4f}")
        
        return metrics
    
    def _compute_ap_per_class(
        self,
        predictions: List[Dict],
        targets: List[Dict],
        iou_threshold: float
    ) -> Dict[int, float]:
        """
        Compute Average Precision for each class.
        
        Args:
            predictions: List of prediction dictionaries
            targets: List of target dictionaries
            iou_threshold: IoU threshold
            
        Returns:
            Dictionary mapping class_id to AP
        """
        # Group predictions and targets by class
        class_preds = defaultdict(list)
        class_targets = defaultdict(list)
        
        for pred in predictions:
            if 'boxes' in pred and 'labels' in pred and 'scores' in pred:
                boxes = pred['boxes'].cpu().numpy()
                labels = pred['labels'].cpu().numpy()
                scores = pred['scores'].cpu().numpy()
                
                for box, label, score in zip(boxes, labels, scores):
                    class_preds[label].append({
                        'box': box,
                        'score': score
                    })
        
        for target in targets:
            if 'boxes' in target and 'labels' in target:
                boxes = target['boxes'].cpu().numpy()
                labels = target['labels'].cpu().numpy()
                
                for box, label in zip(boxes, labels):
                    class_targets[label].append({'box': box})
        
        # Calculate AP for each class
        ap_per_class = {}
        
        for class_id in set(list(class_preds.keys()) + list(class_targets.keys())):
            preds = class_preds.get(class_id, [])
            targets = class_targets.get(class_id, [])
            
            if len(preds) == 0 or len(targets) == 0:
                ap_per_class[class_id] = 0.0
                continue
            
            # Sort predictions by score
            preds = sorted(preds, key=lambda x: x['score'], reverse=True)
            
            # Calculate precision and recall
            tp = np.zeros(len(preds))
            fp = np.zeros(len(preds))
            
            matched_targets = set()
            
            for i, pred in enumerate(preds):
                best_iou = 0
                best_target_idx = -1
                
                for j, target in enumerate(targets):
                    if j in matched_targets:
                        continue
                    
                    iou = self._calculate_iou(pred['box'], target['box'])
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_target_idx = j
                
                if best_iou >= iou_threshold:
                    tp[i] = 1
                    matched_targets.add(best_target_idx)
                else:
                    fp[i] = 1
            
            # Calculate cumulative precision and recall
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            
            recalls = tp_cumsum / len(targets)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
            
            # Calculate AP using 11-point interpolation
            ap = self._calculate_ap_11point(recalls, precisions)
            ap_per_class[class_id] = ap
        
        return ap_per_class
    
    @staticmethod
    def _calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Calculate Intersection over Union (IoU) between two boxes.
        
        Args:
            box1: First bounding box [x1, y1, x2, y2] or [x, y, w, h]
            box2: Second bounding box [x1, y1, x2, y2] or [x, y, w, h]
            
        Returns:
            IoU value
        """
        # Convert to [x1, y1, x2, y2] format if needed
        if len(box1) == 4 and len(box2) == 4:
            # Assume COCO format [x, y, w, h]
            box1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
            box2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]
        
        # Calculate intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Calculate union
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    @staticmethod
    def _calculate_ap_11point(recalls: np.ndarray, precisions: np.ndarray) -> float:
        """
        Calculate Average Precision using 11-point interpolation.
        
        Args:
            recalls: Recall values
            precisions: Precision values
            
        Returns:
            Average Precision
        """
        ap = 0.0
        
        for t in np.linspace(0, 1, 11):
            mask = recalls >= t
            if np.any(mask):
                ap += np.max(precisions[mask])
        
        return ap / 11
    
    @timing_decorator
    def benchmark_inference(
        self,
        model: torch.nn.Module,
        input_size: Tuple[int, int] = (640, 640),
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark model inference latency.
        
        Args:
            model: Model to benchmark
            input_size: Input image size (height, width)
            num_iterations: Number of inference iterations
            warmup_iterations: Number of warmup iterations
            
        Returns:
            Dictionary with latency metrics
        """
        logger.info(f"Benchmarking inference latency ({num_iterations} iterations)...")
        
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, *input_size).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = model([dummy_input])
        
        # Benchmark
        latencies = []
        
        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.time()
                _ = model([dummy_input])
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                latency = (time.time() - start) * 1000  # Convert to ms
                latencies.append(latency)
        
        metrics = {
            'mean_latency_ms': np.mean(latencies),
            'median_latency_ms': np.median(latencies),
            'std_latency_ms': np.std(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'throughput_fps': 1000 / np.mean(latencies)
        }
        
        logger.info(f"Mean latency: {metrics['mean_latency_ms']:.2f}ms "
                   f"({metrics['throughput_fps']:.1f} FPS)")
        
        return metrics
    
    def evaluate_compression(
        self,
        model_path: str,
        dataloader: torch.utils.data.DataLoader,
        model: Optional[torch.nn.Module] = None
    ) -> Dict[str, any]:
        """
        Comprehensive evaluation of compressed model.
        
        Args:
            model_path: Path to saved model
            dataloader: Validation data loader
            model: Optional model instance (if not provided, load from path)
            
        Returns:
            Dictionary with all evaluation metrics
        """
        logger.info("Evaluating compressed model...")
        
        # Get model size
        model_size_mb = get_model_size_mb(model_path)
        
        # Load model if not provided
        if model is None:
            from src.models import ModelRegistry
            registry = ModelRegistry(self.config)
            model = registry.load_model(model_path)
        
        # Calculate mAP
        map_metrics = self.calculate_map(model, dataloader)
        
        # Benchmark inference
        inference_metrics = self.benchmark_inference(model)
        
        # Check if targets are met
        targets_met = {
            'size_target_met': model_size_mb <= self.target_size_mb,
            'map_target_met': map_metrics['mAP'] >= self.target_map,
            'latency_target_met': inference_metrics['mean_latency_ms'] <= self.target_inference_ms
        }
        
        # Combine all metrics
        results = {
            'model_size_mb': model_size_mb,
            'target_size_mb': self.target_size_mb,
            **map_metrics,
            **inference_metrics,
            **targets_met,
            'all_targets_met': all(targets_met.values())
        }
        
        # Log summary
        logger.info("=" * 60)
        logger.info("Compression Evaluation Results:")
        logger.info(f"  Model Size: {model_size_mb:.2f} MB (target: {self.target_size_mb} MB) "
                   f"{'✓' if targets_met['size_target_met'] else '✗'}")
        logger.info(f"  mAP: {map_metrics['mAP']:.4f} (target: {self.target_map}) "
                   f"{'✓' if targets_met['map_target_met'] else '✗'}")
        logger.info(f"  Latency: {inference_metrics['mean_latency_ms']:.2f} ms "
                   f"(target: {self.target_inference_ms} ms) "
                   f"{'✓' if targets_met['latency_target_met'] else '✗'}")
        logger.info("=" * 60)
        
        return results


if __name__ == "__main__":
    import argparse
    from src.utils import load_config
    from src.data import COCODataLoader
    from src.models import ModelRegistry
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Initialize components
    evaluator = CVEvaluator(config)
    data_loader = COCODataLoader(config)
    
    # Load data
    _, val_ds, test_ds = data_loader.load_and_split()
    _, val_loader, test_loader = data_loader.create_dataloaders(
        val_ds, val_ds, test_ds
    )
    
    if args.benchmark:
        # Full evaluation
        results = evaluator.evaluate_compression(args.model_path, test_loader)
        
        # Log to MLflow
        mlflow.log_metrics(results)
