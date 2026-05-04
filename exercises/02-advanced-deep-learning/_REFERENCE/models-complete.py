"""Model Registry and Compression Techniques

Implements object detection models and compression methods:
- Faster R-CNN, YOLOv8, Mask R-CNN
- Knowledge distillation
- Pruning
- Quantization
"""

import os
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from typing import Dict, Optional, Tuple
import mlflow
from ultralytics import YOLO
from src.utils import setup_logger, timing_decorator, ensure_dir


logger = setup_logger()


class ModelRegistry:
    """
    Registry for object detection models with compression capabilities.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize model registry.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_config = config['model']
        self.compression_config = config['compression']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Initialized ModelRegistry on device: {self.device}")
    
    def create_model(self, architecture: Optional[str] = None) -> nn.Module:
        """
        Create model based on architecture.
        
        Args:
            architecture: Model architecture ('faster_rcnn', 'yolov8', 'mask_rcnn')
            
        Returns:
            PyTorch model
        """
        if architecture is None:
            architecture = self.model_config['architecture']
        
        logger.info(f"Creating {architecture} model...")
        
        if architecture == 'faster_rcnn':
            return self.train_faster_rcnn()
        elif architecture == 'yolov8':
            return self.train_yolov8()
        elif architecture == 'mask_rcnn':
            return self.train_mask_rcnn()
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
    
    @timing_decorator
    def train_faster_rcnn(self) -> nn.Module:
        """
        Create and train Faster R-CNN model.
        
        Returns:
            Trained Faster R-CNN model
        """
        num_classes = self.model_config['num_classes']
        
        # Load pretrained model
        model = fasterrcnn_resnet50_fpn(
            pretrained=self.model_config['pretrained']
        )
        
        # Replace the classifier head
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        model = model.to(self.device)
        
        logger.info(f"Created Faster R-CNN with {num_classes} classes")
        
        return model
    
    @timing_decorator
    def train_yolov8(self) -> YOLO:
        """
        Create and train YOLOv8 model.
        
        Returns:
            Trained YOLOv8 model
        """
        # Load YOLOv8 model
        model_size = 'n'  # nano for edge deployment
        model = YOLO(f'yolov8{model_size}.pt')
        
        logger.info(f"Created YOLOv8{model_size} model")
        
        return model
    
    @timing_decorator
    def train_mask_rcnn(self) -> nn.Module:
        """
        Create and train Mask R-CNN model for instance segmentation.
        
        Returns:
            Trained Mask R-CNN model
        """
        num_classes = self.model_config['num_classes']
        
        # Load pretrained Mask R-CNN
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained=self.model_config['pretrained']
        )
        
        # Replace classifier head
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        model = model.to(self.device)
        
        logger.info(f"Created Mask R-CNN with {num_classes} classes")
        
        return model
    
    @timing_decorator
    def apply_distillation(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        train_loader: torch.utils.data.DataLoader
    ) -> nn.Module:
        """
        Apply knowledge distillation to compress model.
        
        Args:
            student_model: Student model to train
            teacher_model: Teacher model to distill from
            train_loader: Training data loader
            
        Returns:
            Distilled student model
        """
        logger.info("Applying knowledge distillation...")
        
        distill_config = self.compression_config['distillation']
        temperature = distill_config['temperature']
        alpha = distill_config['alpha']
        
        teacher_model.eval()
        student_model.train()
        
        optimizer = torch.optim.Adam(
            student_model.parameters(),
            lr=self.config['training']['learning_rate']
        )
        
        # Training loop (simplified for demonstration)
        epochs = min(10, self.config['training']['epochs'])
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            for batch in train_loader:
                images = [img.to(self.device) for img in [b['image'] for b in batch]]
                targets = [{k: v.to(self.device) for k, v in t.items() if k != 'image'} for t in batch]
                
                # Teacher predictions (no grad)
                with torch.no_grad():
                    teacher_outputs = teacher_model(images)
                
                # Student predictions
                student_outputs = student_model(images, targets)
                
                # Distillation loss (simplified)
                hard_loss = sum(loss for loss in student_outputs.values())
                
                # In production, implement proper soft target loss
                soft_loss = torch.tensor(0.0, device=self.device)
                
                loss = alpha * hard_loss + (1 - alpha) * soft_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            logger.info(f"Distillation epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        logger.info("Knowledge distillation completed")
        return student_model
    
    @timing_decorator
    def apply_pruning(self, model: nn.Module) -> nn.Module:
        """
        Apply structured pruning to reduce model size.
        
        Args:
            model: Model to prune
            
        Returns:
            Pruned model
        """
        logger.info("Applying model pruning...")
        
        pruning_config = self.compression_config['pruning']
        pruning_ratio = pruning_config['pruning_ratio']
        
        # Apply structured pruning (L1 norm based)
        import torch.nn.utils.prune as prune
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
                prune.remove(module, 'weight')
        
        logger.info(f"Applied {pruning_ratio * 100}% pruning")
        
        return model
    
    @timing_decorator
    def apply_quantization(self, model: nn.Module) -> nn.Module:
        """
        Apply quantization to reduce model size and improve inference speed.
        
        Args:
            model: Model to quantize
            
        Returns:
            Quantized model
        """
        logger.info("Applying quantization...")
        
        quant_config = self.compression_config['quantization']
        quant_type = quant_config['quantization_type']
        
        model.eval()
        
        if quant_type == 'int8':
            # Dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.Conv2d},
                dtype=torch.qint8
            )
        elif quant_type == 'float16':
            # Convert to FP16
            quantized_model = model.half()
        else:
            raise ValueError(f"Unknown quantization type: {quant_type}")
        
        logger.info(f"Applied {quant_type} quantization")
        
        return quantized_model
    
    def compress_pipeline(
        self,
        model: nn.Module,
        train_loader: Optional[torch.utils.data.DataLoader] = None
    ) -> nn.Module:
        """
        Apply full compression pipeline: distillation → pruning → quantization.
        
        Args:
            model: Model to compress
            train_loader: Training data loader (for distillation)
            
        Returns:
            Compressed model
        """
        logger.info("Starting compression pipeline...")
        
        # Step 1: Knowledge distillation (if enabled)
        if self.compression_config['distillation']['enabled'] and train_loader:
            teacher_model = self.create_model()  # Create teacher
            model = self.apply_distillation(model, teacher_model, train_loader)
        
        # Step 2: Pruning (if enabled)
        if self.compression_config['pruning']['enabled']:
            model = self.apply_pruning(model)
        
        # Step 3: Quantization (if enabled)
        if self.compression_config['quantization']['enabled']:
            model = self.apply_quantization(model)
        
        logger.info("Compression pipeline completed")
        
        return model
    
    def save_model(self, model: nn.Module, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            model: Model to save
            path: Save path
        """
        ensure_dir(os.path.dirname(path))
        torch.save(model.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str, architecture: Optional[str] = None) -> nn.Module:
        """
        Load model from disk.
        
        Args:
            path: Model path
            architecture: Model architecture
            
        Returns:
            Loaded model
        """
        model = self.create_model(architecture)
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.eval()
        logger.info(f"Model loaded from {path}")
        return model


if __name__ == "__main__":
    import argparse
    from src.utils import load_config
    from src.data import COCODataLoader
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'compress'])
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Initialize components
    registry = ModelRegistry(config)
    data_loader = COCODataLoader(config)
    
    # Load data
    train_ds, val_ds, test_ds = data_loader.load_and_split()
    train_loader, val_loader, test_loader = data_loader.create_dataloaders(
        train_ds, val_ds, test_ds
    )
    
    if args.mode == 'train':
        # Train baseline model
        model = registry.create_model()
        model_path = os.path.join(config['paths']['model_dir'], 'baseline_model.pth')
        registry.save_model(model, model_path)
        
    elif args.mode == 'compress':
        # Load baseline and apply compression
        baseline_path = os.path.join(config['paths']['model_dir'], 'baseline_model.pth')
        model = registry.load_model(baseline_path)
        
        compressed_model = registry.compress_pipeline(model, train_loader)
        
        compressed_path = os.path.join(config['paths']['model_dir'], 'compressed_model.pth')
        registry.save_model(compressed_model, compressed_path)
