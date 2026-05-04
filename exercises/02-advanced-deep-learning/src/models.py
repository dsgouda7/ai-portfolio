"""Advanced Deep Learning Model Training with Transfer Learning

This module provides:
- Abstract DeepModel base class for plug-and-play architectures
- Concrete implementations: ResNet, Transformer, EfficientNet (with TODOs)
- ExperimentRunner for comparing deep learning models
- Immediate feedback with rich console output and training curves

Learning objectives:
1. Implement ResNet with residual connections and skip connections
2. Build Transformer with self-attention and positional encoding
3. Apply transfer learning (freeze/unfreeze layers, fine-tuning)
4. Use advanced training techniques (mixed precision, gradient accumulation)
5. Compare architectures using plug-and-play registry pattern
6. Monitor training with real-time metrics and loss curves

Advanced concepts:
- Residual connections solve vanishing gradients
- Skip connections enable deeper networks (100+ layers)
- Self-attention captures long-range dependencies
- Transfer learning reuses pretrained ImageNet weights
- Mixed precision (FP16) speeds up training 2-3x
- Gradient accumulation simulates larger batch sizes
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import torchvision.models as models

logger = logging.getLogger("deeplearning")
console = Console()


@dataclass
class TrainingConfig:
    """Configuration for deep learning training."""
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    early_stopping_patience: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    random_state: int = 42


class DeepModel(ABC):
    """Abstract base class for all deep learning models.
    
    Provides common interface for plug-and-play experimentation.
    Subclasses implement build_model(), train(), and predict() methods.
    """
    
    def __init__(self, name: str, num_classes: int = 10):
        """Initialize deep model with name for display.
        
        Args:
            name: Model name for leaderboard display
            num_classes: Number of output classes
        """
        self.name = name
        self.num_classes = num_classes
        self.model = None
        self.metrics = {}
        self.training_history = {"train_loss": [], "val_loss": [], "val_acc": []}
    
    @abstractmethod
    def build_model(self) -> nn.Module:
        """Build and return the neural network architecture.
        
        Returns:
            PyTorch nn.Module
        """
        pass
    
    @abstractmethod
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: TrainingConfig
    ) -> Dict[str, float]:
        """Train model and return metrics with immediate console feedback.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
        
        Returns:
            Dictionary with metrics: {
                "val_loss": float,
                "val_acc": float,
                "train_loss": float,
                "best_epoch": int
            }
        """
        pass
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Make predictions on new data.
        
        Args:
            X: Input tensor [batch_size, channels, height, width]
        
        Returns:
            Predictions [batch_size, num_classes]
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
        return outputs
    
    def save(self, path: str) -> None:
        """Save trained model to disk."""
        if self.model is None:
            raise ValueError("Cannot save untrained model")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'metrics': self.metrics,
            'training_history': self.training_history
        }, path)
        logger.info(f"Saved {self.name} to {path}")
    
    @classmethod
    def load(cls, path: str, num_classes: int) -> "DeepModel":
        """Load trained model from disk."""
        checkpoint = torch.load(path)
        instance = cls.__new__(cls)
        instance.build_model()
        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance.metrics = checkpoint.get('metrics', {})
        instance.training_history = checkpoint.get('training_history', {})
        return instance


class ResNetModel(DeepModel):
    """ResNet-50 with residual connections and transfer learning.
    
    ResNet Architecture (He et al., 2015):
    - Residual blocks: F(x) + x (identity shortcut)
    - Skip connections allow gradients to flow backward
    - Enables training 50-200+ layer networks
    - Batch normalization after each conv layer
    
    Transfer Learning Strategy:
    1. Load pretrained ImageNet weights (1.28M images, 1000 classes)
    2. Freeze early layers (feature extractors) → faster training
    3. Replace final FC layer for our num_classes
    4. Fine-tune last few layers on our dataset
    """
    
    def __init__(self, num_classes: int = 10, pretrained: bool = True, freeze_layers: int = 3):
        """Initialize ResNet model.
        
        Args:
            num_classes: Number of output classes
            pretrained: Use ImageNet pretrained weights
            freeze_layers: Number of initial layer groups to freeze (0-4)
                          0 = train all, 4 = freeze all except final FC
        """
        super().__init__(f"ResNet-50 (freeze={freeze_layers})", num_classes)
        self.pretrained = pretrained
        self.freeze_layers = freeze_layers
    
    def build_model(self) -> nn.Module:
        """TODO: Build ResNet-50 with transfer learning and frozen layers."""
        # TODO: Your implementation here
        raise NotImplementedError("Implement ResNet-50 with transfer learning")
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: TrainingConfig
    ) -> Dict[str, float]:
        """TODO: Train ResNet with mixed precision and gradient accumulation."""
        # TODO: Your implementation here
        raise NotImplementedError("Implement ResNet training")


class TransformerModel(DeepModel):
    """Vision Transformer (ViT) with self-attention and positional encoding.
    
    Transformer Architecture (Vaswani et al., 2017; Dosovitskiy et al., 2020):
    - Self-attention: learns relationships between all image patches
    - Positional encoding: adds position information (no conv structure)
    - Multi-head attention: attends to different representation subspaces
    - Feed-forward networks: per-patch transformations
    
    How it works:
    1. Split image into 16×16 patches (224×224 → 14×14 = 196 patches)
    2. Flatten patches and project to embedding dimension
    3. Add positional embeddings (learnable or sinusoidal)
    4. Pass through N transformer encoder layers
    5. Use [CLS] token embedding for classification
    
    Key difference from ResNet:
    - ResNet: local receptive fields (conv kernels)
    - Transformer: global receptive field (attention across all patches)
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
        freeze_encoder: bool = False
    ):
        """Initialize Vision Transformer.
        
        Args:
            num_classes: Number of output classes
            pretrained: Use pretrained weights (ViT-B/16 on ImageNet-21k)
            freeze_encoder: Freeze transformer encoder blocks
        """
        super().__init__(f"ViT-B/16 (freeze={freeze_encoder})", num_classes)
        self.pretrained = pretrained
        self.freeze_encoder = freeze_encoder
    
    def build_model(self) -> nn.Module:
        """TODO: Build Vision Transformer (ViT-B/16) with transfer learning."""
        # TODO: Your implementation here
        raise NotImplementedError("Implement Vision Transformer")
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: TrainingConfig
    ) -> Dict[str, float]:
        """TODO: Implement training with learning rate warm-up."""
        # TODO: Your implementation here
        raise NotImplementedError("Implement Transformer training")


class EfficientNetModel(DeepModel):
    """EfficientNet-B0 with compound scaling and transfer learning.
    
    EfficientNet (Tan & Le, 2019):
    - Compound scaling: balance depth, width, resolution
    - Mobile inverted bottleneck (MBConv) blocks
    - Squeeze-and-excitation (SE) blocks for channel attention
    """
    
    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        """Initialize EfficientNet-B0."""
        super().__init__("EfficientNet-B0", num_classes)
        self.pretrained = pretrained
    
    def build_model(self) -> nn.Module:
        """TODO: Build EfficientNet-B0 with transfer learning."""
        # TODO: Your implementation here
        raise NotImplementedError("Implement EfficientNet-B0")
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: TrainingConfig
    ) -> Dict[str, float]:
        """TODO: Reuse ResNet.train() implementation."""
        # TODO: Your implementation here
        raise NotImplementedError("Implement EfficientNet training")


class ExperimentRunner:
    """Experiment runner for comparing deep learning models."""
    
    def __init__(self):
        """Initialize experiment runner."""
        self.models: Dict[str, DeepModel] = {}
        self.results: List[Dict[str, Any]] = []
    
    def register(self, name: str, model: DeepModel) -> None:
        """Register a model for comparison."""
        self.models[name] = model
        console.print(f"  → Registered: {name}", style="dim")
    
    def run_experiment(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: TrainingConfig
    ) -> None:
        """TODO: Run all registered models and collect results."""
        # TODO: Your implementation here
        raise NotImplementedError("Implement experiment runner")
    
    def print_leaderboard(self) -> None:
        """TODO: Print beautiful leaderboard with Rich table."""
        # TODO: Your implementation here
        raise NotImplementedError("Implement leaderboard printing")
    
    def get_best_model(self) -> DeepModel:
        """Get the best performing model."""
        if not self.results:
            raise ValueError("No results available. Run experiment first.")
        return self.results[0]["model"]


# ============================================
# UTILITY FUNCTIONS
# ============================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_batch_norm(model: nn.Module) -> None:
    """Freeze batch normalization layers during fine-tuning."""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
