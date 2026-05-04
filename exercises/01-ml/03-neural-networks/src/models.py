"""Neural network training with experiment framework for UnifiedAI

This module provides:
- Abstract NeuralNetwork interface for plug-and-play models
- Concrete implementations: MLPClassifier, CNNClassifier (with TODOs)
- ExperimentRunner for comparing multiple neural architectures
- Immediate feedback with rich console output showing loss/accuracy per epoch

Learning objectives:
1. Implement MLP (fully connected) and CNN architectures
2. Understand backpropagation and gradient descent in practice
3. Monitor training with live epoch-by-epoch feedback
4. Compare different architectures using registry pattern
5. Master neural network hyperparameters (learning rate, batch size, dropout)
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from tensorflow import keras
from tensorflow.keras import layers, callbacks

logger = logging.getLogger("unifiedai")
console = Console()

# Suppress TensorFlow logs (only show errors)
tf.get_logger().setLevel('ERROR')


@dataclass
class ModelConfig:
    """Configuration for neural network training."""
    batch_size: int = 32
    epochs: int = 50
    early_stopping_patience: int = 5
    random_state: int = 42
    verbose: bool = True


class NeuralNetwork(ABC):
    """Abstract base class for all neural network classifiers.
    
    Provides common interface for plug-and-play experimentation.
    Subclasses implement build_model() and train() methods.
    
    Key concepts:
    - Forward pass: Input → hidden layers → output (predictions)
    - Backpropagation: Compute gradients by chain rule
    - Gradient descent: Update weights to minimize loss
    """
    
    def __init__(self, name: str):
        """Initialize neural network with name for display."""
        self.name = name
        self.model = None
        self.metrics = {}
        self.history = None
    
    @abstractmethod
    def build_model(self, input_shape: tuple, n_classes: int) -> keras.Model:
        """Build neural network architecture.
        
        Args:
            input_shape: Shape of input data (excluding batch dimension)
            n_classes: Number of output classes
        
        Returns:
            Compiled Keras model
        """
        pass
    
    @abstractmethod
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: np.ndarray, 
        y_val: np.ndarray,
        config: ModelConfig
    ) -> Dict[str, float]:
        """Train model and return metrics with immediate console feedback.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            config: Training configuration
        
        Returns:
            Dictionary with metrics: {
                "train_loss": float, 
                "train_accuracy": float, 
                "val_loss": float, 
                "val_accuracy": float,
                "epochs_trained": int
            }
        """
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X, verbose=0)
    
    def save(self, path: str) -> None:
        """Save trained model to disk."""
        if self.model is None:
            raise ValueError("Cannot save untrained model")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        logger.info(f"Saved {self.name} to {path}")
    
    @classmethod
    def load(cls, path: str) -> "NeuralNetwork":
        """Load trained model from disk."""
        model = keras.models.load_model(path)
        instance = cls.__new__(cls)
        instance.model = model
        return instance


class MLPClassifier(NeuralNetwork):
    """Multi-Layer Perceptron (fully connected neural network).
    
    Architecture:
        Input → Dense → ReLU → Dropout → Dense → ReLU → Dropout → ... → Output (Softmax)
    
    Key concepts:
    - Dense layer: Fully connected (every input connects to every output)
    - ReLU activation: f(x) = max(0, x) — introduces non-linearity
    - Dropout: Randomly drop neurons during training — prevents overfitting
    - Softmax output: Converts logits to probabilities (sum = 1.0)
    
    Mathematical details:
    - Forward pass: y = softmax(W_n * ReLU(W_n-1 * ... * ReLU(W_1 * x)))
    - Loss: Cross-entropy = -sum(y_true * log(y_pred))
    - Backprop: Compute ∂Loss/∂W_i for each layer using chain rule
    """
    
    def __init__(
        self, 
        architecture: List[int] = [128, 64, 32],
        activation: str = 'relu',
        dropout: float = 0.3,
        learning_rate: float = 0.001
    ):
        """Initialize MLP classifier.
        
        Args:
            architecture: List of hidden layer sizes (e.g., [128, 64, 32])
            activation: Activation function ('relu', 'tanh', 'sigmoid')
            dropout: Dropout rate (0.0 = no dropout, 0.5 = drop 50% of neurons)
            learning_rate: Learning rate for Adam optimizer
        """
        arch_str = "→".join(map(str, architecture))
        super().__init__(f"MLP ({arch_str}, dropout={dropout})")
        self.architecture = architecture
        self.activation = activation
        self.dropout = dropout
        self.learning_rate = learning_rate
    
    def build_model(self, input_shape: tuple, n_classes: int) -> keras.Model:
        """TODO: Build Sequential model with Dense+Dropout layers, compile with Adam optimizer."""
        # TODO: Your implementation here
        raise NotImplementedError("Implement MLP architecture - see TODO above")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: ModelConfig
    ) -> Dict[str, float]:
        """TODO: Build model, train with early stopping + LiveFeedback callback, return metrics."""
        # TODO: Your implementation here
        raise NotImplementedError("Implement MLP training - see TODO above")


class CNNClassifier(NeuralNetwork):
    """Convolutional Neural Network for 1D feature sequences.
    
    Architecture:
        Input → Conv1D → MaxPool → Dropout → Conv1D → MaxPool → Dropout → 
        GlobalAvgPool → Dense → Dropout → Output (Softmax)
    
    Key concepts:
    - Conv1D: Sliding window that learns local patterns
    - MaxPooling: Downsample by taking maximum in each window
    - Global Average Pooling: Reduce each feature map to single value
    - CNNs excel at learning hierarchical spatial patterns
    
    Mathematical details:
    - Convolution: (f * g)[n] = sum_m f[m] * g[n - m]
    - Each filter learns different pattern (edges, gradients, etc.)
    - Deeper layers combine lower-level patterns into complex features
    """
    
    def __init__(
        self,
        filters: List[int] = [64, 32, 16],
        kernel_size: int = 3,
        pool_size: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 0.001
    ):
        """Initialize CNN classifier.
        
        Args:
            filters: List of filter counts per conv layer (e.g., [64, 32, 16])
            kernel_size: Size of convolution window
            pool_size: Size of max pooling window
            dropout: Dropout rate
            learning_rate: Learning rate for Adam optimizer
        """
        filters_str = "→".join(map(str, filters))
        super().__init__(f"CNN (filters={filters_str}, kernel={kernel_size})")
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dropout = dropout
        self.learning_rate = learning_rate
    
    def build_model(self, input_shape: tuple, n_classes: int) -> keras.Model:
        """TODO: Build Sequential CNN with Conv1D+MaxPool+Dropout blocks, GlobalAvgPool, Dense output."""
        # TODO: Your implementation here
        raise NotImplementedError("Implement CNN architecture - see TODO above")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: ModelConfig
    ) -> Dict[str, float]:
        """TODO: Reshape input to (samples, features, 1), then train like MLP with early stopping."""
        # TODO: Your implementation here
        raise NotImplementedError("Implement CNN training - see TODO above")


class ExperimentRunner:
    """Run experiments with multiple neural networks and compare results.
    
    Provides plug-and-play framework for trying different architectures:
    1. Register neural networks to try
    2. Run all experiments with live training feedback
    3. Print leaderboard sorted by validation accuracy
    
    Example:
        >>> runner = ExperimentRunner()
        >>> runner.register("Small MLP", MLPClassifier(architecture=[64, 32]))
        >>> runner.register("Deep MLP", MLPClassifier(architecture=[128, 64, 32, 16]))
        >>> runner.register("CNN", CNNClassifier(filters=[64, 32]))
        >>> runner.run_experiment(X_train, y_train, X_val, y_val, ModelConfig())
        >>> runner.print_leaderboard()
    """
    
    def __init__(self):
        """Initialize empty experiment runner."""
        self.networks: Dict[str, NeuralNetwork] = {}
        self.results: List[Dict[str, Any]] = []
    
    def register(self, name: str, network: NeuralNetwork):
        """Register a neural network to try in experiments.
        
        Args:
            name: Display name for results
            network: NeuralNetwork instance to train
        """
        self.networks[name] = network
        console.print(f"Registered: {name}", style="dim")
    
    def run_experiment(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: ModelConfig
    ):
        """TODO: Loop through registered networks, train each, store results."""
        # TODO: Your implementation here
        raise NotImplementedError("Implement experiment runner - see TODO above")
    
    def print_leaderboard(self):
        """TODO: Create rich Table, sort results by val_accuracy, print with best model highlighted."""
        # TODO: Your implementation here
        raise NotImplementedError("Implement leaderboard - see TODO above")
    
    def get_best_model(self) -> NeuralNetwork:
        """Return neural network with highest validation accuracy."""
        if not self.results:
            raise ValueError("No experiments run yet")
        best_result = max(self.results, key=lambda x: x["val_accuracy"])
        return self.networks[best_result["model"]]
