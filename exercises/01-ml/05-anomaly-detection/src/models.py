"""Model training with experiment framework for FraudShield

This module provides:
- Abstract AnomalyDetector interface for plug-and-play models
- Concrete implementations: IsolationForest, LocalOutlierFactor, Autoencoder (with TODOs)
- ExperimentRunner for comparing multiple detectors
- Immediate feedback with rich console output showing anomaly scores

Learning objectives:
1. Implement unsupervised anomaly detection algorithms
2. Compare detectors using plug-and-play registry pattern
3. Tune contamination thresholds for precision/recall trade-offs
4. See anomaly scores and metrics immediately after each model trains
5. Experiment with different algorithms and observe impact
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

logger = logging.getLogger("fraudshield")
console = Console()


@dataclass
class ModelConfig:
    """Configuration for model training."""
    contamination: float = 0.1  # Expected proportion of anomalies
    random_state: int = 42
    verbose: bool = True


class AnomalyDetector(ABC):
    """Abstract base class for all anomaly detectors.
    
    Provides common interface for plug-and-play experimentation.
    Subclasses implement train() and predict() methods.
    
    Key difference from regression: Models output binary predictions (0/1)
    and anomaly scores (higher = more anomalous).
    """
    
    def __init__(self, name: str):
        """Initialize detector with name for display."""
        self.name = name
        self.model = None
        self.metrics = {}
        self.threshold = None  # For reconstruction-based methods
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, config: ModelConfig) -> Dict[str, float]:
        """Train anomaly detector and return metrics with immediate console feedback.
        
        Args:
            X: Training features (standardized)
            y: Training labels (0=normal, 1=anomaly) - used only for validation
            config: Training configuration
        
        Returns:
            Dictionary with metrics: {"precision": float, "recall": float, 
                                      "f1": float, "roc_auc": float}
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make binary predictions on new data (0=normal, 1=anomaly)."""
        pass
    
    @abstractmethod
    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Get anomaly scores (higher = more anomalous)."""
        pass
    
    def save(self, path: str) -> None:
        """Save trained model to disk."""
        if self.model is None:
            raise ValueError("Cannot save untrained model")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        save_data = {"model": self.model, "threshold": self.threshold}
        joblib.dump(save_data, path)
        logger.info(f"Saved {self.name} to {path}")
    
    @classmethod
    def load(cls, path: str) -> "AnomalyDetector":
        """Load trained model from disk."""
        save_data = joblib.load(path)
        instance = cls.__new__(cls)
        instance.model = save_data["model"]
        instance.threshold = save_data.get("threshold")
        return instance


class IsolationForestDetector(AnomalyDetector):
    """Isolation Forest anomaly detector.
    
    Isolation Forest builds an ensemble of decision trees and identifies anomalies
    as points that require fewer splits to isolate. Anomalies are "easier to isolate"
    because they're far from the dense regions of normal data.
    
    Key parameters:
    - contamination: Expected proportion of anomalies (e.g., 0.1 = 10%)
    - n_estimators: Number of trees (more = more stable, but slower)
    """
    
    def __init__(self, n_estimators: int = 100):
        """Initialize Isolation Forest detector.
        
        Args:
            n_estimators: Number of isolation trees
        """
        super().__init__(f"IsolationForest (n={n_estimators})")
        self.n_estimators = n_estimators
    
    def train(self, X: pd.DataFrame, y: pd.Series, config: ModelConfig) -> Dict[str, float]:
        """TODO: Fit IsolationForest, compute metrics, and print feedback."""
        # TODO: Your implementation here
        raise NotImplementedError("Implement Isolation Forest training")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """TODO: Return binary predictions (0=normal, 1=anomaly)."""
        # TODO: Your implementation here
        raise NotImplementedError("Implement predict")
    
    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        """TODO: Return anomaly scores using decision_function."""
        # TODO: Your implementation here
        raise NotImplementedError("Implement predict_scores")


class LocalOutlierFactorDetector(AnomalyDetector):
    """Local Outlier Factor (LOF) anomaly detector.
    
    LOF measures the local deviation of a point's density compared to its neighbors.
    Points in low-density regions (far from neighbors) are considered anomalies.
    
    Key parameters:
    - n_neighbors: Number of neighbors to consider (smaller = more local)
    - contamination: Expected proportion of anomalies
    """
    
    def __init__(self, n_neighbors: int = 20):
        """Initialize LOF detector.
        
        Args:
            n_neighbors: Number of neighbors for density estimation
        """
        super().__init__(f"LOF (k={n_neighbors})")
        self.n_neighbors = n_neighbors
    
    def train(self, X: pd.DataFrame, y: pd.Series, config: ModelConfig) -> Dict[str, float]:
        """TODO: Fit LocalOutlierFactor with novelty=True, compute metrics, and print feedback."""
        # TODO: Your implementation here
        raise NotImplementedError("Implement LOF training")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make binary predictions (same implementation as IsolationForest)."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        y_pred_raw = self.model.predict(X)
        return (y_pred_raw == -1).astype(int)
    
    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Get anomaly scores (same implementation as IsolationForest)."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        scores = self.model.decision_function(X)
        return -scores


class AutoencoderDetector(AnomalyDetector):
    """Autoencoder-based anomaly detector.
    
    Autoencoder learns to reconstruct normal data. Anomalies have high
    reconstruction errors because they weren't well-represented in training.
    
    Architecture: Input → Encoder → Bottleneck → Decoder → Reconstructed Input
    
    Key parameters:
    - encoding_dim: Size of bottleneck layer (smaller = more compression)
    - epochs: Number of training epochs
    """
    
    def __init__(self, encoding_dim: int = 10, epochs: int = 50):
        """Initialize Autoencoder detector.
        
        Args:
            encoding_dim: Dimension of encoded representation
            epochs: Number of training epochs
        """
        super().__init__(f"Autoencoder (dim={encoding_dim}, e={epochs})")
        self.encoding_dim = encoding_dim
        self.epochs = epochs
    
    def train(self, X: pd.DataFrame, y: pd.Series, config: ModelConfig) -> Dict[str, float]:
        """TODO: Build autoencoder, train on normal data only, set threshold, compute metrics."""
        # TODO: Your implementation here
        raise NotImplementedError("Implement Autoencoder training")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """TODO: Reconstruct data and compare errors against threshold."""
        # TODO: Your implementation here
        raise NotImplementedError("Implement predict")
    
    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        """TODO: Return mean squared reconstruction errors as anomaly scores."""
        # TODO: Your implementation here
        raise NotImplementedError("Implement predict_scores")


class ExperimentRunner:
    """Run experiments with multiple anomaly detectors and compare results.
    
    Provides plug-and-play framework for trying different detectors:
    1. Register detectors to try
    2. Run all experiments with immediate feedback
    3. Print leaderboard sorted by F1 score
    
    Example:
        >>> runner = ExperimentRunner()
        >>> runner.register("IF-100", IsolationForestDetector(n_estimators=100))
        >>> runner.register("LOF-20", LocalOutlierFactorDetector(n_neighbors=20))
        >>> runner.register("AE", AutoencoderDetector(encoding_dim=10))
        >>> runner.run_experiment(X_train, y_train, ModelConfig())
        >>> runner.print_leaderboard()
    """
    
    def __init__(self):
        """Initialize empty experiment runner."""
        self.detectors: Dict[str, AnomalyDetector] = {}
        self.results: List[Dict[str, Any]] = []
    
    def register(self, name: str, detector: AnomalyDetector):
        """Register a detector to try in experiments.
        
        Args:
            name: Display name for results
            detector: AnomalyDetector instance to train
        """
        self.detectors[name] = detector
        console.print(f"Registered: {name}", style="dim")
    
    def run_experiment(self, X: pd.DataFrame, y: pd.Series, config: ModelConfig):
        """TODO: Train all registered detectors and store results."""
        # TODO: Your implementation here
        raise NotImplementedError("Implement experiment runner")
    
    def print_leaderboard(self):
        """TODO: Display sorted results table and highlight best detector."""
        # TODO: Your implementation here
        raise NotImplementedError("Implement leaderboard")
    
    def get_best_detector(self) -> AnomalyDetector:
        """Return detector with highest F1 score."""
        if not self.results:
            raise ValueError("No experiments run yet")
        best_result = max(self.results, key=lambda x: x["f1"])
        return self.detectors[best_result["model"]]
